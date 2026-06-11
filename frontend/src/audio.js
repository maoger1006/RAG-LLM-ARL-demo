// Browser-side audio: microphone / system-audio capture streamed to the
// backend as 16 kHz PCM16 mono, and playback of server-synthesized TTS.
//
// getUserMedia requires a secure context (https or http://localhost). When
// the backend runs on a remote machine, access it through an SSH tunnel:
//   ssh -L 8800:localhost:8800 user@server   ->  open http://localhost:8800

const TARGET_RATE = 16000
const FLUSH_SAMPLES = 4096 // ~0.25 s at 16 kHz after resampling

const WORKLET_CODE = `
class PCMForwarder extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0]
    if (ch) this.port.postMessage(ch.slice(0))
    return true
  }
}
registerProcessor('pcm-forwarder', PCMForwarder)
`

function downsampleToInt16(float32, fromRate) {
  const ratio = fromRate / TARGET_RATE
  const outLength = Math.floor(float32.length / ratio)
  const out = new Int16Array(outLength)
  for (let i = 0; i < outLength; i++) {
    const pos = i * ratio
    const lo = Math.floor(pos)
    const hi = Math.min(lo + 1, float32.length - 1)
    const frac = pos - lo
    let sample = float32[lo] * (1 - frac) + float32[hi] * frac
    sample = Math.max(-1, Math.min(1, sample))
    out[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff
  }
  return out
}

export function audioSupported() {
  return Boolean(navigator.mediaDevices?.getUserMedia && window.AudioContext)
}

async function wrapStream(stream, onChunk) {
  const ctx = new AudioContext()
  const workletUrl = URL.createObjectURL(
    new Blob([WORKLET_CODE], { type: 'application/javascript' }),
  )
  await ctx.audioWorklet.addModule(workletUrl)
  URL.revokeObjectURL(workletUrl)

  const source = ctx.createMediaStreamSource(stream)
  const node = new AudioWorkletNode(ctx, 'pcm-forwarder')

  let muted = false
  let buffered = []
  let bufferedSamples = 0
  const flushThreshold = Math.ceil((FLUSH_SAMPLES * ctx.sampleRate) / TARGET_RATE)

  node.port.onmessage = (e) => {
    if (muted) return
    buffered.push(e.data)
    bufferedSamples += e.data.length
    if (bufferedSamples >= flushThreshold) {
      const merged = new Float32Array(bufferedSamples)
      let offset = 0
      for (const part of buffered) {
        merged.set(part, offset)
        offset += part.length
      }
      buffered = []
      bufferedSamples = 0
      onChunk(downsampleToInt16(merged, ctx.sampleRate))
    }
  }

  source.connect(node)

  return {
    setMuted(value) { muted = value },
    stop() {
      try { node.disconnect(); source.disconnect() } catch { /* already gone */ }
      ctx.close().catch(() => {})
      stream.getTracks().forEach((t) => t.stop())
    },
  }
}

/** Capture the user's microphone. */
export async function startMicCapture(onChunk) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  })
  return wrapStream(stream, onChunk)
}

/** Capture shared system/tab audio (the "Speaker" channel). Chrome only:
 *  the user must tick "Share audio" in the screen-share dialog. */
export async function startSystemCapture(onChunk) {
  const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true })
  if (!stream.getAudioTracks().length) {
    stream.getTracks().forEach((t) => t.stop())
    throw new Error('No audio shared — tick "Share audio" in the share dialog (Chrome/Edge).')
  }
  return wrapStream(stream, onChunk)
}

/**
 * One audio session = one /ws/audio connection + one live capture.
 * `capture` is startMicCapture or startSystemCapture.
 */
export function openAudioSession({ target, channel = 1, capture, onError }) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${location.host}/ws/audio`)
  ws.binaryType = 'arraybuffer'

  let handle = null
  let stopped = false

  ws.onopen = async () => {
    ws.send(JSON.stringify({ target, channel }))
    try {
      handle = await capture((int16) => {
        if (ws.readyState === WebSocket.OPEN) ws.send(int16.buffer)
      })
      if (stopped) handle.stop() // stopped before capture finished starting
    } catch (err) {
      onError?.(err)
      ws.close()
    }
  }
  ws.onerror = () => onError?.(new Error('audio websocket error'))

  return {
    setMuted(value) { handle?.setMuted(value) },
    stop() {
      stopped = true
      handle?.stop()
      if (ws.readyState === WebSocket.OPEN) {
        try { ws.send('end') } catch { /* closing anyway */ }
      }
      ws.close()
    },
  }
}

/** Sequential player for server TTS clips ({type:"speak"} events). */
export function createTtsPlayer(onPlayingChange) {
  const queue = []
  let playing = false
  let current = null

  const playNext = () => {
    if (!queue.length) {
      playing = false
      current = null
      onPlayingChange?.(false)
      return
    }
    if (!playing) {
      playing = true
      onPlayingChange?.(true)
    }
    current = new Audio(queue.shift())
    current.onended = playNext
    current.onerror = playNext
    current.play().catch(playNext)
  }

  return {
    enqueue(url) {
      queue.push(url)
      if (!playing) playNext()
    },
    get playing() { return playing },
    stopAll() {
      queue.length = 0
      if (current) { current.pause(); current = null }
      playing = false
      onPlayingChange?.(false)
    },
  }
}
