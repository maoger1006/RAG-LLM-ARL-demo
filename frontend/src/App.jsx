import { useCallback, useEffect, useRef, useState } from 'react'
import * as api from './api.js'
import { useEvents } from './useEvents.js'
import {
  audioSupported, createTtsPlayer, openAudioSession,
  startMicCapture, startSystemCapture,
} from './audio.js'
import ControlBar from './components/ControlBar.jsx'
import MessagePanel from './components/MessagePanel.jsx'
import QuestionBar from './components/QuestionBar.jsx'
import RoiModal from './components/RoiModal.jsx'
import { SparklesIcon } from './components/icons.jsx'

const EMPTY_PANELS = { transcripts: [], qa: [], video: [], summary: [] }

export default function App() {
  const [panels, setPanels] = useState(EMPTY_PANELS)
  const [settings, setSettings] = useState({
    response_type: 'concise mode',
    read_aloud: false,
    correct_content: false,
    use_retrieval: true,
    stt_running: false,
    conversation_running: false,
  })
  const [question, setQuestion] = useState('')
  const [roiModal, setRoiModal] = useState(null) // { filename, mode: 'parse' | 'process' }
  const [voiceActive, setVoiceActive] = useState(false)
  const [speakerSharing, setSpeakerSharing] = useState(false)
  const [busy, setBusy] = useState(false)

  // audio sessions live in refs: they hold WebSocket + AudioContext handles
  const sttSessionRef = useRef(null)
  const speakerSessionRef = useRef(null)
  const convSessionRef = useRef(null)
  const voiceSessionRef = useRef(null)
  const voiceDoneResolver = useRef(null)
  const playerRef = useRef(null)
  if (!playerRef.current) {
    // pause the mic while TTS plays so the assistant does not hear itself
    playerRef.current = createTtsPlayer((playing) => {
      sttSessionRef.current?.setMuted(playing)
      convSessionRef.current?.setMuted(playing)
      voiceSessionRef.current?.setMuted(playing)
    })
  }

  const appendTo = (panel, item) =>
    setPanels((prev) => ({ ...prev, [panel]: [...prev[panel], item] }))

  // ---------------------------------------------------------- server events
  const handleEvent = useCallback((ev) => {
    switch (ev.type) {
      case '__reset__':
        setPanels(EMPTY_PANELS)
        break
      case 'system':
        appendTo(ev.panel === 'qa' ? 'qa' : 'transcripts', { label: 'System', text: ev.text })
        break
      case 'transcript':
        appendTo('transcripts', { label: ev.label, text: ev.text })
        break
      case 'qa':
        appendTo('qa', { q: ev.question, a: ev.answer })
        break
      case 'video':
        appendTo('video', { label: ev.head, text: ev.text })
        break
      case 'summary':
        appendTo('summary', { label: ev.head, text: ev.text })
        break
      case 'voice_transcript':
        setQuestion(ev.text)
        break
      case 'voice_done':
        voiceDoneResolver.current?.(ev.text)
        voiceDoneResolver.current = null
        break
      case 'speak':
        playerRef.current?.enqueue(ev.url)
        break
      case 'settings':
        setSettings(ev.settings)
        break
      default:
        break
    }
  }, [])

  useEvents(handleEvent)

  useEffect(() => {
    api.get('/api/state').then(setSettings).catch(() => {})
  }, [])

  // ---------------------------------------------------------- audio helpers
  const audioError = (err) => {
    const hint = audioSupported()
      ? ''
      : ' Microphone access needs a secure context — open the app via' +
        ' http://localhost (e.g. ssh -L 8800:localhost:8800 user@server).'
    appendTo('transcripts', { label: 'System', text: `Audio error: ${err.message || err}.${hint}` })
  }

  const checkAudio = () => {
    if (audioSupported()) return true
    audioError(new Error('getUserMedia unavailable'))
    return false
  }

  // ---------------------------------------------------------- actions
  const updateSettings = (patch) => {
    setSettings((prev) => ({ ...prev, ...patch })) // optimistic; server echoes back
    api.post('/api/settings', patch)
  }

  const toggleStt = () => {
    if (sttSessionRef.current) {
      sttSessionRef.current.stop()
      sttSessionRef.current = null
      if (speakerSessionRef.current) {
        speakerSessionRef.current.stop()
        speakerSessionRef.current = null
        setSpeakerSharing(false)
      }
    } else if (settings.stt_running) {
      appendTo('transcripts', { label: 'System', text: 'STT is already running in another window.' })
    } else if (checkAudio()) {
      sttSessionRef.current = openAudioSession({
        target: 'transcript', channel: 1, capture: startMicCapture, onError: audioError,
      })
    }
  }

  const toggleSpeakerShare = () => {
    if (speakerSessionRef.current) {
      speakerSessionRef.current.stop()
      speakerSessionRef.current = null
      setSpeakerSharing(false)
    } else if (checkAudio()) {
      speakerSessionRef.current = openAudioSession({
        target: 'transcript',
        channel: 2,
        capture: startSystemCapture,
        onError: (err) => {
          audioError(err)
          speakerSessionRef.current = null
          setSpeakerSharing(false)
        },
      })
      setSpeakerSharing(true)
    }
  }

  const toggleConversation = () => {
    if (convSessionRef.current) {
      convSessionRef.current.stop()
      convSessionRef.current = null
      playerRef.current?.stopAll()
    } else if (settings.conversation_running) {
      appendTo('qa', { label: 'System', text: 'Conversation is already running in another window.' })
    } else if (checkAudio()) {
      convSessionRef.current = openAudioSession({
        target: 'conversation', capture: startMicCapture, onError: audioError,
      })
    }
  }

  const submitQuestion = async (text) => {
    const q = (text ?? question).trim()
    if (!q) {
      appendTo('qa', { label: 'System', text: 'Please enter a question.' })
      return
    }
    setBusy(true)
    try {
      const res = await api.post('/api/question', { question: q })
      if (!res.error) setQuestion('')
    } finally {
      setBusy(false)
    }
  }

  const voiceStart = () => {
    if (voiceSessionRef.current || !checkAudio()) return
    setVoiceActive(true)
    voiceSessionRef.current = openAudioSession({
      target: 'question', capture: startMicCapture, onError: audioError,
    })
  }

  const voiceStop = async () => {
    const session = voiceSessionRef.current
    if (!session) return
    voiceSessionRef.current = null
    setVoiceActive(false)
    // closing the stream makes the server flush the recognizer and emit voice_done
    const transcript = await new Promise((resolve) => {
      voiceDoneResolver.current = resolve
      setTimeout(() => resolve(null), 8000)
      session.stop()
    })
    const text = (transcript || '').trim()
    if (text) {
      setQuestion(text)
      submitQuestion(text) // mirror PyQt: releasing the button submits
    }
  }

  const handleUpload = async (file, mode) => {
    // mode 'process': File Management button; mode 'parse': Video Parser button
    const res = await api.uploadFile(file)
    if (!res.ok) return
    if (mode === 'parse' || res.needs_roi) {
      setRoiModal({ filename: res.filename, mode })
    }
  }

  const confirmRoi = async (roi) => {
    const { filename, mode } = roiModal
    setRoiModal(null)
    const endpoint = mode === 'parse' ? '/api/video/parse' : '/api/video/process'
    await api.post(endpoint, { filename, roi })
  }

  const videoSummary = () => api.post('/api/video/summary')

  const exitApp = async () => {
    if (!window.confirm('Exit will clear ./source and ./docs and stop the server. Continue?')) return
    await api.post('/api/exit')
  }

  // ---------------------------------------------------------- keyboard shortcuts
  const actions = useRef({})
  actions.current = {
    toggleStt, toggleConversation, updateSettings, settings,
    videoSummary, exitApp, voiceStart, voiceStop,
  }

  useEffect(() => {
    const isTyping = (e) =>
      e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA'

    const onKeyDown = (e) => {
      const a = actions.current
      if (e.key === 'Enter' && !isTyping(e)) {
        // hold Enter = push-to-talk voice input (same as the PyQt event filter)
        if (!e.repeat) {
          e.preventDefault()
          a.voiceStart()
        }
        return
      }
      if (isTyping(e)) return
      switch (e.code) {
        case 'Space': e.preventDefault(); a.toggleStt(); break
        case 'KeyC': a.toggleConversation(); break
        case 'KeyR': a.updateSettings({ read_aloud: !a.settings.read_aloud }); break
        case 'KeyA': a.updateSettings({ correct_content: !a.settings.correct_content }); break
        case 'KeyD': a.updateSettings({ use_retrieval: !a.settings.use_retrieval }); break
        case 'KeyT': a.updateSettings({ response_type: 'detail mode' }); break
        case 'KeyY': a.updateSettings({ response_type: 'concise mode' }); break
        case 'KeyS': a.videoSummary(); break
        case 'KeyE': a.exitApp(); break
        default: break
      }
    }

    const onKeyUp = (e) => {
      if (e.key === 'Enter' && !isTyping(e) && !e.repeat) {
        actions.current.voiceStop()
      }
    }

    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
    }
  }, [])

  // ---------------------------------------------------------- layout
  return (
    <div className="app">
      <header className="app-header">
        <div className="brand-mark"><SparklesIcon size={20} /></div>
        <div>
          <div className="brand-title">Multi-RAG</div>
          <div className="brand-sub">Multimodal retrieval-augmented assistant</div>
        </div>
        <div className="header-hints">
          <span><kbd>Space</kbd> STT</span>
          <span><kbd>C</kbd> Conversation</span>
          <span>Hold <kbd>Enter</kbd> to talk</span>
        </div>
      </header>

      <ControlBar
        settings={settings}
        speakerSharing={speakerSharing}
        onToggleStt={toggleStt}
        onToggleSpeakerShare={toggleSpeakerShare}
        onToggleConversation={toggleConversation}
        onToggleSetting={(key) => updateSettings({ [key]: !settings[key] })}
        onUpload={(file) => handleUpload(file, 'process')}
        onVideoParse={(file) => handleUpload(file, 'parse')}
        onSummary={videoSummary}
        onExit={exitApp}
      />

      <main className="panels">
        <MessagePanel title="Transcripts" items={panels.transcripts} />
        <MessagePanel title="Q&A" items={panels.qa} />
        <div className="panel-stack">
          <MessagePanel title="Video Summary" items={panels.summary} />
          <MessagePanel title="Video Transcripts / Frames" items={panels.video} />
        </div>
      </main>

      <QuestionBar
        settings={settings}
        question={question}
        busy={busy}
        voiceActive={voiceActive}
        onQuestionChange={setQuestion}
        onSubmit={() => submitQuestion()}
        onResponseType={(value) => updateSettings({ response_type: value })}
        onToggleRetrieval={() => updateSettings({ use_retrieval: !settings.use_retrieval })}
        onVoiceStart={voiceStart}
        onVoiceStop={voiceStop}
      />

      {roiModal && (
        <RoiModal
          filename={roiModal.filename}
          onConfirm={confirmRoi}
          onCancel={() => setRoiModal(null)}
        />
      )}
    </div>
  )
}
