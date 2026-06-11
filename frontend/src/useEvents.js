import { useEffect, useRef } from 'react'

/**
 * WebSocket connection to the backend event bus with auto-reconnect.
 * On every (re)connect a synthetic {type:'__reset__'} event is delivered
 * first so the app can clear its panels before the server replays history.
 */
export function useEvents(onEvent) {
  const cbRef = useRef(onEvent)
  cbRef.current = onEvent

  useEffect(() => {
    let ws = null
    let closed = false
    let timer = null

    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/ws`)
      ws.onopen = () => cbRef.current({ type: '__reset__' })
      ws.onmessage = (e) => {
        try {
          cbRef.current(JSON.parse(e.data))
        } catch {
          /* ignore malformed frames */
        }
      }
      ws.onclose = () => {
        if (!closed) timer = setTimeout(connect, 1500)
      }
    }

    connect()
    return () => {
      closed = true
      clearTimeout(timer)
      if (ws) ws.close()
    }
  }, [])
}
