import { useEffect, useRef } from 'react'
import { ChatIcon } from './icons.jsx'

// conversation-mode questions arrive tagged with the Hume.ai emotion label,
// e.g. "How does this work?  [Emotion: Curiosity]" — render it as a chip
const EMOTION_RE = /\s*\[Emotion:\s*([^\]]+)\]\s*$/

function QuestionText({ text }) {
  const match = typeof text === 'string' ? text.match(EMOTION_RE) : null
  if (!match) return text
  return (
    <>
      {text.replace(EMOTION_RE, '')}
      <span className="emotion-chip">{match[1]}</span>
    </>
  )
}

export default function MessagePanel({ title, items }) {
  const boxRef = useRef(null)

  useEffect(() => {
    const box = boxRef.current
    if (box) box.scrollTop = box.scrollHeight
  }, [items])

  return (
    <section className="panel">
      <div className="panel-head">
        <span className="panel-title">{title}</span>
        {items.length > 0 && <span className="panel-count">{items.length}</span>}
      </div>
      <div className="panel-box" ref={boxRef}>
        {items.length === 0 && (
          <div className="panel-empty">
            <ChatIcon size={22} />
            Nothing here yet
          </div>
        )}
        {items.map((item, i) =>
          item.q !== undefined ? (
            <div className="qa-item" key={i}>
              <div className="bubble q"><QuestionText text={item.q} /></div>
              <div className="bubble a">{item.a}</div>
            </div>
          ) : (
            <p key={i} className={item.label === 'System' ? 'msg system' : 'msg'}>
              {item.label ? (
                <span className={item.label === 'System' ? 'tag system' : 'tag'}>
                  {item.label}
                </span>
              ) : null}
              {item.text}
            </p>
          ),
        )}
      </div>
    </section>
  )
}
