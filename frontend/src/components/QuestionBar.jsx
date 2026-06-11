import { MicIcon, SendIcon } from './icons.jsx'

export default function QuestionBar({
  settings, question, busy, voiceActive,
  onQuestionChange, onSubmit, onResponseType, onToggleRetrieval,
  onVoiceStart, onVoiceStop,
}) {
  return (
    <div className="question-bar">
      <div className="options-row">
        <span className="options-label">Response Type</span>
        <div className="segmented" role="radiogroup" aria-label="Response type">
          <button
            type="button"
            title="Y"
            className={settings.response_type === 'concise mode' ? 'seg-btn active' : 'seg-btn'}
            onClick={() => onResponseType('concise mode')}
          >
            Concise
          </button>
          <button
            type="button"
            title="T"
            className={settings.response_type === 'detail mode' ? 'seg-btn active' : 'seg-btn'}
            onClick={() => onResponseType('detail mode')}
          >
            Detail
          </button>
        </div>

        <label className="switch" title="D">
          <input
            type="checkbox"
            checked={settings.use_retrieval}
            onChange={onToggleRetrieval}
          />
          <span className="switch-track" />
          Vector Retrieval (Adaptive)
        </label>
      </div>

      <form
        className="input-row"
        onSubmit={(e) => {
          e.preventDefault()
          onSubmit()
        }}
      >
        <input
          className="question-input"
          type="text"
          value={question}
          placeholder="Type a question, or hold Enter / Voice Input to speak…"
          onChange={(e) => onQuestionChange(e.target.value)}
        />
        <button
          type="button"
          className={voiceActive ? 'btn voice active' : 'btn voice'}
          onMouseDown={onVoiceStart}
          onMouseUp={onVoiceStop}
          onMouseLeave={() => voiceActive && onVoiceStop()}
          onTouchStart={(e) => { e.preventDefault(); onVoiceStart() }}
          onTouchEnd={(e) => { e.preventDefault(); onVoiceStop() }}
        >
          {voiceActive ? <span className="rec-dot" /> : <MicIcon />}
          {voiceActive ? 'Listening…' : 'Voice Input'}
        </button>
        <button type="submit" className="btn primary" disabled={busy}>
          <SendIcon />
          {busy ? 'Thinking…' : 'Submit'}
        </button>
      </form>
    </div>
  )
}
