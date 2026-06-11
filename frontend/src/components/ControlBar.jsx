import { useRef } from 'react'
import {
  ChatIcon, FilmIcon, MicIcon, PowerIcon, SpeakerIcon, SummaryIcon, UploadIcon,
} from './icons.jsx'

export default function ControlBar({
  settings, speakerSharing, onToggleStt, onToggleSpeakerShare,
  onToggleConversation, onToggleSetting,
  onUpload, onVideoParse, onSummary, onExit,
}) {
  const uploadInput = useRef(null)
  const videoInput = useRef(null)

  const pickFile = (ref, handler) => (e) => {
    const file = e.target.files?.[0]
    if (file) handler(file)
    e.target.value = '' // allow re-selecting the same file
  }

  return (
    <div className="control-bar">
      <button
        className={settings.stt_running ? 'btn active' : 'btn'}
        onClick={onToggleStt}
        title="Space"
      >
        {settings.stt_running ? <span className="rec-dot" /> : <MicIcon />}
        {settings.stt_running ? 'Stop STT' : 'Start STT'}
      </button>

      {settings.stt_running && (
        <button
          className={speakerSharing ? 'btn active' : 'btn'}
          onClick={onToggleSpeakerShare}
          title="Share a tab/screen with audio; it is transcribed as the Speaker channel"
        >
          {speakerSharing ? <span className="rec-dot" /> : <SpeakerIcon />}
          {speakerSharing ? 'Stop Speaker Audio' : 'Share Speaker Audio'}
        </button>
      )}

      <button
        className={settings.conversation_running ? 'btn active' : 'btn'}
        onClick={onToggleConversation}
        title="C"
      >
        {settings.conversation_running ? <span className="rec-dot" /> : <ChatIcon />}
        {settings.conversation_running ? 'Stop Conversation' : 'Talk with AI (Emotion Detection)'}
      </button>

      <span className="bar-divider" />

      <label className="switch" title="R">
        <input
          type="checkbox"
          checked={settings.read_aloud}
          onChange={() => onToggleSetting('read_aloud')}
        />
        <span className="switch-track" />
        Read Aloud
      </label>

      <label className="switch" title="A">
        <input
          type="checkbox"
          checked={settings.correct_content}
          onChange={() => onToggleSetting('correct_content')}
        />
        <span className="switch-track" />
        Correct Content
      </label>

      <span className="bar-divider" />

      <button className="btn" onClick={() => uploadInput.current.click()}>
        <UploadIcon />
        File Management
      </button>
      <input
        ref={uploadInput}
        type="file"
        hidden
        accept=".pdf,.docx,.xlsx,.pptx,.png,.jpg,.jpeg,.txt,.mp4"
        onChange={pickFile(uploadInput, onUpload)}
      />

      <button className="btn" onClick={() => videoInput.current.click()}>
        <FilmIcon />
        Video Parser
      </button>
      <input
        ref={videoInput}
        type="file"
        hidden
        accept=".mp4"
        onChange={pickFile(videoInput, onVideoParse)}
      />

      <button className="btn" onClick={onSummary} title="S">
        <SummaryIcon />
        Summary
      </button>

      <span className="bar-spacer" />

      <button className="btn exit" onClick={onExit} title="E">
        <PowerIcon />
        Exit
      </button>
    </div>
  )
}
