import { useRef, useState } from 'react'

/**
 * Video preview + ROI selection, replacing the Qt VideoPreviewDialog.
 * Drag a rectangle over the playing video; on release the rectangle is
 * scaled from display coordinates to original video pixels and confirmed,
 * matching the behavior of the PyQt overlay.
 */
export default function RoiModal({ filename, onConfirm, onCancel }) {
  const videoRef = useRef(null)
  const overlayRef = useRef(null)
  const startRef = useRef(null)
  const [rect, setRect] = useState(null)

  const pointInOverlay = (e) => {
    const bounds = overlayRef.current.getBoundingClientRect()
    return {
      x: Math.min(Math.max(e.clientX - bounds.left, 0), bounds.width),
      y: Math.min(Math.max(e.clientY - bounds.top, 0), bounds.height),
    }
  }

  const normalized = (a, b) => ({
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    w: Math.abs(a.x - b.x),
    h: Math.abs(a.y - b.y),
  })

  const onMouseDown = (e) => {
    if (e.button !== 0) return
    startRef.current = pointInOverlay(e)
    setRect({ ...startRef.current, w: 0, h: 0 })
  }

  const onMouseMove = (e) => {
    if (!startRef.current) return
    setRect(normalized(startRef.current, pointInOverlay(e)))
  }

  const onMouseUp = (e) => {
    if (!startRef.current) return
    const finalRect = normalized(startRef.current, pointInOverlay(e))
    startRef.current = null
    setRect(finalRect)

    if (finalRect.w < 4 || finalRect.h < 4) return // ignore accidental clicks

    const video = videoRef.current
    if (!video || !video.videoWidth) return
    // scale display coordinates back to original video pixels
    const scaleX = video.videoWidth / video.clientWidth
    const scaleY = video.videoHeight / video.clientHeight
    onConfirm([
      Math.round(finalRect.x * scaleX),
      Math.round(finalRect.y * scaleY),
      Math.round(finalRect.w * scaleX),
      Math.round(finalRect.h * scaleY),
    ])
  }

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <div className="modal-title">Video Preview and ROI Selection</div>
        <p className="modal-hint">
          Drag a rectangle on the video to choose the region of interest.
        </p>
        <div className="video-wrap">
          <video
            ref={videoRef}
            src={`/source/${encodeURIComponent(filename)}`}
            autoPlay
            muted
            loop
          />
          <div
            ref={overlayRef}
            className="roi-overlay"
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
            onMouseLeave={() => { startRef.current = null }}
          >
            {rect && (
              <div
                className="roi-rect"
                style={{ left: rect.x, top: rect.y, width: rect.w, height: rect.h }}
              />
            )}
          </div>
        </div>
        <div className="modal-actions">
          <button className="btn ghost" onClick={onCancel}>Cancel</button>
          <button className="btn primary" onClick={() => onConfirm(null)}>
            Use Full Frame
          </button>
        </div>
      </div>
    </div>
  )
}
