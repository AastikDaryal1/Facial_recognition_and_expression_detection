/**
 * src/pages/LivePage.jsx
 * ───────────────────────
 * Live webcam detection page.
 * Extracted from old App.jsx — same UI, JWT auth replacing API key.
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import { ScanFace, HelpCircle, User } from 'lucide-react'
import { predictBase64 } from '../api'

const emotionColors = {
  Angry: '#FF4C4C', Fear: '#8E44AD', Happy: '#FFD93D',
  Neutral: '#BDC3C7', Sad: '#3498DB', Surprise: '#FF9F43',
}

function getEmotionColor(emotion) {
  if (!emotion) return '#BDC3C7'
  const match = Object.keys(emotionColors).find(e => e.toLowerCase() === emotion.toLowerCase())
  return match ? emotionColors[match] : '#BDC3C7'
}

function Legend() {
  return (
    <div className="dual-legend glass-card fade-in-up">
      <div className="legend-section">
        <h4 className="legend-title">Emotion (Color)</h4>
        <div className="legend-items">
          {Object.entries(emotionColors).map(([emotion, color]) => (
            <div key={emotion} className="legend-item">
              <span className="legend-color" style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}` }}></span>
              <span className="legend-text">{emotion}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="legend-divider"></div>
      <div className="legend-section">
        <h4 className="legend-title">Identity (Style)</h4>
        <div className="legend-items">
          <div className="legend-item style-known">
            <span className="legend-style-box solid"></span>
            <span className="legend-text"><User size={14} className="inline-icon" /> Known</span>
          </div>
          <div className="legend-item style-unknown">
            <span className="legend-style-box dashed"></span>
            <span className="legend-text"><HelpCircle size={14} className="inline-icon" /> Unknown</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function LivePage() {
  const videoRef        = useRef(null)
  const canvasRef       = useRef(null)
  const intervalRef     = useRef(null)
  const [active,        setActive]        = useState(false)
  const [faces,         setFaces]         = useState([])
  const [error,         setError]         = useState('')
  const [processing,    setProcessing]    = useState(false)
  const [frameCount,    setFrameCount]    = useState(0)

  const stopCamera = useCallback(() => {
    clearInterval(intervalRef.current)
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop())
      videoRef.current.srcObject = null
    }
    setActive(false)
    setFaces([])
    setProcessing(false)
  }, [])

  useEffect(() => () => stopCamera(), [stopCamera])

  const startCamera = async () => {
    setError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      videoRef.current.srcObject = stream
      await videoRef.current.play()
      setActive(true)

      intervalRef.current = setInterval(async () => {
        if (!videoRef.current || !canvasRef.current) return
        const canvas = canvasRef.current
        canvas.width  = videoRef.current.videoWidth
        canvas.height = videoRef.current.videoHeight
        canvas.getContext('2d').drawImage(videoRef.current, 0, 0)
        const b64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1]

        setProcessing(true)
        try {
          const data = await predictBase64(b64, 'live.jpg')
          setFaces(data.results || [])
          setFrameCount(c => c + 1)
        } catch {
          // Silently skip failed frames
        }
        setProcessing(false)
      }, 1500)

    } catch {
      setError('Could not access camera. Please check permissions.')
    }
  }

  const takeSnapshot = () => {
    if (!canvasRef.current) return
    const link = document.createElement('a')
    link.download = `snapshot_${Date.now()}.jpg`
    link.href = canvasRef.current.toDataURL('image/jpeg')
    link.click()
  }

  return (
    <section className="fade-in">
      <div className="page-heading">
        <h2>Real-Time Detection</h2>
        <p>Live webcam stream with dynamic face boxes and emotion labels.</p>
      </div>

      <div className="live-layout">
        <div className="live-feed-panel glass-card">
          <div className="video-container">
            <video
              ref       = {videoRef}
              autoPlay
              playsInline
              muted
              className = "live-video"
              style     = {{ display: active ? 'block' : 'none' }}
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />

            {!active && (
              <div className="video-placeholder">
                <ScanFace size={48} color="#334155" />
                <p>Camera feed will appear here</p>
              </div>
            )}

            {/* Face overlays */}
            {active && faces.map((face, idx) => {
              const isUnknown = !face.name || face.name.toLowerCase() === 'unknown'
              const color     = getEmotionColor(face.emotion)
              return (
                <div
                  key       = {idx}
                  className = "face-box"
                  style={{
                    left        : `${(face.x / (videoRef.current?.videoWidth || 640)) * 100}%`,
                    top         : `${(face.y / (videoRef.current?.videoHeight || 480)) * 100}%`,
                    width       : `${(face.w / (videoRef.current?.videoWidth || 640)) * 100}%`,
                    height      : `${(face.h / (videoRef.current?.videoHeight || 480)) * 100}%`,
                    borderColor : color,
                    borderStyle : isUnknown ? 'dashed' : 'solid',
                    boxShadow   : `0 0 10px ${color}`,
                  }}
                >
                  <span style={{ backgroundColor: 'rgba(15,23,42,0.85)', color, border: `1px solid ${color}` }}>
                    {isUnknown
                      ? <><HelpCircle size={12} className="inline-icon" /> UNKNOWN - {face.emotion}</>
                      : <><User size={12} className="inline-icon" /> {face.name} - {face.emotion}</>
                    }
                  </span>
                </div>
              )
            })}
          </div>

          <div className="live-controls">
            {!active
              ? <button className="primary-btn" onClick={startCamera}>▶ Start Camera</button>
              : <button className="secondary-btn" onClick={stopCamera}>⏹ Stop Camera</button>
            }
            {active && (
              <button className="secondary-btn" onClick={takeSnapshot}>
                📸 Snapshot
              </button>
            )}
          </div>
        </div>

        {/* Live analytics panel */}
        <div className="live-analytics glass-card">
          <h3>Live Analytics</h3>
          <p className="muted" style={{ fontSize: '0.85rem' }}>
            Faces Detected: <strong style={{ color: '#f1f5f9' }}>{faces.length}</strong>
          </p>
          {processing && <p style={{ color: '#6366f1', fontSize: '0.8rem' }}>Processing frame...</p>}
          {active && frameCount > 0 && (
            <p style={{ color: '#64748b', fontSize: '0.8rem' }}>Frames processed: {frameCount}</p>
          )}
          {faces.length === 0
            ? <p className="muted" style={{ fontSize: '0.85rem' }}>No active faces detected yet.</p>
            : faces.map((face, idx) => {
                const isUnknown = !face.name || face.name.toLowerCase() === 'unknown'
                const color     = getEmotionColor(face.emotion)
                return (
                  <div key={idx} className="glass-card" style={{
                    padding : '0.75rem', marginTop: '0.75rem',
                    borderLeft: `3px solid ${color}`,
                  }}>
                    <p style={{ margin: 0, fontWeight: 600, color: isUnknown ? '#94a3b8' : '#f1f5f9', fontSize: '0.9rem' }}>
                      {isUnknown ? 'Unknown Person' : face.name}
                    </p>
                    <p style={{ margin: '0.25rem 0 0', color, fontSize: '0.8rem' }}>
                      {face.emotion} ({Math.round((face.emotion_conf || 0) * 100)}%)
                    </p>
                  </div>
                )
              })
          }
          {error && <p className="error" style={{ marginTop: '1rem' }}>{error}</p>}
        </div>
      </div>

      <Legend />
    </section>
  )
}
