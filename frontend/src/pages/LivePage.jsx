/**
 * src/pages/LivePage.jsx
 * ───────────────────────
 * Live webcam detection page.
 * Same logic as original — uses api.js (Bearer token) instead of X-API-Key.
 */

import { useEffect, useRef, useState } from 'react'
import { Camera, CameraOff, CircleDot, HelpCircle, User, UserRoundSearch } from 'lucide-react'
import { predictImage, predictBase64 } from '../api'
import FaceCard     from '../components/FaceCard'
import SkeletonCard from '../components/SkeletonCard'
import Legend       from '../components/Legend'

const emotionColors = {
  Angry: '#FF4C4C', Fear: '#8E44AD', Happy: '#FFD93D',
  Neutral: '#BDC3C7', Sad: '#3498DB', Surprise: '#FF9F43',
}

function getEmotionColor(emotion) {
  if (!emotion) return '#BDC3C7'
  const match = Object.keys(emotionColors).find(
    (e) => e.toLowerCase() === emotion.toLowerCase()
  )
  return match ? emotionColors[match] : '#BDC3C7'
}

export default function LivePage() {
  const videoRef = useRef(null)
  const [streaming, setStreaming]             = useState(false)
  const [permissionError, setPermissionError] = useState('')
  const [detectedFaces, setDetectedFaces]     = useState([])

  const [snapshotImage, setSnapshotImage]     = useState(null)
  const [isSimulating, setIsSimulating]       = useState(false)
  const [simulatedProgress, setSimulatedProgress] = useState(0)
  const [statusText, setStatusText]           = useState('')
  const [snapshotResult, setSnapshotResult]   = useState(null)
  const [isSnapshotMode, setIsSnapshotMode]   = useState(false)
  const [snapshotDims, setSnapshotDims]       = useState({ w: 1, h: 1 })
  const [hoveredFaceId, setHoveredFaceId]     = useState(null)

  const startCamera = async () => {
    setPermissionError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      if (videoRef.current) videoRef.current.srcObject = stream
      setStreaming(true)
    } catch {
      setPermissionError('Camera permission denied or unavailable.')
    }
  }

  const stopCamera = () => {
    videoRef.current?.srcObject?.getTracks().forEach((t) => t.stop())
    if (videoRef.current) videoRef.current.srcObject = null
    setStreaming(false)
    setDetectedFaces([])
  }

  const takeSnapshot = async () => {
    if (!videoRef.current) return
    const canvas = document.createElement('canvas')
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0)

    stopCamera()
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9)
    setSnapshotImage(dataUrl)
    setSnapshotDims({ w: canvas.width, h: canvas.height })
    setIsSnapshotMode(true)
    setIsSimulating(true)
    setSimulatedProgress(0)
    setStatusText('Capturing frame...')
    setSnapshotResult(null)
    setPermissionError('')

    canvas.toBlob(async (blob) => {
      if (!blob) return
      try {
        const file = new File([blob], 'snapshot.jpg', { type: 'image/jpeg' })
        const fetchPromise = predictImage(file)

        let cur = 0
        const interval = setInterval(() => {
          cur += 2
          if (cur >= 90) { clearInterval(interval) }
          else {
            setSimulatedProgress(cur)
            if (cur < 30) setStatusText('Scanning faces...')
            else if (cur < 60) setStatusText('Analyzing emotions...')
            else setStatusText('Matching identities...')
          }
        }, 50)

        const data = await fetchPromise
        clearInterval(interval)
        setSimulatedProgress(100)
        setStatusText('Finalizing results...')
        setTimeout(() => { setSnapshotResult(data); setIsSimulating(false) }, 500)
      } catch (err) {
        setPermissionError(err.message || 'Snapshot processing failed.')
        setIsSimulating(false)
      }
    }, 'image/jpeg', 0.9)
  }

  const resumeLive = () => {
    setSnapshotImage(null)
    setIsSnapshotMode(false)
    setSnapshotResult(null)
    setIsSimulating(false)
    startCamera()
  }

  useEffect(() => () => stopCamera(), [])

  useEffect(() => {
    if (!streaming) return
    let isActive = true
    let timeoutId = null
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')

    const processFrame = async () => {
      if (!isActive) return
      if (!videoRef.current || videoRef.current.videoWidth === 0) {
        timeoutId = setTimeout(processFrame, 500); return
      }
      canvas.width = videoRef.current.videoWidth
      canvas.height = videoRef.current.videoHeight
      ctx.drawImage(videoRef.current, 0, 0)
      const base64Image = canvas.toDataURL('image/jpeg', 0.8)
      const image_b64 = base64Image.replace(/^data:image\/[a-z]+;base64,/, '')

      try {
        const data = await predictBase64(image_b64, 'live.jpg')
        if (isActive) {
          setDetectedFaces(data.results.map((face) => ({
            id: face.face_idx, name: face.name, emotion: face.emotion, face_image: face.face_image,
            x: (face.x / canvas.width) * 100, y: (face.y / canvas.height) * 100,
            w: (face.w / canvas.width) * 100, h: (face.h / canvas.height) * 100,
          })))
        }
      } catch { /* ignore live errors */ }
      if (isActive) timeoutId = setTimeout(processFrame, 200)
    }

    processFrame()
    return () => { isActive = false; if (timeoutId) clearTimeout(timeoutId) }
  }, [streaming])

  const renderFaceBoxes = (faces, dims, progress, isSimMode) =>
    faces.map((face, idx) => {
      const isUnknown = !face.name || face.name.toLowerCase() === 'unknown'
      const color = getEmotionColor(face.emotion)
      const effectiveW = dims ? Math.min(dims.w, 640) : null
      const scale = dims ? effectiveW / dims.w : 1
      const effectiveH = dims ? dims.h * scale : null
      const left  = dims ? `${(face.x / effectiveW) * 100}%` : `${face.x}%`
      const top   = dims ? `${(face.y / effectiveH) * 100}%` : `${face.y}%`
      const width = dims ? `${(face.w / effectiveW) * 100}%` : `${face.w}%`
      const height = dims ? `${(face.h / effectiveH) * 100}%` : `${face.h}%`
      const analyzing = isSimMode && progress < 70
      const hovered = hoveredFaceId === (face.face_idx ?? face.id)

      return (
        <div
          key={idx}
          className={`face-box scale-in ${hovered ? 'highlighted' : ''}`}
          onMouseEnter={() => setHoveredFaceId(face.face_idx ?? face.id)}
          onMouseLeave={() => setHoveredFaceId(null)}
          style={{
            left, top, width, height,
            borderColor: analyzing ? '#64748b' : (hovered ? '#fff' : color),
            borderStyle: isUnknown && !analyzing ? 'dashed' : 'solid',
            boxShadow: analyzing ? 'none' : (hovered ? `0 0 25px 2px ${color}` : `0 0 10px ${color}`),
            zIndex: hovered ? 100 : 1
          }}
        >
          <span style={{
            backgroundColor: isUnknown && !analyzing ? '#000' : 'rgba(15,23,42,0.85)',
            color: analyzing ? '#e2e8f0' : color,
            border: `1px solid ${analyzing ? '#64748b' : color}`
          }}>
            {analyzing ? <>Analyzing...</> :
              isUnknown ? <><HelpCircle size={12} className="inline-icon" /> UNKNOWN - {face.emotion}</> :
              <><User size={12} className="inline-icon" /> {face.name} - {face.emotion}</>}
          </span>
        </div>
      )
    })

  return (
    <section className="fade-in live-section">
      <div className="page-heading">
        <h2>Real-Time Detection</h2>
        <p>Live webcam stream with dynamic face boxes and emotion labels.</p>
      </div>

      <div className="live-layout">
        <article className="glass-card camera-card">
          <div className="camera-frame">
            {isSnapshotMode && snapshotImage ? (
              <div className="preview-image-container" style={{ width: '100%', height: '100%', margin: 0 }}>
                <img
                  src={snapshotImage} alt="Snapshot"
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  className={`preview-image ${isSimulating && simulatedProgress < 90 ? 'processing' : ''}`}
                />
                {isSimulating && simulatedProgress < 100 && <div className="scanning-beam"></div>}
                {snapshotResult && simulatedProgress >= 40 &&
                  renderFaceBoxes(snapshotResult.results, snapshotDims, simulatedProgress, isSimulating)}
              </div>
            ) : (
              <>
                <video ref={videoRef} autoPlay muted playsInline />
                {!streaming && (
                  <div className="camera-placeholder">
                    <UserRoundSearch size={48} />
                    <p>Camera feed will appear here</p>
                  </div>
                )}
                {renderFaceBoxes(detectedFaces, null, 100, false)}
              </>
            )}
          </div>

          <div className="controls">
            {isSnapshotMode ? (
              <button className="secondary-btn" onClick={resumeLive}><Camera size={16} /> Retake</button>
            ) : !streaming ? (
              <button className="primary-btn" onClick={startCamera}>
                <CircleDot size={16} color="#ef4444" /> Start Camera
              </button>
            ) : (
              <button className="secondary-btn" onClick={stopCamera}><CameraOff size={16} /> Stop Camera</button>
            )}
            <button
              className="primary-btn"
              style={{ marginLeft: 'auto' }}
              disabled={!streaming || isSnapshotMode}
              onClick={takeSnapshot}
            >
              <Camera size={16} /> {isSimulating ? 'Processing...' : 'Snapshot'}
            </button>
          </div>
          {permissionError && <p className="error">{permissionError}</p>}
        </article>

        <article className="glass-card stats-card">
          {isSnapshotMode ? (
            <>
              <h3>Snapshot Results</h3>
              {isSimulating && (
                <div className="processing-status-container">
                  <div className="status-text">{statusText}</div>
                  <div className="progress-bar-bg"><div className="progress-bar-fill" style={{ width: `${simulatedProgress}%` }}></div></div>
                </div>
              )}
              {isSimulating && simulatedProgress >= 30 && !snapshotResult && (
                <div className="results-list"><SkeletonCard /><SkeletonCard /></div>
              )}
              {!isSimulating && snapshotResult && snapshotResult.results.length === 0 && (
                <p className="warning">No face detected in the snapshot.</p>
              )}
              {!isSimulating && snapshotResult && snapshotResult.results.length > 0 && (
                <div className="results-list">
                  <div className="badge-row fade-in">
                    <span className="badge">Faces: {snapshotResult.n_faces}</span>
                    <span className="badge">Identified: {snapshotResult.n_identified}</span>
                    <span className="badge">Unknown: {snapshotResult.n_faces - snapshotResult.n_identified}</span>
                  </div>
                  {(() => {
                    const known   = snapshotResult.results.filter(p => p.name !== 'UNKNOWN' && p.name.toLowerCase() !== 'unknown')
                    const unknown = snapshotResult.results.filter(p => p.name === 'UNKNOWN' || p.name.toLowerCase() === 'unknown')
                    return (
                      <>
                        {known.length > 0 && (
                          <div className="result-section known-section fade-in">
                            <h4 className="section-title">Known Individuals</h4>
                            <div className="section-divider known-divider"></div>
                            {known.map((face, idx) => <FaceCard key={`snap-known-${face.face_idx}`} face={face} delay={idx * 0.1} onHover={setHoveredFaceId} isHighlighted={hoveredFaceId === face.face_idx} />)}
                          </div>
                        )}
                        {unknown.length > 0 && (
                          <div className="result-section unknown-section fade-in" style={{ marginTop: known.length > 0 ? '1.5rem' : '0' }}>
                            <h4 className="section-title">Unknown Individuals</h4>
                            <div className="section-divider unknown-divider"></div>
                            {unknown.map((face, idx) => <FaceCard key={`snap-unknown-${face.face_idx}`} face={face} delay={idx * 0.1} onHover={setHoveredFaceId} isHighlighted={hoveredFaceId === face.face_idx} />)}
                          </div>
                        )}
                      </>
                    )
                  })()}
                </div>
              )}
            </>
          ) : (
            <>
              <h3>Live Analytics</h3>
              <p className="stat">Faces Detected: <strong>{detectedFaces.length}</strong></p>
              <div className="live-feed-list">
                {detectedFaces.length === 0 && <p className="muted">No active faces detected yet.</p>}
                {detectedFaces.map((face, idx) => (
                  <FaceCard key={`feed-${face.id}`} face={face} delay={idx * 0.1} onHover={setHoveredFaceId} isHighlighted={hoveredFaceId === face.id} />
                ))}
              </div>
            </>
          )}
        </article>
      </div>
      <Legend />
    </section>
  )
}
