/**
 * src/pages/LivePage.jsx
 * ──────────────────────
 * Real-time face and emotion detection page.
 * Keeps Shubh's stable, high-fidelity webcam lifecycle,
 * requestAnimationFrame interpolation, smoothing logic,
 * and coordinate scaling systems completely intact.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react'
import {
  Camera,
  CameraOff,
  User,
  HelpCircle,
  CircleDot,
  Zap,
  Cpu,
  UserRoundSearch,
  ShieldCheck,
  RefreshCw,
} from 'lucide-react'
import { predictBase64 } from '../api'

// ── Helpers ──────────────────────────────────────────────────────────────────
function getEmotionColor(emotion) {
  const hex = {
    angry: '#ef4444',
    happy: '#22c55e',
    neutral: '#eab308',
    sad: '#3b82f6',
    surprise: '#a855f7',
    unknown: '#64748b',
  }
  return hex[emotion?.toLowerCase()] || hex.unknown
}

// ── Components ────────────────────────────────────────────────────────────────

function StatusCard({ icon: Icon, title, subtitle, variant = 'info' }) {
  const borderColors = {
    info: 'rgba(59, 130, 246, 0.2)',
    warning: 'rgba(234, 179, 8, 0.2)',
    danger: 'rgba(239, 68, 68, 0.2)',
  }
  const iconColors = {
    info: 'var(--accent-blue)',
    warning: 'var(--warning)',
    danger: 'var(--danger)',
  }
  return (
    <div
      className="glass-card status-card scale-in"
      style={{
        padding: '2rem 1.5rem',
        textAlign: 'center',
        background: 'rgba(15, 23, 42, 0.4)',
        border: `1px solid ${borderColors[variant] || borderColors.info}`,
        borderRadius: '1rem',
      }}
    >
      <div
        style={{
          background: 'rgba(15, 23, 42, 0.6)',
          width: '60px',
          height: '60px',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          margin: '0 auto 1.25rem auto',
          boxShadow: '0 8px 32px rgba(0,0,0,0.2)',
        }}
      >
        <Icon size={28} style={{ color: iconColors[variant] }} />
      </div>
      <h4 style={{ fontWeight: 600, fontSize: '1rem', marginBottom: '6px' }}>{title}</h4>
      <p className="muted" style={{ fontSize: '0.85rem', lineHeight: 1.4, margin: 0 }}>
        {subtitle}
      </p>
    </div>
  )
}

function SkeletonCard() {
  return (
    <div
      className="glass-card skeleton-card shimmer"
      style={{
        padding: '1rem',
        borderRadius: '0.75rem',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        marginBottom: '0.75rem',
        background: 'rgba(15, 23, 42, 0.2)',
      }}
    >
      <div style={{ width: '48px', height: '48px', borderRadius: '8px', background: '#1e293b' }}></div>
      <div style={{ flexGrow: 1 }}>
        <div style={{ height: '14px', width: '40%', background: '#1e293b', borderRadius: '4px', marginBottom: '8px' }}></div>
        <div style={{ height: '10px', width: '25%', background: '#1e293b', borderRadius: '4px' }}></div>
      </div>
    </div>
  )
}

function FaceCard({ face, delay = 0, onHover, isHighlighted }) {
  const color = getEmotionColor(face.emotion)
  const isUnknown = !face.name || face.name.toLowerCase() === 'unknown' || face.name.toLowerCase() === 'unknown subject'

  return (
    <div
      className={`glass-card face-card scale-in ${isHighlighted ? 'highlighted' : ''}`}
      onMouseEnter={() => onHover && onHover(face.id)}
      onMouseLeave={() => onHover && onHover(null)}
      style={{
        animationDelay: `${delay}s`,
        borderLeft: `4px solid ${color}`,
        display: 'flex',
        alignItems: 'center',
        gap: '1rem',
        padding: '1rem',
        borderRadius: '0.75rem',
        marginBottom: '0.75rem',
        transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
        background: isHighlighted ? 'rgba(30, 41, 59, 0.5)' : 'rgba(15, 23, 42, 0.2)',
        boxShadow: isHighlighted ? `0 0 20px rgba(15, 23, 42, 0.3)` : 'none',
      }}
    >
      <div style={{ position: 'relative' }}>
        {face.face_image ? (
          <img
            src={face.face_image}
            alt={face.name}
            style={{
              width: '48px',
              height: '48px',
              borderRadius: '8px',
              objectFit: 'cover',
              border: `2px solid ${color}`,
              boxShadow: `0 0 10px ${color}40`,
            }}
          />
        ) : (
          <div
            style={{
              width: '48px',
              height: '48px',
              borderRadius: '8px',
              background: 'rgba(30, 41, 59, 0.8)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: `2px dashed ${color}`,
            }}
          >
            {isUnknown ? <HelpCircle size={20} style={{ color }} /> : <User size={20} style={{ color }} />}
          </div>
        )}
      </div>

      <div style={{ flexGrow: 1 }}>
        <h4
          style={{
            fontSize: '0.95rem',
            fontWeight: 600,
            margin: 0,
            color: '#f1f5f9',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          {isUnknown ? 'Unknown Subject' : face.name}
        </h4>
        <p style={{ margin: '4px 0 0 0', fontSize: '0.8rem', color: color, fontWeight: 500 }}>
          {face.emotion?.toUpperCase()}
        </p>
      </div>
    </div>
  )
}

function Legend() {
  const emotions = ['angry', 'happy', 'neutral', 'sad', 'surprise']
  return (
    <div className="legend-container glass-card fade-in" style={{ marginTop: '2rem' }}>
      <div className="legend-section">
        <h4 className="legend-title">Emotion Gating</h4>
        <div className="legend-items">
          {emotions.map((emotion) => (
            <div className="legend-item" key={emotion}>
              <span className="legend-color-box" style={{ backgroundColor: getEmotionColor(emotion) }}></span>
              <span className="legend-text" style={{ textTransform: 'capitalize' }}>
                {emotion}
              </span>
            </div>
          ))}
        </div>
      </div>
      <div className="legend-divider"></div>
      <div className="legend-section">
        <h4 className="legend-title">Identity (Style)</h4>
        <div className="legend-items">
          <div className="legend-item">
            <span
              className="legend-color-box"
              style={{ border: '2px solid var(--accent-blue)', background: 'transparent' }}
            ></span>
            <span className="legend-text">Known Individual</span>
          </div>
          <div className="legend-item">
            <span
              className="legend-color-box"
              style={{ border: '2px dashed #ef4444', background: 'transparent' }}
            ></span>
            <span className="legend-text">Unknown Subject</span>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Main LivePage Component ──────────────────────────────────────────────────

export default function LivePage() {
  const videoRef = useRef(null)
  const [streaming, setStreaming] = useState(false)
  const [permissionError, setPermissionError] = useState('')
  const [detectedFaces, setDetectedFaces] = useState([])

  const [snapshotImage, setSnapshotImage] = useState(null)
  const [isSimulating, setIsSimulating] = useState(false)
  const [simulatedProgress, setSimulatedProgress] = useState(0)
  const [statusText, setStatusText] = useState('')
  const [snapshotResult, setSnapshotResult] = useState(null)
  const [isSnapshotMode, setIsSnapshotMode] = useState(false)
  const [snapshotDims, setSnapshotDims] = useState({ w: 1, h: 1 })
  const [hoveredFaceId, setHoveredFaceId] = useState(null)

  // PRODUCTION LIFECYCLE MANAGEMENT
  const activeSessionRef = useRef(0)
  const isProcessingRef = useRef(false)
  const abortControllerRef = useRef(null)
  const canvasRef = useRef(document.createElement('canvas'))
  const lifecycleRef = useRef('IDLE') // IDLE, STARTING, RUNNING, STOPPING

  const log = (msg, data = '') => {
    const timestamp = new Date().toISOString().split('T')[1].split('Z')[0]
    console.log(`[VisionX][${timestamp}] ${msg}`, data)
  }

  const startCamera = async () => {
    if (lifecycleRef.current === 'STARTING' || lifecycleRef.current === 'RUNNING') {
      log('startCamera ignored - already in progress or running')
      return
    }

    log('--- STARTING CAMERA ---')
    lifecycleRef.current = 'STARTING'
    setPermissionError('')

    try {
      // 1. Cleanup any ghost tracks first
      log('Step 1: Disposing old session...')
      stopCamera()

      // 2. Request fresh stream
      log('Step 2: Requesting getUserMedia...')
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 },
        },
      })

      if (!videoRef.current) throw new Error('Video element unmounted')

      // 3. Attach and Play
      log('Step 3: Attaching stream and playing...')
      videoRef.current.srcObject = stream

      // We wrap play() in a promise to ensure we wait for the hardware to wake up
      await new Promise((resolve, reject) => {
        if (!videoRef.current) return reject()
        videoRef.current.onplaying = () => {
          log('Event: video.onplaying fired')
          resolve()
        }
        videoRef.current.onerror = reject
        videoRef.current.play().catch(reject)
      })

      // 4. Finalize session
      activeSessionRef.current += 1
      lifecycleRef.current = 'RUNNING'
      setStreaming(true)
      log(`--- CAMERA READY (Session #${activeSessionRef.current}) ---`)
    } catch (err) {
      log('--- CAMERA FAILED ---', err)
      lifecycleRef.current = 'IDLE'
      setPermissionError('Camera error: ' + (err.message || 'Unknown failure'))
    }
  }

  const stopCamera = () => {
    log('--- STOPPING CAMERA ---')
    lifecycleRef.current = 'STOPPING'

    // 1. Kill Inference
    if (abortControllerRef.current) {
      log('Killing inference loop AbortController')
      abortControllerRef.current.abort()
    }
    activeSessionRef.current += 1 // Increment again to be sure

    // 2. Kill Hardware
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject
      stream.getTracks().forEach((track) => {
        log(`Stopping track: ${track.label}`)
        track.stop()
      })
      videoRef.current.srcObject = null
      if (videoRef.current.load) {
        videoRef.current.load() // Force reset video element
      }
    }

    setStreaming(false)
    setDetectedFaces([])
    setSmoothedFaces([])
    targetFacesRef.current = []
    lifecycleRef.current = 'IDLE'
    log('--- CAMERA STOPPED ---')
  }

  const takeSnapshot = async () => {
    if (!videoRef.current) return

    const canvas = document.createElement('canvas')
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)

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

    canvas.toBlob(
      async (blob) => {
        if (!blob) return
        const file = new File([blob], 'snapshot.jpg', { type: 'image/jpeg' })

        try {
          let currentProgress = 0
          const interval = setInterval(() => {
            currentProgress += 2
            if (currentProgress >= 90) {
              clearInterval(interval)
            } else {
              setSimulatedProgress(currentProgress)
              if (currentProgress < 30) setStatusText('Scanning faces...')
              else if (currentProgress < 60) setStatusText('Analyzing emotions...')
              else setStatusText('Matching identities...')
            }
          }, 50)

          // Use the modular predictBase64 helper from api.js (attaches JWT automatically)
          const base64Image = dataUrl.replace(/^data:image\/[a-z]+;base64,/, '')
          const data = await predictBase64(base64Image, 'snapshot.jpg')

          clearInterval(interval)
          setSimulatedProgress(100)
          setStatusText('Finalizing results...')

          setTimeout(() => {
            setSnapshotResult(data)
            setIsSimulating(false)
          }, 500)
        } catch (err) {
          setPermissionError(err.message || 'Snapshot processing failed.')
          setIsSimulating(false)
        }
      },
      'image/jpeg',
      0.9
    )
  }

  const resumeLive = () => {
    setSnapshotImage(null)
    setIsSnapshotMode(false)
    setSnapshotResult(null)
    setIsSimulating(false)
    startCamera()
  }

  // Refined Smoothing Logic for Ultra-Responsive UI
  const [smoothedFaces, setSmoothedFaces] = useState([])
  const targetFacesRef = useRef([])
  const lastUpdateRef = useRef(Date.now())

  useEffect(() => {
    let rafId
    const smoothingFactor = 0.15 // Lower = smoother/slower, Higher = snappier

    const updateSmoothedPositions = () => {
      const now = Date.now()
      const dt = now - lastUpdateRef.current
      lastUpdateRef.current = now

      setSmoothedFaces((prev) => {
        // Map current smoothed faces to target faces by ID
        const targets = targetFacesRef.current

        // If no targets, clear
        if (targets.length === 0) return []

        return targets.map((target) => {
          const existing = prev.find((p) => p.id === target.id)
          if (!existing) return { ...target } // New face, snap immediately

          // Interpolate coordinates
          return {
            ...target,
            x: existing.x + (target.x - existing.x) * smoothingFactor,
            y: existing.y + (target.y - existing.y) * smoothingFactor,
            w: existing.w + (target.w - existing.w) * smoothingFactor,
            h: existing.h + (target.h - existing.h) * smoothingFactor,
          }
        })
      })

      rafId = requestAnimationFrame(updateSmoothedPositions)
    }

    if (streaming) {
      rafId = requestAnimationFrame(updateSmoothedPositions)
    }

    return () => cancelAnimationFrame(rafId)
  }, [streaming])

  useEffect(() => () => stopCamera(), [])

  useEffect(() => {
    if (!streaming) return

    log(`[Inference] Initializing loop for Session #${activeSessionRef.current}`)
    const currentSession = activeSessionRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    const controller = new AbortController()
    abortControllerRef.current = controller

    const processFrame = async () => {
      // 1. SESSION GUARD
      if (activeSessionRef.current !== currentSession || controller.signal.aborted) {
        log(`[Inference] Terminating loop (Session mismatch or aborted)`)
        return
      }
      if (isProcessingRef.current) return

      const video = videoRef.current
      if (!video || video.paused || video.ended || video.readyState < 2) {
        // Spin until hardware is ready
        requestAnimationFrame(processFrame)
        return
      }

      isProcessingRef.current = true
      const frameStart = Date.now()

      try {
        const MAX_INF_WIDTH = 640
        const scale = Math.min(1, MAX_INF_WIDTH / video.videoWidth)
        canvas.width = video.videoWidth * scale
        canvas.height = video.videoHeight * scale
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

        const base64Image = canvas.toDataURL('image/jpeg', 0.6)
        const image_b64 = base64Image.replace(/^data:image\/[a-z]+;base64,/, '')

        log(`[Inference] API Request started (Session #${currentSession})`)

        // Use our central API caller (carries tokens and abort signals correctly)
        const data = await predictBase64(image_b64, 'live.jpg')

        // 2. POST-AWAIT SESSION GUARD
        if (activeSessionRef.current !== currentSession || controller.signal.aborted) {
          log(`[Inference] API returned but session expired. Discarding.`)
          return
        }

        log(`[Inference] API Success: ${data.results.length} faces detected in ${Date.now() - frameStart}ms`)

        const items = data.results.map((face) => {
          let localCrop = null
          try {
            const cropCanvas = document.createElement('canvas')
            const cropCtx = cropCanvas.getContext('2d')
            const cropSize = 120
            cropCanvas.width = cropSize
            cropCanvas.height = cropSize
            cropCtx.drawImage(canvas, face.x, face.y, face.w, face.h, 0, 0, cropSize, cropSize)
            localCrop = cropCanvas.toDataURL('image/jpeg', 0.85)
          } catch (e) {
            /* ignore crop error */
          }

          return {
            id: face.face_idx,
            name: face.name,
            emotion: face.emotion,
            face_image: localCrop,
            x: (face.x / canvas.width) * 100,
            y: (face.y / canvas.height) * 100,
            w: (face.w / canvas.width) * 100,
            h: (face.h / canvas.height) * 100,
          }
        })

        targetFacesRef.current = items
        setDetectedFaces(items)
      } catch (err) {
        if (err.name === 'AbortError') {
          log('[Inference] Fetch aborted intentionally')
        } else {
          log(`[Inference] Cycle Error: ${err.message}`)
        }
      } finally {
        isProcessingRef.current = false
        if (activeSessionRef.current === currentSession && !controller.signal.aborted) {
          // Stable 150ms throttle as calibrated in Shubh
          setTimeout(() => {
            if (activeSessionRef.current === currentSession && !controller.signal.aborted) {
              requestAnimationFrame(processFrame)
            }
          }, 150)
        }
      }
    }

    log(`[Inference] Scheduling loop initialization...`)
    const initTimeout = setTimeout(processFrame, 500)

    return () => {
      log(`[Inference] Cleaning up Session #${currentSession}`)
      controller.abort()
      clearTimeout(initTimeout)
      isProcessingRef.current = false
    }
  }, [streaming])

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
                  src={snapshotImage}
                  alt="Snapshot"
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  className={`preview-image ${isSimulating && simulatedProgress < 90 ? 'processing' : ''}`}
                />
                {isSimulating && simulatedProgress < 100 && <div className="scanning-beam"></div>}
                {snapshotResult &&
                  simulatedProgress >= 40 &&
                  snapshotResult.results.map((face, idx) => {
                    const isUnknown =
                      !face.name ||
                      face.name.toLowerCase() === 'unknown' ||
                      face.name.toLowerCase() === 'unknown subject'
                    const color = getEmotionColor(face.emotion)
                    const effectiveW = Math.min(snapshotDims.w, 640)
                    const scale = effectiveW / snapshotDims.w
                    const effectiveH = snapshotDims.h * scale

                    return (
                      <div
                        key={`snap-${idx}`}
                        className={`face-box scale-in ${hoveredFaceId === face.face_idx ? 'highlighted' : ''}`}
                        onMouseEnter={() => setHoveredFaceId(face.face_idx)}
                        onMouseLeave={() => setHoveredFaceId(null)}
                        style={{
                          left: `${(face.x / effectiveW) * 100}%`,
                          top: `${(face.y / effectiveH) * 100}%`,
                          width: `${(face.w / effectiveW) * 100}%`,
                          height: `${(face.h / effectiveH) * 100}%`,
                          borderColor:
                            isSimulating && simulatedProgress < 70
                              ? '#64748b'
                              : hoveredFaceId === face.face_idx
                              ? '#fff'
                              : color,
                          borderStyle: isUnknown && (!isSimulating || simulatedProgress >= 70) ? 'dashed' : 'solid',
                          boxShadow:
                            isSimulating && simulatedProgress < 70
                              ? 'none'
                              : hoveredFaceId === face.face_idx
                              ? `0 0 25px 2px ${color}`
                              : `0 0 10px ${color}`,
                          zIndex: hoveredFaceId === face.face_idx ? 100 : 1,
                        }}
                      >
                        <span
                          className={isSimulating && simulatedProgress < 70 ? 'label-analyzing' : ''}
                          style={{
                            backgroundColor:
                              isUnknown && (!isSimulating || simulatedProgress >= 70) ? '#000000' : 'rgba(15, 23, 42, 0.85)',
                            color: isSimulating && simulatedProgress < 70 ? '#e2e8f0' : color,
                            border: `1px solid ${isSimulating && simulatedProgress < 70 ? '#64748b' : color}`,
                          }}
                        >
                          {isSimulating && simulatedProgress < 70 ? (
                            <>Analyzing...</>
                          ) : isUnknown ? (
                            <>
                              <HelpCircle size={12} className="inline-icon" /> Unknown Subject - {face.emotion}
                            </>
                          ) : (
                            <>
                              <User size={12} className="inline-icon" /> {face.name} - {face.emotion}
                            </>
                          )}
                        </span>
                      </div>
                    )
                  })}
              </div>
            ) : (
              <div className="live-feed-container" style={{ width: '100%', height: '100%', position: 'relative' }}>
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  style={{
                    display: streaming ? 'block' : 'none',
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                  }}
                />
                {!streaming && (
                  <div className="camera-placeholder">
                    <UserRoundSearch size={48} />
                    <p>Camera feed ready</p>
                  </div>
                )}
                {streaming &&
                  smoothedFaces.map((face) => {
                    const isUnknown =
                      !face.name ||
                      face.name.toLowerCase() === 'unknown' ||
                      face.name.toLowerCase() === 'unknown subject'
                    const color = getEmotionColor(face.emotion)
                    return (
                      <div
                        key={face.id}
                        className={`face-box ${hoveredFaceId === face.id ? 'highlighted' : ''}`}
                        onMouseEnter={() => setHoveredFaceId(face.id)}
                        onMouseLeave={() => setHoveredFaceId(null)}
                        style={{
                          left: `${face.x}%`,
                          top: `${face.y}%`,
                          width: `${face.w}%`,
                          height: `${face.h}%`,
                          borderColor: hoveredFaceId === face.id ? '#fff' : color,
                          borderStyle: isUnknown ? 'dashed' : 'solid',
                          boxShadow: hoveredFaceId === face.id ? `0 0 25px 2px ${color}` : `0 0 10px ${color}`,
                          zIndex: hoveredFaceId === face.id ? 100 : 1,
                          transition: 'none',
                          willChange: 'left, top, width, height',
                        }}
                      >
                        <span
                          style={{
                            backgroundColor: isUnknown ? '#000000' : 'rgba(15, 23, 42, 0.85)',
                            color: color,
                            border: `1px solid ${color}`,
                          }}
                        >
                          {isUnknown ? (
                            <>
                              <HelpCircle size={12} className="inline-icon" /> Unknown Subject - {face.emotion}
                            </>
                          ) : (
                            <>
                              <User size={12} className="inline-icon" /> {face.name} - {face.emotion}
                            </>
                          )}
                        </span>
                      </div>
                    )
                  })}
              </div>
            )}
          </div>

          <div className="controls">
            {isSnapshotMode ? (
              <button className="secondary-btn" onClick={resumeLive}>
                <Camera size={16} /> Retake
              </button>
            ) : !streaming ? (
              <button className="primary-btn" onClick={startCamera}>
                <CircleDot size={16} color="#ef4444" /> Start Camera
              </button>
            ) : (
              <button className="secondary-btn" onClick={stopCamera}>
                <CameraOff size={16} /> Stop Camera
              </button>
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

        <article className="glass-card stats-card" style={{ display: 'flex', flexDirection: 'column' }}>
          {isSnapshotMode ? (
            <>
              <h3>Snapshot Results</h3>
              {isSimulating && (
                <div className="processing-status-container" style={{ marginBottom: '1.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <span className="status-text kicker" style={{ color: 'var(--accent-blue)' }}>
                      {statusText}
                    </span>
                    <span className="status-text kicker">{simulatedProgress}%</span>
                  </div>
                  <div className="progress-bar-bg" style={{ height: '6px', borderRadius: '3px' }}>
                    <div
                      className="progress-bar-fill shimmer-bg"
                      style={{ width: `${simulatedProgress}%`, height: '100%', borderRadius: '3px' }}
                    ></div>
                  </div>
                </div>
              )}
              {isSimulating && simulatedProgress >= 30 && !snapshotResult && (
                <div className="results-list">
                  <SkeletonCard />
                  <SkeletonCard />
                </div>
              )}
              {!isSimulating && snapshotResult && snapshotResult.results.length === 0 && (
                <div className="premium-empty-state" style={{ flexGrow: 1, display: 'flex', alignItems: 'center' }}>
                  <StatusCard
                    icon={User}
                    title="Zero Matches"
                    subtitle="The AI detector could not identify any clear faces in this snapshot."
                    variant="warning"
                  />
                </div>
              )}
              {!isSimulating && snapshotResult && snapshotResult.results.length > 0 && (
                <div className="results-list" style={{ flexGrow: 1, overflowY: 'auto' }}>
                  <div className="badge-row fade-in" style={{ marginBottom: '1rem' }}>
                    <span className="badge">Faces: {snapshotResult.n_faces}</span>
                    <span className="badge">Identified: {snapshotResult.n_identified}</span>
                    <span className="badge">Unknown: {snapshotResult.n_faces - snapshotResult.n_identified}</span>
                  </div>
                  {(() => {
                    const isKnownName = (n) =>
                      n && n.toUpperCase() !== 'UNKNOWN' && n.toLowerCase() !== 'unknown subject'
                    const known = snapshotResult.results.filter((p) => isKnownName(p.name))
                    const unknown = snapshotResult.results.filter((p) => !isKnownName(p.name))
                    return (
                      <>
                        {known.length > 0 && (
                          <div className="result-section known-section fade-in">
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                              <User size={16} style={{ color: 'var(--success)' }} />
                              <h4 className="section-title">Verified Members</h4>
                            </div>
                            <div className="section-divider known-divider"></div>
                            {known.map((face, idx) => (
                              <FaceCard
                                key={`snap-known-${face.face_idx}`}
                                face={{
                                  id: face.face_idx,
                                  name: face.name,
                                  emotion: face.emotion,
                                  face_image: face.face_image,
                                }}
                                delay={idx * 0.1}
                                onHover={setHoveredFaceId}
                                isHighlighted={hoveredFaceId === face.face_idx}
                              />
                            ))}
                          </div>
                        )}
                        {unknown.length > 0 && (
                          <div
                            className="result-section unknown-section fade-in"
                            style={{ marginTop: known.length > 0 ? '1.5rem' : '0' }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                              <HelpCircle size={16} style={{ color: 'var(--danger)' }} />
                              <h4 className="section-title">Unknown Subjects</h4>
                            </div>
                            <div className="section-divider unknown-divider"></div>
                            {unknown.map((face, idx) => (
                              <FaceCard
                                key={`snap-unknown-${face.face_idx}`}
                                face={{
                                  id: face.face_idx,
                                  name: face.name,
                                  emotion: face.emotion,
                                  face_image: face.face_image,
                                }}
                                delay={idx * 0.1}
                                onHover={setHoveredFaceId}
                                isHighlighted={hoveredFaceId === face.face_idx}
                              />
                            ))}
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
              {!streaming ? (
                <div className="premium-empty-state" style={{ flexGrow: 1, display: 'flex', alignItems: 'center' }}>
                  <StatusCard
                    icon={Cpu}
                    title="System Dormant"
                    subtitle="Start the camera feed to trigger dynamic emotion analysis and face tracking."
                  />
                </div>
              ) : (
                <div className="results-list" style={{ flexGrow: 1, overflowY: 'auto' }}>
                  <div className="badge-row fade-in" style={{ marginBottom: '1rem' }}>
                    <span className="badge">Active Faces: {detectedFaces.length}</span>
                    <span className="badge">
                      Identified: {detectedFaces.filter((f) => f.name && f.name.toLowerCase() !== 'unknown' && f.name.toLowerCase() !== 'unknown subject').length}
                    </span>
                  </div>
                  {detectedFaces.length === 0 ? (
                    <div
                      className="premium-empty-state"
                      style={{ height: '70%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                    >
                      <p className="muted" style={{ fontSize: '0.85rem' }}>
                        No faces currently in view...
                      </p>
                    </div>
                  ) : (
                    detectedFaces.map((face, idx) => (
                      <FaceCard
                        key={face.id}
                        face={face}
                        delay={idx * 0.05}
                        onHover={setHoveredFaceId}
                        isHighlighted={hoveredFaceId === face.id}
                      />
                    ))
                  )}
                </div>
              )}
            </>
          )}
        </article>
      </div>
      <Legend />
    </section>
  )
}
