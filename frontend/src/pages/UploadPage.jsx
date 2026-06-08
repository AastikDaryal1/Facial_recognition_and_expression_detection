/**
 * src/pages/UploadPage.jsx — Shubh branch layout with Shubh-II JWT auth
 */
import React, { useState, useEffect, useRef } from 'react'
import {
  Upload, User, HelpCircle, Camera, CameraOff, CloudUpload,
  Sparkles, Zap, Cpu, CheckCircle, FileImage, RefreshCw,
  AlertTriangle, ShieldCheck,
} from 'lucide-react'
import { predictImage } from '../api'

const emotionColors = {
  Angry: '#FF4C4C', Fear: '#8E44AD', Happy: '#FFD93D',
  Neutral: '#BDC3C7', Sad: '#3498DB', Surprise: '#FF9F43',
}
function getEmotionColor(emotion) {
  if (!emotion) return '#BDC3C7'
  const match = Object.keys(emotionColors).find(e => e.toLowerCase() === emotion.toLowerCase())
  return match ? emotionColors[match] : '#BDC3C7'
}

function FaceAvatar({ face, color }) {
  const isUnknown = !face.name || face.name.toUpperCase() === 'UNKNOWN' || face.name.toLowerCase() === 'unknown subject'
  return (
    <div className={`avatar-square ${isUnknown ? 'unknown-avatar' : 'known-avatar'}`} style={{
      borderColor: color, boxShadow: `0 0 10px ${color}80`, borderStyle: isUnknown ? 'dashed' : 'solid'
    }}>
      {face.face_image ? (
        <img src={face.face_image} alt={face.name || 'Unknown'} className="avatar-image"
          onError={(e) => { e.target.style.display = 'none' }} />
      ) : (
        <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(30,41,59,0.8)' }}>
          {isUnknown ? <HelpCircle size={20} style={{ color }} /> : <User size={20} style={{ color }} />}
        </div>
      )}
    </div>
  )
}

function FaceCard({ face, delay, onHover, isHighlighted }) {
  const isUnknown = !face.name || face.name.toLowerCase() === 'unknown' || face.name.toLowerCase() === 'unknown subject'
  const color = getEmotionColor(face.emotion)
  return (
    <div
      className={`face-card fade-in-up ${isUnknown ? 'face-card-unknown' : 'face-card-known'} ${isHighlighted ? 'highlighted' : ''}`}
      onMouseEnter={() => onHover && onHover(face.face_idx !== undefined ? face.face_idx : face.id)}
      onMouseLeave={() => onHover && onHover(null)}
      style={{
        borderColor: isHighlighted ? '#fff' : `${color}44`, borderStyle: isUnknown ? 'dashed' : 'solid',
        boxShadow: isHighlighted ? `0 0 25px ${color}` : 'none', animationDelay: delay ? `${delay}s` : '0s',
        zIndex: isHighlighted ? 10 : 1,
        background: isHighlighted ? `linear-gradient(135deg, rgba(30,41,59,0.8), ${color}11)` : 'var(--card)'
      }}
    >
      <div className="face-card-left"><FaceAvatar face={face} color={color} /></div>
      <div className="face-card-middle">
        <p style={{ fontWeight: 600, color: isUnknown ? 'var(--muted)' : 'var(--text)' }}>
          {isUnknown ? 'Unrecognised' : face.name}
        </p>
        <span style={{ fontSize: '0.75rem', padding: '2px 8px', borderRadius: '4px',
          backgroundColor: `${color}22`, color, border: `1px solid ${color}44`,
          textTransform: 'uppercase', fontWeight: 700 }}>{face.emotion}</span>
      </div>
      <div className="face-card-right">
        {isUnknown ? <HelpCircle size={18} style={{ color: 'var(--muted)', opacity: 0.5 }} /> :
          <div className="status-icon-ring" style={{ width: '28px', height: '28px', backgroundColor: 'transparent' }}>
            <User size={14} style={{ color: 'var(--success)' }} />
          </div>}
      </div>
    </div>
  )
}

function SkeletonCard() {
  return (
    <div className="face-card shimmer-bg" style={{ opacity: 0.5, borderStyle: 'solid' }}>
      <div className="face-card-left">
        <div className="avatar-square" style={{ background: 'rgba(255,255,255,0.05)', border: 'none' }}></div>
      </div>
      <div className="face-card-middle">
        <div style={{ height: '14px', width: '60%', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', marginBottom: '8px' }}></div>
        <div style={{ height: '10px', width: '40%', background: 'rgba(255,255,255,0.05)', borderRadius: '4px' }}></div>
      </div>
    </div>
  )
}

function StatusCard({ icon: Icon, title, subtitle, variant = "default" }) {
  return (
    <div className="status-card fade-in" style={{
      borderColor: variant === "warning" ? "rgba(250,204,21,0.2)" : "rgba(59,130,246,0.2)",
      background: variant === "warning" ? "rgba(250,204,21,0.05)" : "rgba(30,41,59,0.4)"
    }}>
      <div className="status-icon-ring" style={{
        color: variant === "warning" ? "var(--warning)" : "var(--accent-blue)",
        backgroundColor: variant === "warning" ? "rgba(250,204,21,0.1)" : "rgba(59,130,246,0.1)"
      }}><Icon size={24} className={variant === "default" ? "float" : ""} /></div>
      <div style={{ textAlign: 'center' }}>
        <h4 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 600 }}>{title}</h4>
        <p style={{ margin: '4px 0 0', fontSize: '0.9rem', color: 'var(--muted)', maxWidth: '240px' }}>{subtitle}</p>
      </div>
    </div>
  )
}

function Legend() {
  return (
    <div className="dual-legend glass-card fade-in-up" style={{ animationDelay: '0.3s' }}>
      <div className="legend-section">
        <h4 className="legend-title">Emotion (Color)</h4>
        <div className="legend-items">
          {Object.entries(emotionColors).map(([emotion, color]) => (
            <div key={emotion} className="legend-item">
              <div className="legend-color" style={{ backgroundColor: color, boxShadow: `0 0 10px ${color}` }}></div>
              <span className="legend-text">{emotion}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="legend-divider"></div>
      <div className="legend-section">
        <h4 className="legend-title">Identity (Style)</h4>
        <div className="legend-items">
          <div className="legend-item"><div className="legend-style-box solid"></div><span className="legend-text">Known Member</span></div>
          <div className="legend-item"><div className="legend-style-box dashed"></div><span className="legend-text">Unknown Subject</span></div>
        </div>
      </div>
    </div>
  )
}

// ── Main Component ───────────────────────────────────────────────────────────
export default function UploadPage() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [dragActive, setDragActive] = useState(false)
  const [hoveredFaceId, setHoveredFaceId] = useState(null)
  const [simulatedProgress, setSimulatedProgress] = useState(0)
  const [isSimulating, setIsSimulating] = useState(false)
  const [statusText, setStatusText] = useState('')
  const [isConverting, setIsConverting] = useState(false)
  const [previewError, setPreviewError] = useState(false)
  const [imgDims, setImgDims] = useState({ w: 1, h: 1 })
  const [showCamera, setShowCamera] = useState(false)
  const videoRef = useRef(null)

  useEffect(() => {
    if (!file) { setPreviewUrl(''); setPreviewError(false); return }
    setPreviewError(false)
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => {
    return () => { if (videoRef.current?.srcObject) videoRef.current.srcObject.getTracks().forEach(t => t.stop()) }
  }, [])

  const handleSelection = async (selectedFile) => {
    setResult(null); setError('')
    if (!selectedFile) return
    if (selectedFile.size > 10 * 1024 * 1024) { setError('File too large (max 10MB).'); return }
    const ext = selectedFile.name?.split('.').pop().toLowerCase()
    if (['heic', 'heif'].includes(ext)) {
      setIsConverting(true)
      try {
        const heic2anyMod = await import('heic2any')
        const heic2any = heic2anyMod.default || heic2anyMod
        const blob = await heic2any({ blob: selectedFile, toType: 'image/jpeg', quality: 0.9 })
        const converted = new File([Array.isArray(blob) ? blob[0] : blob],
          selectedFile.name.replace(/\.[^/.]+$/, '.jpg'), { type: 'image/jpeg' })
        setFile(converted)
      } catch { setFile(selectedFile) }
      finally { setIsConverting(false) }
    } else { setFile(selectedFile) }
  }

  const handleDrop = (e) => { e.preventDefault(); setDragActive(false); handleSelection(e.dataTransfer.files?.[0]) }

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      setShowCamera(true)
      setTimeout(() => { if (videoRef.current) { videoRef.current.srcObject = stream; videoRef.current.play() } }, 100)
    } catch { setError('Could not access camera.') }
  }
  const stopCamera = () => {
    if (videoRef.current?.srcObject) videoRef.current.srcObject.getTracks().forEach(t => t.stop())
    setShowCamera(false)
  }
  const capturePhoto = () => {
    if (!videoRef.current) return
    const canvas = document.createElement('canvas')
    canvas.width = videoRef.current.videoWidth; canvas.height = videoRef.current.videoHeight
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
    canvas.toBlob(blob => {
      if (blob) { handleSelection(new File([blob], 'capture.jpg', { type: 'image/jpeg' })); stopCamera() }
    }, 'image/jpeg', 0.9)
  }

  const runPrediction = async () => {
    if (!file) return
    setLoading(true); setIsSimulating(true); setSimulatedProgress(0)
    setStatusText('Scanning faces...'); setError(''); setResult(null)
    try {
      const fetchPromise = predictImage(file)
      let prog = 0
      const iv = setInterval(() => {
        prog += 2
        if (prog >= 90) clearInterval(iv)
        else {
          setSimulatedProgress(prog)
          if (prog < 30) setStatusText('Scanning faces...')
          else if (prog < 60) setStatusText('Analyzing emotions...')
          else setStatusText('Matching identities...')
        }
      }, 50)
      const data = await fetchPromise
      clearInterval(iv); setSimulatedProgress(100); setStatusText('Finalizing results...')
      setTimeout(() => { setResult(data); setIsSimulating(false); setLoading(false) }, 500)
    } catch (err) { setError(err.message || 'Prediction failed.'); setIsSimulating(false); setLoading(false) }
  }

  return (
    <section className="fade-in">
      <div className="page-heading">
        <h2>Image Upload Detection</h2>
        <p>Drop a photo, choose from device, or snap a picture.</p>
      </div>

      <div className={`upload-zone glass-card ${dragActive ? 'drag-active' : ''} ${isConverting ? 'processing' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
        onDragLeave={() => setDragActive(false)} onDrop={handleDrop}
        onClick={() => !isConverting && document.getElementById('upload-file-input').click()}>
        {isConverting ? (
          <div className="conversion-status">
            <div className="shimmer-loader" style={{ width: '40px', height: '40px', borderRadius: '50%', margin: '0 auto 1rem' }}></div>
            <p style={{ fontWeight: 600, color: 'var(--neon-cyan)' }}>Converting iPhone image...</p>
            <p className="muted" style={{ fontSize: '0.85rem' }}>Optimizing HEIC for AI detection</p>
          </div>
        ) : (
          <>
            <div className="upload-icon-wrapper">
              <div className="upload-glow"></div>
              <CloudUpload size={42} className={dragActive ? 'float' : ''} style={{ color: dragActive ? 'var(--neon-cyan)' : 'var(--accent-blue)', position: 'relative' }} />
            </div>
            <div style={{ marginBottom: '0.5rem' }}>
              <p style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text)', marginBottom: '4px' }}>
                {dragActive ? 'Drop to Initialize' : 'AI Analysis Hub'}
              </p>
              <p className="muted" style={{ fontSize: '0.85rem' }}>Drop image here or click to browse</p>
            </div>
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }} onClick={e => e.stopPropagation()}>
              <label className="secondary-btn" onClick={e => e.stopPropagation()}>
                Choose Image
                <input id="upload-file-input" type="file" accept=".jpg,.jpeg,.png,.webp,.heic,.heif" hidden
                  onChange={e => handleSelection(e.target.files?.[0])} />
              </label>
              <button className="secondary-btn" onClick={e => { e.stopPropagation(); startCamera() }}>
                <Camera size={18} style={{ marginRight: '8px' }} /> Open Camera
              </button>
            </div>
          </>
        )}
      </div>

      {showCamera && (
        <div className="glass-card fade-in-up" style={{ marginTop: '1.5rem', textAlign: 'center' }}>
          <h3 style={{ marginBottom: '1rem' }}>Camera Capture</h3>
          <video ref={videoRef} autoPlay playsInline muted
            style={{ width: '100%', maxWidth: '640px', borderRadius: '8px', border: '1px solid #334155', marginBottom: '1rem' }} />
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
            <button className="primary-btn" onClick={capturePhoto}><Camera size={18} style={{ marginRight: '8px' }} /> Snap Photo</button>
            <button className="secondary-btn" onClick={stopCamera}>Cancel</button>
          </div>
        </div>
      )}

      <div className="upload-layout">
        <article className="glass-card preview-panel" style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
            <h3 style={{ fontSize: '1.1rem' }}>Source Preview</h3>
            {previewUrl && <span className="badge" style={{ background: 'rgba(59,130,246,0.1)', color: 'var(--accent-blue)', border: '1px solid rgba(59,130,246,0.2)' }}>{imgDims.w}x{imgDims.h}px</span>}
          </div>
          <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            {previewUrl ? (
              <div className="preview-image-container fade-in" style={{ boxShadow: '0 0 40px rgba(0,0,0,0.5)', borderRadius: '1rem', overflow: 'hidden' }}>
                {previewError ? (
                  <div className="preview-fallback glass-card" style={{ padding: '3rem 1.5rem', textAlign: 'center', background: 'rgba(15,23,42,0.4)' }}>
                    <div style={{ background: 'rgba(59,130,246,0.1)', width: '60px', height: '60px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1rem' }}>
                      <ShieldCheck size={30} style={{ color: 'var(--accent-blue)' }} />
                    </div>
                    <p style={{ fontWeight: 600 }}>Apple Format Detected</p>
                    <p className="muted" style={{ fontSize: '0.85rem' }}>Visual preview unavailable, but AI analysis is fully supported.</p>
                  </div>
                ) : (
                  <img src={previewUrl} alt="Selected preview"
                    className={`preview-image ${isSimulating && simulatedProgress < 90 ? 'processing' : ''}`}
                    onLoad={e => setImgDims({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
                    onError={() => setPreviewError(true)}
                    style={{ width: '100%', height: 'auto', borderRadius: '0' }} />
                )}
                {isSimulating && simulatedProgress < 100 && <div className="scanning-beam"></div>}
                {result && simulatedProgress >= 40 && result.results.map((face, idx) => {
                  const isUnknown = !face.name || face.name.toLowerCase() === 'unknown' || face.name.toLowerCase() === 'unknown subject'
                  const color = getEmotionColor(face.emotion)
                  const effectiveW = Math.min(imgDims.w, 640)
                  const scale = effectiveW / imgDims.w
                  const effectiveH = imgDims.h * scale
                  return (
                    <div key={idx}
                      className={`face-box scale-in ${hoveredFaceId === face.face_idx ? 'highlighted' : ''}`}
                      onMouseEnter={() => setHoveredFaceId(face.face_idx)}
                      onMouseLeave={() => setHoveredFaceId(null)}
                      style={{
                        left: `${(face.x / effectiveW) * 100}%`, top: `${(face.y / effectiveH) * 100}%`,
                        width: `${(face.w / effectiveW) * 100}%`, height: `${(face.h / effectiveH) * 100}%`,
                        borderColor: (isSimulating && simulatedProgress < 70) ? '#64748b' : (hoveredFaceId === face.face_idx ? '#fff' : color),
                        borderStyle: isUnknown && (!isSimulating || simulatedProgress >= 70) ? 'dashed' : 'solid',
                        boxShadow: (isSimulating && simulatedProgress < 70) ? 'none' : (hoveredFaceId === face.face_idx ? `0 0 25px 2px ${color}` : `0 0 10px ${color}`),
                        zIndex: hoveredFaceId === face.face_idx ? 100 : 1
                      }}>
                      <span className={isSimulating && simulatedProgress < 70 ? 'label-analyzing' : ''} style={{
                        backgroundColor: isUnknown && (!isSimulating || simulatedProgress >= 70) ? '#000' : 'rgba(15,23,42,0.85)',
                        color: (isSimulating && simulatedProgress < 70) ? '#e2e8f0' : color,
                        border: `1px solid ${(isSimulating && simulatedProgress < 70) ? '#64748b' : color}`
                      }}>
                        {(isSimulating && simulatedProgress < 70) ? 'Analyzing...' :
                          isUnknown ? <><HelpCircle size={12} className="inline-icon" /> Unknown Subject - {face.emotion}</> :
                          <><User size={12} className="inline-icon" /> {face.name} - {face.emotion}</>}
                      </span>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="premium-empty-state">
                <StatusCard icon={CloudUpload} title="No Image Selected" subtitle="Upload a JPG or PNG to begin AI processing." />
              </div>
            )}
          </div>
          <button className="primary-btn" onClick={runPrediction} disabled={loading || !file} style={{ width: '100%', padding: '0.8rem' }}>
            {loading ? <><Zap size={18} className="pulse" /> Processing...</> : <><Cpu size={18} /> Detect Face & Emotion</>}
          </button>
        </article>

        <article className="glass-card result-panel">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
            <h3 style={{ fontSize: '1.1rem' }}>AI Analysis Results</h3>
            {result && !isSimulating && <span className="badge" style={{ background: 'rgba(74,222,128,0.1)', color: 'var(--success)', border: '1px solid rgba(74,222,128,0.2)' }}>COMPLETED</span>}
          </div>

          {isSimulating && (
            <div className="processing-status-container fade-in" style={{ marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span className="status-text kicker" style={{ color: 'var(--accent-blue)' }}>{statusText}</span>
                <span className="status-text kicker">{simulatedProgress}%</span>
              </div>
              <div className="progress-bar-bg" style={{ height: '6px', borderRadius: '3px' }}>
                <div className="progress-bar-fill shimmer-bg" style={{ width: `${simulatedProgress}%`, height: '100%', borderRadius: '3px' }}></div>
              </div>
            </div>
          )}

          {error && (
            <div className="fade-in" style={{ padding: '1rem 0' }}>
              <StatusCard icon={Zap} title="System Notice" subtitle={error} variant="warning" />
            </div>
          )}

          {isSimulating && simulatedProgress >= 30 && !result && (
            <div className="results-list"><SkeletonCard /><SkeletonCard /><SkeletonCard /></div>
          )}

          {!isSimulating && !error && result && result.results.length === 0 && (
            <div className="premium-empty-state">
              <StatusCard icon={User} title="Zero Matches" subtitle="The AI detector could not identify any clear faces in this image." variant="warning" />
            </div>
          )}

          {!isSimulating && !error && result && result.results.length > 0 && (
            <div className="results-list">
              <div className="badge-row fade-in" style={{ marginBottom: '1rem' }}>
                <span className="badge">Faces: {result.n_faces}</span>
                <span className="badge">Identified: {result.n_identified}</span>
                <span className="badge">Unknown: {result.n_faces - result.n_identified}</span>
              </div>
              {(() => {
                const isKnownName = n => n && n.toUpperCase() !== 'UNKNOWN' && n.toLowerCase() !== 'unknown subject' && n.toLowerCase() !== 'unrecognised'
                const known = result.results.filter(p => isKnownName(p.name))
                const unknown = result.results.filter(p => !isKnownName(p.name))
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
                          <FaceCard key={`known-${face.face_idx}`} face={face} delay={idx * 0.1}
                            onHover={setHoveredFaceId} isHighlighted={hoveredFaceId === face.face_idx} />
                        ))}
                      </div>
                    )}
                    {unknown.length > 0 && (
                      <div className="result-section unknown-section fade-in" style={{ marginTop: known.length > 0 ? '1.5rem' : '0' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                          <HelpCircle size={16} style={{ color: 'var(--danger)' }} />
                          <h4 className="section-title">Unknown Subjects</h4>
                        </div>
                        <div className="section-divider unknown-divider"></div>
                        {unknown.map((face, idx) => (
                          <FaceCard key={`unknown-${face.face_idx}`} face={face} delay={idx * 0.1}
                            onHover={setHoveredFaceId} isHighlighted={hoveredFaceId === face.face_idx} />
                        ))}
                      </div>
                    )}
                  </>
                )
              })()}
            </div>
          )}

          {!isSimulating && !error && !result && (
            <div className="premium-empty-state">
              <StatusCard icon={Cpu} title="Ready for Analysis" subtitle="Upload an image and click the button to trigger AI face & emotion detection." />
            </div>
          )}
        </article>
      </div>
      <Legend />
    </section>
  )
}
