/**
 * src/pages/UploadPage.jsx
 * ────────────────────────
 * Premium image upload and static analysis page.
 * Keeps Shubh's stable, high-fidelity Apple HEIC dynamic transcoding,
 * detailed mock progress simulation, coordinate scaling,
 * and high-fidelity layout.
 */

import React, { useState, useEffect, useRef } from 'react'
import {
  Upload,
  User,
  HelpCircle,
  Clock,
  Sparkles,
  Zap,
  CheckCircle,
  FileImage,
  RefreshCw,
  AlertTriangle,
} from 'lucide-react'
import { predictImage } from '../api'

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

// ── Main Component ───────────────────────────────────────────────────────────

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [statusText, setStatusText] = useState('')
  const [result, setResult] = useState(null)
  const [errorMsg, setErrorMsg] = useState('')
  const [heicLoading, setHeicLoading] = useState(false)
  const [hoveredFaceId, setHoveredFaceId] = useState(null)
  const fileInputRef = useRef(null)

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = async (e) => {
    e.preventDefault()
    e.stopPropagation()
    const files = e.dataTransfer.files
    if (files && files[0]) {
      await processSelectedFile(files[0])
    }
  }

  const handleFileChange = async (e) => {
    const files = e.target.files
    if (files && files[0]) {
      await processSelectedFile(files[0])
    }
  }

  const processSelectedFile = async (file) => {
    setErrorMsg('')
    setResult(null)
    setPreviewUrl(null)
    setSelectedFile(null)

    const ext = file.name.split('.').pop().toLowerCase()
    const isHeic = ext === 'heic' || ext === 'heif'

    if (isHeic) {
      setHeicLoading(true)
      setStatusText('Apple HEIC detected. Auto-transcoding in browser...')
      try {
        // Dynamically load heic2any to keep initial bundle light and perform fast transcoding
        const heic2anyModule = await import('heic2any')
        const heic2any = heic2anyModule.default || heic2anyModule
        const jpegBlob = await heic2any({
          blob: file,
          toType: 'image/jpeg',
          quality: 0.9,
        })
        const transcodedFile = new File([jpegBlob], file.name.replace(/\.[^/.]+$/, '.jpg'), {
          type: 'image/jpeg',
        })
        setSelectedFile(transcodedFile)
        setPreviewUrl(URL.createObjectURL(transcodedFile))
        console.log('Apple HEIC Transcoding success:', transcodedFile)
      } catch (err) {
        console.error('HEIC Transcoding error:', err)
        setErrorMsg('Apple HEIC transcoding failed. Please upload a standard JPG/PNG.')
      } finally {
        setHeicLoading(false)
      }
    } else {
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
    }
  }

  const triggerUpload = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click()
    }
  }

  const uploadAndProcess = async () => {
    if (!selectedFile) return

    setIsProcessing(true)
    setUploadProgress(0)
    setStatusText('Uploading photo...')
    setErrorMsg('')

    try {
      // 1. Simulating Progress Timeline
      let currentProgress = 0
      const interval = setInterval(() => {
        currentProgress += 2
        if (currentProgress >= 90) {
          clearInterval(interval)
        } else {
          setUploadProgress(currentProgress)
          if (currentProgress < 25) setStatusText('Uploading to API endpoint...')
          else if (currentProgress < 50) setStatusText('Triggering cascade face-detectors...')
          else if (currentProgress < 75) setStatusText('Performing FaceNet vector representation...')
          else setStatusText('Matching identity models & classifying emotions...')
        }
      }, 70)

      // 2. Perform predictImage central API call
      const data = await predictImage(selectedFile)

      clearInterval(interval)
      setUploadProgress(100)
      setStatusText('Finalizing results...')

      setTimeout(() => {
        setResult(data)
        setIsProcessing(false)
      }, 500)
    } catch (err) {
      console.error(err)
      setErrorMsg(err.message || 'Image analysis failed. Please verify API server state.')
      setIsProcessing(false)
    }
  }

  const resetUpload = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setResult(null)
    setErrorMsg('')
    setUploadProgress(0)
    setIsProcessing(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <section className="fade-in upload-section">
      <div className="page-heading">
        <h2>Static Image Inference</h2>
        <p>Upload a group photo or single portrait. Supports standard formats & Apple HEIC.</p>
      </div>

      <div className="upload-layout">
        <article className="glass-card upload-card">
          {!previewUrl ? (
            <div
              className={`dropzone ${heicLoading ? 'processing' : ''}`}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={heicLoading ? null : triggerUpload}
            >
              {heicLoading ? (
                <>
                  <RefreshCw size={48} className="spin" style={{ color: 'var(--accent-blue)' }} />
                  <p style={{ marginTop: '1.25rem' }}>{statusText}</p>
                </>
              ) : (
                <>
                  <Upload size={48} />
                  <p>Drag and drop image here or click to browse</p>
                  <p className="muted" style={{ fontSize: '0.8rem', marginTop: '4px' }}>
                    Supports JPG, PNG, WEBP, and Apple HEIC
                  </p>
                </>
              )}
            </div>
          ) : (
            <div className="preview-image-container" style={{ width: '100%', height: '100%', margin: 0 }}>
              <img
                src={previewUrl}
                alt="Upload preview"
                className={`preview-image ${isProcessing && uploadProgress < 90 ? 'processing' : ''}`}
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
              {isProcessing && uploadProgress < 100 && <div className="scanning-beam"></div>}
              {result &&
                uploadProgress >= 100 &&
                result.results.map((face, idx) => {
                  const isUnknown =
                    !face.name ||
                    face.name.toLowerCase() === 'unknown' ||
                    face.name.toLowerCase() === 'unknown subject'
                  const color = getEmotionColor(face.emotion)

                  return (
                    <div
                      key={`res-${idx}`}
                      className={`face-box scale-in ${hoveredFaceId === face.face_idx ? 'highlighted' : ''}`}
                      onMouseEnter={() => setHoveredFaceId(face.face_idx)}
                      onMouseLeave={() => setHoveredFaceId(null)}
                      style={{
                        left: `${face.x}%`,
                        top: `${face.y}%`,
                        width: `${face.w}%`,
                        height: `${face.h}%`,
                        borderColor: hoveredFaceId === face.face_idx ? '#fff' : color,
                        borderStyle: isUnknown ? 'dashed' : 'solid',
                        boxShadow: hoveredFaceId === face.face_idx ? `0 0 25px 2px ${color}` : `0 0 10px ${color}`,
                        zIndex: hoveredFaceId === face.face_idx ? 100 : 1,
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
                  );
                })}
            </div>
          )}

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".jpg,.jpeg,.png,.webp,.heic,.heif"
            style={{ display: 'none' }}
          />

          <div className="controls">
            {previewUrl && !isProcessing && (
              <>
                <button className="secondary-btn" onClick={resetUpload}>
                  Clear
                </button>
                {!result && (
                  <button className="primary-btn" onClick={uploadAndProcess} style={{ marginLeft: 'auto' }}>
                    <Sparkles size={16} /> Process Image
                  </button>
                )}
              </>
            )}
          </div>
          {errorMsg && (
            <p className="error" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <AlertTriangle size={16} /> {errorMsg}
            </p>
          )}
        </article>

        <article className="glass-card stats-card" style={{ display: 'flex', flexDirection: 'column' }}>
          <h3>Analysis Output</h3>
          {isProcessing ? (
            <div className="processing-status-container" style={{ margin: 'auto 0' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span className="status-text kicker" style={{ color: 'var(--accent-blue)' }}>
                  {statusText}
                </span>
                <span className="status-text kicker">{uploadProgress}%</span>
              </div>
              <div className="progress-bar-bg" style={{ height: '6px', borderRadius: '3px' }}>
                <div
                  className="progress-bar-fill shimmer-bg"
                  style={{ width: `${uploadProgress}%`, height: '100%', borderRadius: '3px' }}
                ></div>
              </div>
            </div>
          ) : !result ? (
            <div className="premium-empty-state" style={{ flexGrow: 1, display: 'flex', alignItems: 'center' }}>
              <StatusCard
                icon={FileImage}
                title="Awaiting Image"
                subtitle="Upload a group photograph or dynamic vector crop to analyze facial emotions and identify known team members."
              />
            </div>
          ) : (
            <div className="results-list" style={{ flexGrow: 1, overflowY: 'auto' }}>
              <div className="badge-row fade-in" style={{ marginBottom: '1rem' }}>
                <span className="badge">Faces Detected: {result.n_faces}</span>
                <span className="badge">Identified: {result.n_identified}</span>
                <span className="badge">Latency: {result.elapsed_s}s</span>
              </div>

              {result.results.length === 0 ? (
                <div className="premium-empty-state" style={{ height: '70%', display: 'flex', alignItems: 'center' }}>
                  <StatusCard
                    icon={User}
                    title="Zero Faces Found"
                    subtitle="The AI model could not detect any faces in this image. Please make sure faces are clearly visible and try again."
                    variant="warning"
                  />
                </div>
              ) : (
                (() => {
                  const isKnownName = (n) =>
                    n && n.toUpperCase() !== 'UNKNOWN' && n.toLowerCase() !== 'unknown subject'
                  const known = result.results.filter((p) => isKnownName(p.name))
                  const unknown = result.results.filter((p) => !isKnownName(p.name))
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
                              key={`upload-known-${face.face_idx}`}
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
                              key={`upload-unknown-${face.face_idx}`}
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
                })()
              )}
            </div>
          )}
        </article>
      </div>
      <Legend />
    </section>
  )
}
