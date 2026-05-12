/**
 * src/pages/UploadPage.jsx
 * ─────────────────────────
 * Image upload detection page.
 * Identical to the original UploadPage, but uses api.js for all fetch calls
 * instead of the old X-API-Key header.
 */

import { useEffect, useRef, useState } from 'react'
import { Camera, CloudUpload, HelpCircle, User } from 'lucide-react'
import { predictImage } from '../api'
import FaceCard    from '../components/FaceCard'
import SkeletonCard from '../components/SkeletonCard'
import Legend      from '../components/Legend'

const allowedImageExtensions = ['.jpg','.jpeg','.png','.webp','.bmp','.gif','.heic','.heif']

function isSupportedImageFile(file) {
  if (!file) return false
  const mimeType = file.type?.toLowerCase() || ''
  if (mimeType.startsWith('image/')) return true
  const lowerName = file.name?.toLowerCase() || ''
  return allowedImageExtensions.some((ext) => lowerName.endsWith(ext))
}

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

export default function UploadPage() {
  const [file, setFile]                     = useState(null)
  const [previewUrl, setPreviewUrl]         = useState('')
  const [loading, setLoading]               = useState(false)
  const [result, setResult]                 = useState(null)
  const [error, setError]                   = useState('')
  const [dragActive, setDragActive]         = useState(false)
  const [hoveredFaceId, setHoveredFaceId]   = useState(null)
  const [simulatedProgress, setSimulatedProgress] = useState(0)
  const [isSimulating, setIsSimulating]     = useState(false)
  const [statusText, setStatusText]         = useState('')
  const [showCamera, setShowCamera]         = useState(false)
  const [imgDims, setImgDims]               = useState({ w: 1, h: 1 })
  const videoRef = useRef(null)

  const MAX_SIZE_MB = import.meta.env.VITE_MAX_UPLOAD_SIZE_MB || 5

  useEffect(() => {
    if (!file) return undefined
    const nextPreview = URL.createObjectURL(file)
    setPreviewUrl(nextPreview)
    return () => URL.revokeObjectURL(nextPreview)
  }, [file])

  useEffect(() => {
    return () => {
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((t) => t.stop())
      }
    }
  }, [])

  const handleSelection = (selectedFile) => {
    setResult(null)
    setError('')
    if (!selectedFile) return
    if (!isSupportedImageFile(selectedFile)) {
      setError('Please upload a valid image file.')
      return
    }
    if (selectedFile.size > MAX_SIZE_MB * 1024 * 1024) {
      window.alert(`File too large. Max size is ${MAX_SIZE_MB}MB.`)
      return
    }
    setFile(selectedFile)
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setDragActive(false)
    handleSelection(event.dataTransfer.files?.[0])
  }

  const runPrediction = async () => {
    if (!file) { setError('Upload an image before running detection.'); return }
    setLoading(true)
    setIsSimulating(true)
    setSimulatedProgress(0)
    setStatusText('Scanning faces...')
    setError('')
    setResult(null)

    try {
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

      setTimeout(() => {
        setResult(data)
        setIsSimulating(false)
        setLoading(false)
      }, 500)
    } catch (fetchError) {
      setError(fetchError.message || 'Could not reach backend API.')
      setIsSimulating(false)
      setLoading(false)
    }
  }

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      setShowCamera(true)
      setTimeout(() => {
        if (videoRef.current) { videoRef.current.srcObject = stream; videoRef.current.play() }
      }, 100)
    } catch {
      setError('Could not access camera. Check permissions.')
    }
  }

  const stopCamera = () => {
    videoRef.current?.srcObject?.getTracks().forEach((t) => t.stop())
    setShowCamera(false)
  }

  const capturePhoto = () => {
    if (!videoRef.current) return
    const canvas = document.createElement('canvas')
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0)
    canvas.toBlob((blob) => {
      if (blob) { handleSelection(new File([blob], 'capture.jpg', { type: 'image/jpeg' })); stopCamera() }
    }, 'image/jpeg', 0.9)
  }

  return (
    <section className="fade-in">
      <div className="page-heading">
        <h2>Image Upload Detection</h2>
        <p>Drop a photo, choose from device, or snap a picture.</p>
      </div>

      <div
        className={`upload-zone glass-card ${dragActive ? 'drag-active' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        <CloudUpload size={34} />
        <p>Drop image here or select from file picker</p>
        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
          <label className="secondary-btn">
            Choose Image
            <input type="file" accept="image/*" hidden onChange={(e) => handleSelection(e.target.files?.[0])} />
          </label>
          <button className="secondary-btn" onClick={startCamera}>
            <Camera size={18} style={{ marginRight: '8px' }} /> Open Camera
          </button>
        </div>
      </div>

      {showCamera && (
        <div className="glass-card fade-in-up" style={{ marginTop: '1.5rem', textAlign: 'center' }}>
          <h3 style={{ marginBottom: '1rem' }}>Camera Capture</h3>
          <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', maxWidth: '640px', borderRadius: '8px', border: '1px solid #334155', marginBottom: '1rem' }} />
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
            <button className="primary-btn" onClick={capturePhoto}><Camera size={18} style={{ marginRight: '8px' }} /> Snap Photo</button>
            <button className="secondary-btn" onClick={stopCamera}>Cancel</button>
          </div>
        </div>
      )}

      <div className="upload-layout">
        <article className="glass-card preview-panel">
          <h3>Preview</h3>
          {previewUrl ? (
            <div className="preview-image-container">
              <img
                src={previewUrl}
                alt="Selected preview"
                className={`preview-image ${isSimulating && simulatedProgress < 90 ? 'processing' : ''}`}
                onLoad={(e) => setImgDims({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
              />
              {isSimulating && simulatedProgress < 100 && <div className="scanning-beam"></div>}
              {result && simulatedProgress >= 40 && result.results.map((face, idx) => {
                const isUnknown = !face.name || face.name.toLowerCase() === 'unknown'
                const color = getEmotionColor(face.emotion)
                const effectiveW = Math.min(imgDims.w, 640)
                const scale = effectiveW / imgDims.w
                const effectiveH = imgDims.h * scale
                return (
                  <div
                    key={idx}
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
                    }}
                  >
                    <span
                      className={isSimulating && simulatedProgress < 70 ? 'label-analyzing' : ''}
                      style={{
                        backgroundColor: isUnknown && (!isSimulating || simulatedProgress >= 70) ? '#000000' : 'rgba(15, 23, 42, 0.85)',
                        color: (isSimulating && simulatedProgress < 70) ? '#e2e8f0' : color,
                        border: `1px solid ${(isSimulating && simulatedProgress < 70) ? '#64748b' : color}`
                      }}
                    >
                      {(isSimulating && simulatedProgress < 70) ? <>Analyzing...</> :
                        isUnknown ? <><HelpCircle size={12} className="inline-icon" /> UNKNOWN - {face.emotion}</> :
                        <><User size={12} className="inline-icon" /> {face.name} - {face.emotion}</>}
                    </span>
                  </div>
                )
              })}
            </div>
          ) : (
            <p className="muted">No image selected yet.</p>
          )}
          <button className="primary-btn" onClick={runPrediction} disabled={loading}>
            {loading ? 'Processing...' : 'Detect Face & Emotion'}
          </button>
        </article>

        <article className="glass-card result-panel">
          <h3>Results</h3>
          {isSimulating && (
            <div className="processing-status-container">
              <div className="status-text">{statusText}</div>
              <div className="progress-bar-bg"><div className="progress-bar-fill" style={{ width: `${simulatedProgress}%` }}></div></div>
            </div>
          )}
          {error && <p className="error">{error}</p>}
          {isSimulating && simulatedProgress >= 30 && !result && (
            <div className="results-list"><SkeletonCard /><SkeletonCard /></div>
          )}
          {!isSimulating && !error && result && result.results.length === 0 && (
            <p className="warning">No face detected in the selected image.</p>
          )}
          {!isSimulating && !error && result && result.results.length > 0 && (
            <div className="results-list">
              <div className="badge-row fade-in">
                <span className="badge">Faces: {result.n_faces}</span>
                <span className="badge">Identified: {result.n_identified}</span>
                <span className="badge">Unknown: {result.n_faces - result.n_identified}</span>
              </div>
              {(() => {
                const known   = result.results.filter(p => p.name !== 'UNKNOWN' && p.name.toLowerCase() !== 'unknown')
                const unknown = result.results.filter(p => p.name === 'UNKNOWN' || p.name.toLowerCase() === 'unknown')
                return (
                  <>
                    {known.length > 0 && (
                      <div className="result-section known-section fade-in">
                        <h4 className="section-title">Known Individuals</h4>
                        <div className="section-divider known-divider"></div>
                        {known.map((face, idx) => <FaceCard key={`known-${face.face_idx}`} face={face} delay={idx * 0.1} onHover={setHoveredFaceId} isHighlighted={hoveredFaceId === face.face_idx} />)}
                      </div>
                    )}
                    {unknown.length > 0 && (
                      <div className="result-section unknown-section fade-in" style={{ marginTop: known.length > 0 ? '1.5rem' : '0' }}>
                        <h4 className="section-title">Unknown Individuals</h4>
                        <div className="section-divider unknown-divider"></div>
                        {unknown.map((face, idx) => <FaceCard key={`unknown-${face.face_idx}`} face={face} delay={idx * 0.1} onHover={setHoveredFaceId} isHighlighted={hoveredFaceId === face.face_idx} />)}
                      </div>
                    )}
                  </>
                )
              })()}
            </div>
          )}
          {!isSimulating && !error && !result && <p className="muted">Prediction output appears here after processing.</p>}
        </article>
      </div>
      <Legend />
    </section>
  )
}
