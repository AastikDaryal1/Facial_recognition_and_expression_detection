import { useEffect, useMemo, useRef, useState, createContext, useContext } from 'react'
import {
  BrowserRouter,
  Link,
  NavLink,
  Route,
  Routes,
  useNavigate,
} from 'react-router-dom'
import {
  Camera,
  CameraOff,
  CloudUpload,
  ImagePlus,
  ScanFace,
  Sparkles,
  UserRoundSearch,
  User,
  HelpCircle,
  Key,
} from 'lucide-react'

const ApiKeyContext = createContext()

export function ApiKeyProvider({ children }) {
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('FACE_API_KEY') || import.meta.env.VITE_API_KEY || '')
  const [showSettings, setShowSettings] = useState(false)

  useEffect(() => {
    if (apiKey) {
      localStorage.setItem('FACE_API_KEY', apiKey)
    } else {
      localStorage.removeItem('FACE_API_KEY')
    }
  }, [apiKey])

  const logout = () => {
    setApiKey('')
  }

  return (
    <ApiKeyContext.Provider value={{ apiKey, setApiKey, showSettings, setShowSettings, logout }}>
      {children}
    </ApiKeyContext.Provider>
  )
}

export function useApiKey() {
  return useContext(ApiKeyContext)
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const demoEmotions = ['Happy', 'Neutral', 'Sad', 'Angry', 'Surprised']
const demoNames = ['Alex', 'Sam', 'Priya', 'Unknown', 'Jordan']

const navItems = [
  { to: '/', label: 'Home' },
  { to: '/upload', label: 'Upload' },
  { to: '/live', label: 'Live Detection' },
]

const allowedImageExtensions = [
  '.jpg',
  '.jpeg',
  '.png',
  '.webp',
  '.bmp',
  '.gif',
  '.heic',
  '.heif',
]

function isSupportedImageFile(file) {
  if (!file) return false
  const mimeType = file.type?.toLowerCase() || ''
  if (mimeType.startsWith('image/')) return true
  const lowerName = file.name?.toLowerCase() || ''
  return allowedImageExtensions.some((ext) => lowerName.endsWith(ext))
}

const emotionColors = {
  Angry: '#FF4C4C',
  Fear: '#8E44AD',
  Happy: '#FFD93D',
  Neutral: '#BDC3C7',
  Sad: '#3498DB',
  Surprise: '#FF9F43',
}

function getEmotionColor(emotion) {
  if (!emotion) return '#BDC3C7'
  const match = Object.keys(emotionColors).find(
    (e) => e.toLowerCase() === emotion.toLowerCase()
  )
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

function FaceAvatar({ name, color }) {
  const [imgError, setImgError] = useState(false)
  const isUnknown = !name || name.toLowerCase() === 'unknown'
  const imageUrl = `/dataset/${name}.jpg`

  return (
    <div className="avatar-box" style={{ borderColor: color, boxShadow: `inset 0 0 15px ${color}33, 0 0 8px ${color}44` }}>
      {!isUnknown && !imgError ? (
        <img 
          src={imageUrl} 
          alt={name} 
          onError={() => setImgError(true)} 
          className="avatar-image"
        />
      ) : isUnknown ? (
        <HelpCircle size={28} color={color} className="avatar-icon" />
      ) : (
        <User size={28} color={color} className="avatar-icon" />
      )}
    </div>
  )
}

function FaceCard({ face, delay }) {
  const isUnknown = !face.name || face.name.toLowerCase() === 'unknown'
  const color = getEmotionColor(face.emotion)

  return (
    <div 
      className="face-card fade-in-up" 
      style={{ 
        borderColor: color, 
        borderStyle: isUnknown ? 'dashed' : 'solid',
        boxShadow: `0 0 8px ${color}33`,
        animationDelay: delay ? `${delay}s` : '0s' 
      }}
    >
      <div className="face-card-left">
        <FaceAvatar name={face.name} color={color} />
      </div>
      <div className="face-card-middle">
        <p style={{ color: isUnknown ? '#ffffff' : 'inherit' }}>
          <strong>Name:</strong> {isUnknown ? 'UNKNOWN' : face.name}
        </p>
        <p>
          <strong>Emotion:</strong> <span style={{ color }}>{face.emotion}</span>
        </p>
      </div>
      <div className="face-card-right">
        {isUnknown ? (
          <span className="identity-badge unknown">
            <HelpCircle size={14} /> Unknown
          </span>
        ) : (
          <span className="identity-badge known">
            <User size={14} /> Known
          </span>
        )}
      </div>
    </div>
  )
}

function AppShell({ children }) {
  const { apiKey, setApiKey, showSettings, setShowSettings, logout } = useApiKey()
  const [tempKey, setTempKey] = useState(apiKey)

  useEffect(() => {
    if (showSettings) setTempKey(apiKey)
  }, [showSettings, apiKey])

  const saveSettings = () => {
    setApiKey(tempKey)
    setShowSettings(false)
  }

  return (
    <div className="app-shell">
      <header className="topbar glass-card">
        <Link className="brand" to="/">
          <Sparkles size={18} />
          <span>AI Vision Console</span>
        </Link>
        <nav className="nav">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            >
              {item.label}
            </NavLink>
          ))}
          {apiKey ? (
            <button className="secondary-btn" onClick={logout} style={{ marginLeft: '0.5rem', padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}>
              Log Out
            </button>
          ) : (
            <button className="primary-btn" onClick={() => setShowSettings(true)} style={{ marginTop: 0, marginLeft: '0.5rem', padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}>
              Log In
            </button>
          )}
        </nav>
      </header>

      {showSettings && (
        <div className="modal-overlay">
          <div className="modal-content glass-card fade-in">
            <h3 style={{marginTop: 0}}>API Settings</h3>
            <p className="muted" style={{marginBottom: '1rem'}}>Enter your authentication key to connect to the backend.</p>
            <input 
              type="password" 
              className="text-input"
              value={tempKey} 
              onChange={(e) => setTempKey(e.target.value)} 
              placeholder="API Key..."
            />
            <div style={{display: 'flex', gap: '1rem', marginTop: '1.5rem', justifyContent: 'flex-end'}}>
              <button className="secondary-btn" onClick={() => setShowSettings(false)}>Cancel</button>
              <button className="primary-btn" style={{marginTop: 0}} onClick={saveSettings}>Save</button>
            </div>
          </div>
        </div>
      )}

      <main className="content-wrap">{children}</main>
    </div>
  )
}

function ActionCard({ icon: Icon, title, description, buttonText, to }) {
  return (
    <article className="action-card glass-card glow-border">
      <div className="action-icon">
        <Icon size={28} />
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
      <Link className="primary-btn" to={to}>
        {buttonText}
      </Link>
    </article>
  )
}

function LandingPage() {
  return (
    <section className="fade-in">
      <div className="hero glass-card">
        <p className="kicker">Computer Vision + Emotion AI</p>
        <h1>Face & Emotion Detection System</h1>
        <p className="hero-subtitle">
          Detect faces, identify known people, and estimate real-time emotions
          from uploaded images or a live camera feed.
        </p>
      </div>
      <div className="action-grid">
        <ActionCard
          icon={ImagePlus}
          title="Upload Photo"
          description="Drop an image and receive face, identity, and emotion predictions in seconds."
          buttonText="Open Upload"
          to="/upload"
        />
        <ActionCard
          icon={ScanFace}
          title="Live Detection"
          description="Use webcam-based real-time analysis with dynamic overlays and live updates."
          buttonText="Open Live View"
          to="/live"
        />
      </div>
    </section>
  )
}

function UploadPage() {
  const { apiKey, setShowSettings } = useApiKey()
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [dragActive, setDragActive] = useState(false)

  const MAX_SIZE_MB = import.meta.env.VITE_MAX_UPLOAD_SIZE_MB || 5

  useEffect(() => {
    if (!file) return undefined
    const nextPreview = URL.createObjectURL(file)
    setPreviewUrl(nextPreview)
    return () => URL.revokeObjectURL(nextPreview)
  }, [file])

  const handleSelection = (selectedFile) => {
    setResult(null)
    setError('')
    if (!selectedFile) return
    if (!isSupportedImageFile(selectedFile)) {
      setError('Please upload a valid image file.')
      return
    }
    if (selectedFile.size > MAX_SIZE_MB * 1024 * 1024) {
      window.alert(`Error: File is too large. Maximum size allowed is ${MAX_SIZE_MB}MB.`)
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
    if (!apiKey) {
      setShowSettings(true)
      return
    }

    if (!file) {
      setError('Upload an image before running detection.')
      return
    }
    setLoading(true)
    setError('')
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_BASE}/predict/image`, {
        method: 'POST',
        headers: {
          'X-API-Key': apiKey,
        },
        body: formData,
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed.')
      }
      setResult(data)
    } catch (fetchError) {
      setError(fetchError.message || 'Could not reach backend API.')
    } finally {
      setLoading(false)
    }
  }

  const [showCamera, setShowCamera] = useState(false)
  const videoRef = useRef(null)

  useEffect(() => {
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop())
      }
    }
  }, [])

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      setShowCamera(true)
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.play()
        }
      }, 100)
    } catch (err) {
      setError('Could not access camera. Please check permissions.')
    }
  }

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop())
    }
    setShowCamera(false)
  }

  const capturePhoto = () => {
    if (!videoRef.current) return
    const canvas = document.createElement('canvas')
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
    
    canvas.toBlob((blob) => {
      if (blob) {
        const capturedFile = new File([blob], 'capture.jpg', { type: 'image/jpeg' })
        handleSelection(capturedFile)
        stopCamera()
      }
    }, 'image/jpeg', 0.9)
  }

  const [imgDims, setImgDims] = useState({ w: 1, h: 1 })

  return (
    <section className="fade-in">
      <div className="page-heading">
        <h2>Image Upload Detection</h2>
        <p>Drop a photo, choose from device, or snap a picture.</p>
      </div>

      <div
        className={`upload-zone glass-card ${dragActive ? 'drag-active' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setDragActive(true)
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        <CloudUpload size={34} />
        <p>Drop image here or select from file picker</p>
        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
          <label className="secondary-btn">
            Choose Image
            <input
              type="file"
              accept="image/*"
              hidden
              onChange={(event) => handleSelection(event.target.files?.[0])}
            />
          </label>
          <button className="secondary-btn" onClick={startCamera}>
            <Camera size={18} style={{ marginRight: '8px' }} />
            Open Camera
          </button>
        </div>
      </div>

      {showCamera && (
        <div className="glass-card fade-in-up" style={{ marginTop: '1.5rem', textAlign: 'center' }}>
          <h3 style={{ marginBottom: '1rem' }}>Camera Capture</h3>
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted 
            style={{ width: '100%', maxWidth: '640px', borderRadius: '8px', border: '1px solid #334155', marginBottom: '1rem' }}
          />
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
            <button className="primary-btn" onClick={capturePhoto}>
              <Camera size={18} style={{ marginRight: '8px' }} />
              Snap Photo
            </button>
            <button className="secondary-btn" onClick={stopCamera}>
              Cancel
            </button>
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
                className="preview-image" 
                onLoad={(e) => setImgDims({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
              />
              {result && result.results.map((face, idx) => {
                const isUnknown = !face.name || face.name.toLowerCase() === 'unknown'
                const color = getEmotionColor(face.emotion)
                
                // The backend resizes images to a max width of 640px before inference.
                // We must calculate the effective dimensions to scale the bounding boxes correctly.
                const effectiveW = Math.min(imgDims.w, 640)
                const scale = effectiveW / imgDims.w
                const effectiveH = imgDims.h * scale

                return (
                  <div
                    key={idx}
                    className="face-box fade-in"
                    style={{
                      left: `${(face.x / effectiveW) * 100}%`,
                      top: `${(face.y / effectiveH) * 100}%`,
                      width: `${(face.w / effectiveW) * 100}%`,
                      height: `${(face.h / effectiveH) * 100}%`,
                      borderColor: color,
                      borderStyle: isUnknown ? 'dashed' : 'solid',
                      boxShadow: `0 0 10px ${color}`
                    }}
                  >
                    <span style={{ 
                      backgroundColor: isUnknown ? '#000000' : 'rgba(15, 23, 42, 0.85)',
                      color: color,
                      border: `1px solid ${color}`
                    }}>
                      {isUnknown ? (
                        <><HelpCircle size={12} className="inline-icon" /> UNKNOWN - {face.emotion}</>
                      ) : (
                        <><User size={12} className="inline-icon" /> {face.name} - {face.emotion}</>
                      )}
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
          {loading && <p className="pulse">Running AI inference...</p>}
          {error && <p className="error">{error}</p>}
          {!loading && !error && result && result.results.length === 0 && (
            <p className="warning">No face detected in the selected image.</p>
          )}
          {!loading && !error && result && result.results.length > 0 && (
            <div className="results-list">
              <div className="badge-row">
                <span className="badge">Faces: {result.n_faces}</span>
                <span className="badge">Identified: {result.n_identified}</span>
                <span className="badge">Unknown: {result.n_faces - result.n_identified}</span>
              </div>
              {result.results.map((face, idx) => (
                <FaceCard key={face.face_idx} face={face} delay={idx * 0.1} />
              ))}
            </div>
          )}
          {!loading && !error && !result && (
            <p className="muted">Prediction output appears here after processing.</p>
          )}
        </article>
      </div>
      <Legend />
    </section>
  )
}

function LivePage() {
  const { apiKey, setShowSettings } = useApiKey()
  const videoRef = useRef(null)
  const [streaming, setStreaming] = useState(false)
  const [permissionError, setPermissionError] = useState('')
  const [detectedFaces, setDetectedFaces] = useState([])
  const intervalRef = useRef(null)

  const startCamera = async () => {
    if (!apiKey) {
      setShowSettings(true)
      return
    }

    setPermissionError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      setStreaming(true)
    } catch (err) {
      setPermissionError('Camera permission denied or unavailable.')
    }
  }

  const stopCamera = () => {
    const stream = videoRef.current?.srcObject
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setStreaming(false)
    setDetectedFaces([])
  }

  useEffect(() => () => stopCamera(), [])

  useEffect(() => {
    let isActive = true
    let timeoutId = null

    if (!streaming) {
      return
    }

    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')

    const processFrame = async () => {
      if (!isActive) return
      
      if (!videoRef.current || videoRef.current.videoWidth === 0) {
        timeoutId = setTimeout(processFrame, 500)
        return
      }

      canvas.width = videoRef.current.videoWidth
      canvas.height = videoRef.current.videoHeight
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
      
      const base64Image = canvas.toDataURL('image/jpeg', 0.8)
      const image_b64 = base64Image.replace(/^data:image\/[a-z]+;base64,/, '')

      try {
        const response = await fetch(`${API_BASE}/predict/base64`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': apiKey,
          },
          body: JSON.stringify({ image_b64, filename: 'live.jpg' }),
        })

        if (response.ok && isActive) {
          const data = await response.json()
          
          const items = data.results.map((face) => ({
            id: face.face_idx,
            name: face.name,
            emotion: face.emotion,
            x: (face.x / canvas.width) * 100,
            y: (face.y / canvas.height) * 100,
            w: (face.w / canvas.width) * 100,
            h: (face.h / canvas.height) * 100,
          }))
          
          setDetectedFaces(items)
        }
      } catch (err) {
        console.error('Live detection error:', err)
      }

      // Schedule the next frame only after this one completes
      if (isActive) {
        timeoutId = setTimeout(processFrame, 200)
      }
    }

    // Start the loop
    processFrame()

    return () => {
      isActive = false
      if (timeoutId) clearTimeout(timeoutId)
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
            <video ref={videoRef} autoPlay muted playsInline />
            {!streaming && (
              <div className="camera-placeholder">
                <UserRoundSearch size={48} />
                <p>Camera feed will appear here</p>
              </div>
            )}
            {detectedFaces.map((face) => {
              const isUnknown = !face.name || face.name.toLowerCase() === 'unknown'
              const color = getEmotionColor(face.emotion)
              return (
                <div
                  key={face.id}
                  className="face-box fade-in"
                  style={{
                    left: `${face.x}%`,
                    top: `${face.y}%`,
                    width: `${face.w}%`,
                    height: `${face.h}%`,
                    borderColor: color,
                    borderStyle: isUnknown ? 'dashed' : 'solid',
                    boxShadow: `0 0 10px ${color}`
                  }}
                >
                  <span style={{ 
                    backgroundColor: isUnknown ? '#000000' : 'rgba(15, 23, 42, 0.85)',
                    color: color,
                    border: `1px solid ${color}`
                  }}>
                    {isUnknown ? (
                      <><HelpCircle size={12} className="inline-icon" /> UNKNOWN - {face.emotion}</>
                    ) : (
                      <><User size={12} className="inline-icon" /> {face.name} - {face.emotion}</>
                    )}
                  </span>
                </div>
              )
            })}
          </div>

          <div className="controls">
            {!streaming ? (
              <button className="primary-btn" onClick={startCamera}>
                <Camera size={16} /> Start Camera
              </button>
            ) : (
              <button className="secondary-btn" onClick={stopCamera}>
                <CameraOff size={16} /> Stop Camera
              </button>
            )}
            <button className="ghost-btn" disabled={!streaming}>
              <Camera size={16} /> Snapshot
            </button>
          </div>
          {permissionError && <p className="error">{permissionError}</p>}
        </article>

        <article className="glass-card stats-card">
          <h3>Live Analytics</h3>
          <p className="stat">
            Faces Detected: <strong>{detectedFaces.length}</strong>
          </p>
          <div className="live-feed-list">
            {detectedFaces.length === 0 && (
              <p className="muted">No active faces detected yet.</p>
            )}
            {detectedFaces.map((face, idx) => (
              <FaceCard key={`feed-${face.id}`} face={face} delay={idx * 0.1} />
            ))}
          </div>
        </article>
      </div>
      <Legend />
    </section>
  )
}

function NotFound() {
  const navigate = useNavigate()
  return (
    <section className="glass-card not-found">
      <h2>Page not found</h2>
      <p>The requested route does not exist.</p>
      <button className="primary-btn" onClick={() => navigate('/')}>
        Return Home
      </button>
    </section>
  )
}

function App() {
  return (
    <ApiKeyProvider>
      <BrowserRouter>
        <AppShell>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/live" element={<LivePage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </AppShell>
      </BrowserRouter>
    </ApiKeyProvider>
  )
}

export default App
