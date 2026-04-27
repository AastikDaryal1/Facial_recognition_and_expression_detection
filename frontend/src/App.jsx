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
  CircleDot,
  Activity,
  ShieldCheck,
  Zap,
  Users,
  Cpu,
} from 'lucide-react'

const ApiKeyContext = createContext()

export function ApiKeyProvider({ children }) {
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('FACE_API_KEY') || import.meta.env.VITE_API_KEY || '')
  const [showSettings, setShowSettings] = useState(false)

  const TIMEOUT_MS = 10 * 60 * 1000;

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

  // 10-minute inactivity auto-logout
  useEffect(() => {
    if (!apiKey) return;

    let timeout;
    const resetTimer = () => {
      clearTimeout(timeout);
      timeout = setTimeout(() => {
        logout();
      }, TIMEOUT_MS);
    };

    resetTimer();

    window.addEventListener('mousemove', resetTimer);
    window.addEventListener('keydown', resetTimer);
    window.addEventListener('click', resetTimer);
    window.addEventListener('scroll', resetTimer);

    return () => {
      clearTimeout(timeout);
      window.removeEventListener('mousemove', resetTimer);
      window.removeEventListener('keydown', resetTimer);
      window.removeEventListener('click', resetTimer);
      window.removeEventListener('scroll', resetTimer);
    };
  }, [apiKey]);

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

function FaceAvatar({ face, color }) {
  const isUnknown = !face.name || face.name.toUpperCase() === 'UNKNOWN'

  const getImage = (person) => {
    return person.face_image || "/dataset/unknown.png";
  };

  return (
    <div className={`avatar-square ${isUnknown ? 'unknown-avatar' : 'known-avatar'}`} style={{
      borderColor: color,
      boxShadow: `0 0 10px ${color}80`,
      borderStyle: isUnknown ? 'dashed' : 'solid'
    }}>
      <img
        src={getImage(face)}
        alt={face.name || 'Unknown'}
        onError={(e) => {
          e.target.src = "/dataset/unknown.png";
        }}
        className="avatar-image"
      />
    </div>
  )
}

function FaceCard({ face, delay, onHover, isHighlighted }) {
  const isUnknown = !face.name || face.name.toLowerCase() === 'unknown'
  const color = getEmotionColor(face.emotion)

  return (
    <div
      className={`face-card fade-in-up ${isUnknown ? 'face-card-unknown' : 'face-card-known'} ${isHighlighted ? 'highlighted' : ''}`}
      onMouseEnter={() => onHover && onHover(face.face_idx !== undefined ? face.face_idx : face.id)}
      onMouseLeave={() => onHover && onHover(null)}
      style={{
        borderColor: isHighlighted ? '#fff' : color,
        borderStyle: isUnknown ? 'dashed' : 'solid',
        boxShadow: isHighlighted ? `0 0 20px ${color}` : `0 0 8px ${color}33`,
        animationDelay: delay ? `${delay}s` : '0s',
        transform: isHighlighted ? 'translateY(-5px) scale(1.02)' : 'none',
        filter: isHighlighted ? 'brightness(1.2)' : 'none',
        zIndex: isHighlighted ? 10 : 1
      }}
    >
      <div className="face-card-left">
        <FaceAvatar face={face} color={color} />
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

function SkeletonCard() {
  return (
    <div className="skeleton-card fade-in-up">
      <div className="skeleton-avatar"></div>
      <div className="skeleton-text-block">
        <div className="skeleton-text short"></div>
        <div className="skeleton-text long"></div>
      </div>
      <div className="skeleton-text short" style={{ width: '60px' }}></div>
    </div>
  )
}

function AppShell({ children }) {
  const { apiKey, setApiKey, showSettings, setShowSettings, logout } = useApiKey()
  const [tempKey, setTempKey] = useState(apiKey)
  const [isValidating, setIsValidating] = useState(false)
  const [errorMsg, setErrorMsg] = useState('')

  useEffect(() => {
    if (showSettings) {
      setTempKey(apiKey)
      setErrorMsg('')
    }
  }, [showSettings, apiKey])

  const saveSettings = async () => {
    if (!tempKey) {
      setErrorMsg('API Key cannot be empty')
      return
    }
    setIsValidating(true)
    setErrorMsg('')
    try {
      const res = await fetch(`${API_BASE}/model/info`, {
        headers: {
          'X-API-Key': tempKey
        }
      })
      if (!res.ok) {
        setErrorMsg('Invalid API Key')
        setIsValidating(false)
        return
      }
      setApiKey(tempKey)
      setShowSettings(false)
    } catch (err) {
      setErrorMsg('Failed to connect to server')
    }
    setIsValidating(false)
  }

  return (
    <div className="app-shell">
      <header className="topbar glass-card">
        <Link className="brand" to="/">
          <Sparkles size={18} />
          <span>VisionX</span>
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
            <h3 style={{ marginTop: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
              🔒 Authentication Required
            </h3>
            <p className="muted" style={{ marginBottom: '1rem' }}>Enter your password to continue and fetch results.</p>
            <input
              type="password"
              className="text-input"
              value={tempKey}
              onChange={(e) => setTempKey(e.target.value)}
              placeholder="Enter Password..."
            />
            {errorMsg && <p style={{color: '#FF4C4C', fontSize: '0.9rem', marginTop: '0.5rem'}}>{errorMsg}</p>}
            <div style={{display: 'flex', gap: '1rem', marginTop: '1.5rem', justifyContent: 'flex-end'}}>
              <button className="secondary-btn" onClick={() => setShowSettings(false)}>Cancel</button>
              <button className="primary-btn" style={{marginTop: 0}} onClick={saveSettings} disabled={isValidating}>
                {isValidating ? 'Verifying...' : 'Continue'}
              </button>
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
  const { apiKey } = useApiKey()
  const [stats, setStats] = useState({ uptime: '0s', latency: '0s', count: 0 })
  const [members, setMembers] = useState([])

  useEffect(() => {
    if (!apiKey) return

    // Fetch system metrics
    fetch(`${import.meta.env.VITE_API_URL}/metrics`, {
      headers: { 'X-API-Key': apiKey }
    })
      .then(res => res.json())
      .then(data => {
        setStats({
          uptime: `${data.uptime_s || 0}s`,
          latency: `${data.avg_latency_s || 0}s`,
          count: data.request_count || 0
        })
      })
      .catch(() => {})

    // Fetch model info (members)
    fetch(`${import.meta.env.VITE_API_URL}/model/info`, {
      headers: { 'X-API-Key': apiKey }
    })
      .then(res => res.json())
      .then(data => {
        if (data.members) setMembers(data.members)
      })
      .catch(() => {})
  }, [apiKey])

  return (
    <section className="fade-in">
      <div className="hero glass-card">
        <p className="kicker">Computer Vision + Emotion AI</p>
        <h1>Face & Emotion Detection System</h1>
        <p className="hero-subtitle">
          Detect faces, identify known people, and estimate real-time emotions
          from uploaded images or a live camera feed.
        </p>

        <div className="stats-grid">
          <div className="stat-card glass-card">
            <div className="stat-icon"><Activity size={18} /></div>
            <div className="stat-info">
              <span className="stat-label">Model Status</span>
              <span className="stat-value text-green">Ready & Active</span>
            </div>
          </div>
          <div className="stat-card glass-card">
            <div className="stat-icon"><Zap size={18} /></div>
            <div className="stat-info">
              <span className="stat-label">Avg. Latency</span>
              <span className="stat-value">{stats.latency}</span>
            </div>
          </div>
          <div className="stat-card glass-card">
            <div className="stat-icon"><Cpu size={18} /></div>
            <div className="stat-info">
              <span className="stat-label">Uptime</span>
              <span className="stat-value">{stats.uptime}</span>
            </div>
          </div>
          <div className="stat-card glass-card">
            <div className="stat-icon"><Users size={18} /></div>
            <div className="stat-info">
              <span className="stat-label">Requests</span>
              <span className="stat-value">{stats.count}</span>
            </div>
          </div>
        </div>
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

      <div className="home-section">
        <div className="section-header">
          <ShieldCheck size={20} className="text-blue" />
          <h2>Core Capabilities</h2>
        </div>
        <div className="features-grid">
          <div className="feature-item">
            <div className="feature-dot blue"></div>
            <div className="feature-content">
              <h4>FaceNet Identification</h4>
              <p>Deep learning embeddings for 99%+ accuracy on known individuals.</p>
            </div>
          </div>
          <div className="feature-item">
            <div className="feature-dot purple"></div>
            <div className="feature-content">
              <h4>Emotion Calibration</h4>
              <p>Multi-stage detection for accurate joy, sadness, and neutral state estimation.</p>
            </div>
          </div>
          <div className="feature-item">
            <div className="feature-dot green"></div>
            <div className="feature-content">
              <h4>Session Security</h4>
              <p>Automatic logout and key validation protecting your analytics.</p>
            </div>
          </div>
        </div>
      </div>

      {members.length > 0 && (
        <div className="home-section">
          <div className="section-header">
            <Users size={20} />
            <h2>Recognition Library</h2>
          </div>
          <div className="members-library">
            {members.map(name => (
              <span key={name} className="member-chip">{name}</span>
            ))}
          </div>
        </div>
      )}
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
  const [hoveredFaceId, setHoveredFaceId] = useState(null)
  const [simulatedProgress, setSimulatedProgress] = useState(0)
  const [isSimulating, setIsSimulating] = useState(false)
  const [statusText, setStatusText] = useState('')

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
    setIsSimulating(true)
    setSimulatedProgress(0)
    setStatusText('Scanning faces...')
    setError('')
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      // Start backend fetch
      const fetchPromise = fetch(`${API_BASE}/predict/image`, {
        method: 'POST',
        headers: {
          'X-API-Key': apiKey,
        },
        body: formData,
      }).then(res => {
        if (!res.ok) {
          return res.json().then(data => { throw new Error(data.detail || 'Prediction failed.') })
        }
        return res.json()
      })

      // Simulate timeline
      let currentProgress = 0
      const interval = setInterval(() => {
        currentProgress += 2 // increment progress

        if (currentProgress >= 90) {
          clearInterval(interval)
        } else {
          setSimulatedProgress(currentProgress)
          if (currentProgress < 30) setStatusText('Scanning faces...')
          else if (currentProgress < 60) setStatusText('Analyzing emotions...')
          else setStatusText('Matching identities...')
        }
      }, 50)

      const data = await fetchPromise

      // Complete simulation once fetch returns
      clearInterval(interval)
      setSimulatedProgress(100)
      setStatusText('Finalizing results...')

      setTimeout(() => {
        setResult(data)
        setIsSimulating(false)
        setLoading(false)
      }, 500) // slight delay to show final 100% state

    } catch (fetchError) {
      setError(fetchError.message || 'Could not reach backend API.')
      setIsSimulating(false)
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
                className={`preview-image ${isSimulating && simulatedProgress < 90 ? 'processing' : ''}`} 
                onLoad={(e) => setImgDims({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
              />
              {isSimulating && simulatedProgress < 100 && (
                <div className="scanning-beam"></div>
              )}
              {result && simulatedProgress >= 40 && result.results.map((face, idx) => {
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
                    className={`face-box scale-in ${hoveredFaceId === face.face_idx ? 'highlighted' : ''}`}
                    onMouseEnter={() => setHoveredFaceId(face.face_idx)}
                    onMouseLeave={() => setHoveredFaceId(null)}
                    style={{
                      left: `${(face.x / effectiveW) * 100}%`,
                      top: `${(face.y / effectiveH) * 100}%`,
                      width: `${(face.w / effectiveW) * 100}%`,
                      height: `${(face.h / effectiveH) * 100}%`,
                      borderColor: (isSimulating && simulatedProgress < 70) ? '#64748b' : (hoveredFaceId === face.face_idx ? '#fff' : color),
                      borderStyle: isUnknown && (!isSimulating || simulatedProgress >= 70) ? 'dashed' : 'solid',
                      boxShadow: (isSimulating && simulatedProgress < 70) ? 'none' : (hoveredFaceId === face.face_idx ? `0 0 25px 2px ${color}` : `0 0 10px ${color}`),
                      zIndex: hoveredFaceId === face.face_idx ? 100 : 1
                    }}
                  >
                    <span className={isSimulating && simulatedProgress < 70 ? 'label-analyzing' : ''} style={{
                      backgroundColor: isUnknown && (!isSimulating || simulatedProgress >= 70) ? '#000000' : 'rgba(15, 23, 42, 0.85)',
                      color: (isSimulating && simulatedProgress < 70) ? '#e2e8f0' : color,
                      border: `1px solid ${(isSimulating && simulatedProgress < 70) ? '#64748b' : color}`
                    }}>
                      {(isSimulating && simulatedProgress < 70) ? (
                        <>Analyzing...</>
                      ) : isUnknown ? (
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

          {isSimulating && (
            <div className="processing-status-container">
              <div className="status-text">{statusText}</div>
              <div className="progress-bar-bg">
                <div className="progress-bar-fill" style={{ width: `${simulatedProgress}%` }}></div>
              </div>
            </div>
          )}

          {error && <p className="error">{error}</p>}

          {isSimulating && simulatedProgress >= 30 && !result && (
            <div className="results-list">
              <SkeletonCard />
              <SkeletonCard />
            </div>
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
                const known = result.results.filter(p => p.name !== "UNKNOWN" && p.name.toLowerCase() !== "unknown");
                const unknown = result.results.filter(p => p.name === "UNKNOWN" || p.name.toLowerCase() === "unknown");
                return (
                  <>
                    {known.length > 0 && (
                      <div className="result-section known-section fade-in">
                        <h4 className="section-title">Known Individuals</h4>
                        <div className="section-divider known-divider"></div>
                        {known.map((face, idx) => (
                          <FaceCard 
                            key={`known-${face.face_idx}`} 
                            face={face} 
                            delay={idx * 0.1} 
                            onHover={setHoveredFaceId}
                            isHighlighted={hoveredFaceId === face.face_idx}
                          />
                        ))}
                      </div>
                    )}
                    {unknown.length > 0 && (
                      <div className="result-section unknown-section fade-in" style={{ marginTop: known.length > 0 ? '1.5rem' : '0' }}>
                        <h4 className="section-title">Unknown Individuals</h4>
                        <div className="section-divider unknown-divider"></div>
                        {unknown.map((face, idx) => (
                          <FaceCard 
                            key={`unknown-${face.face_idx}`} 
                            face={face} 
                            delay={idx * 0.1} 
                            onHover={setHoveredFaceId}
                            isHighlighted={hoveredFaceId === face.face_idx}
                          />
                        ))}
                      </div>
                    )}
                  </>
                );
              })()}
            </div>
          )}

          {!isSimulating && !error && !result && (
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

  const [snapshotImage, setSnapshotImage] = useState(null)
  const [isSimulating, setIsSimulating] = useState(false)
  const [simulatedProgress, setSimulatedProgress] = useState(0)
  const [statusText, setStatusText] = useState('')
  const [snapshotResult, setSnapshotResult] = useState(null)
  const [isSnapshotMode, setIsSnapshotMode] = useState(false)
  const [snapshotDims, setSnapshotDims] = useState({ w: 1, h: 1 })
  const [hoveredFaceId, setHoveredFaceId] = useState(null)

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

    canvas.toBlob(async (blob) => {
      if (!blob) return
      const file = new File([blob], 'snapshot.jpg', { type: 'image/jpeg' })
      const formData = new FormData()
      formData.append('file', file)

      try {
        const fetchPromise = fetch(`${API_BASE}/predict/image`, {
          method: 'POST',
          headers: { 'X-API-Key': apiKey },
          body: formData,
        }).then(res => {
          if (!res.ok) {
            return res.json().then(data => { throw new Error(data.detail || 'Prediction failed.') })
          }
          return res.json()
        })

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

        const data = await fetchPromise

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
            face_image: face.face_image,
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
            {isSnapshotMode && snapshotImage ? (
              <div className="preview-image-container" style={{ width: '100%', height: '100%', margin: 0 }}>
                <img
                  src={snapshotImage}
                  alt="Snapshot"
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  className={`preview-image ${isSimulating && simulatedProgress < 90 ? 'processing' : ''}`}
                />
                {isSimulating && simulatedProgress < 100 && (
                  <div className="scanning-beam"></div>
                )}
                {snapshotResult && simulatedProgress >= 40 && snapshotResult.results.map((face, idx) => {
                  const isUnknown = !face.name || face.name.toLowerCase() === 'unknown'
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
                        borderColor: (isSimulating && simulatedProgress < 70) ? '#64748b' : (hoveredFaceId === face.face_idx ? '#fff' : color),
                        borderStyle: isUnknown && (!isSimulating || simulatedProgress >= 70) ? 'dashed' : 'solid',
                        boxShadow: (isSimulating && simulatedProgress < 70) ? 'none' : (hoveredFaceId === face.face_idx ? `0 0 25px 2px ${color}` : `0 0 10px ${color}`),
                        zIndex: hoveredFaceId === face.face_idx ? 100 : 1
                      }}
                    >
                      <span className={isSimulating && simulatedProgress < 70 ? 'label-analyzing' : ''} style={{
                        backgroundColor: isUnknown && (!isSimulating || simulatedProgress >= 70) ? '#000000' : 'rgba(15, 23, 42, 0.85)',
                        color: (isSimulating && simulatedProgress < 70) ? '#e2e8f0' : color,
                        border: `1px solid ${(isSimulating && simulatedProgress < 70) ? '#64748b' : color}`
                      }}>
                        {(isSimulating && simulatedProgress < 70) ? (
                          <>Analyzing...</>
                        ) : isUnknown ? (
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
              <>
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
                      className={`face-box fade-in ${hoveredFaceId === face.id ? 'highlighted' : ''}`}
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
                        zIndex: hoveredFaceId === face.id ? 100 : 1
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
              </>
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

        <article className="glass-card stats-card">
          {isSnapshotMode ? (
            <>
              <h3>Snapshot Results</h3>
              {isSimulating && (
                <div className="processing-status-container">
                  <div className="status-text">{statusText}</div>
                  <div className="progress-bar-bg">
                    <div className="progress-bar-fill" style={{ width: `${simulatedProgress}%` }}></div>
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
                    const known = snapshotResult.results.filter(p => p.name !== "UNKNOWN" && p.name.toLowerCase() !== "unknown");
                    const unknown = snapshotResult.results.filter(p => p.name === "UNKNOWN" || p.name.toLowerCase() === "unknown");
                    return (
                      <>
                        {known.length > 0 && (
                          <div className="result-section known-section fade-in">
                            <h4 className="section-title">Known Individuals</h4>
                            <div className="section-divider known-divider"></div>
                            {known.map((face, idx) => (
                              <FaceCard 
                                key={`snap-known-${face.face_idx}`} 
                                face={face} 
                                delay={idx * 0.1} 
                                onHover={setHoveredFaceId}
                                isHighlighted={hoveredFaceId === face.face_idx}
                              />
                            ))}
                          </div>
                        )}
                        {unknown.length > 0 && (
                          <div className="result-section unknown-section fade-in" style={{ marginTop: known.length > 0 ? '1.5rem' : '0' }}>
                            <h4 className="section-title">Unknown Individuals</h4>
                            <div className="section-divider unknown-divider"></div>
                            {unknown.map((face, idx) => (
                              <FaceCard 
                                key={`snap-unknown-${face.face_idx}`} 
                                face={face} 
                                delay={idx * 0.1} 
                                onHover={setHoveredFaceId}
                                isHighlighted={hoveredFaceId === face.face_idx}
                              />
                            ))}
                          </div>
                        )}
                      </>
                    );
                  })()}
                </div>
              )}
            </>
          ) : (
            <>
              <h3>Live Analytics</h3>
              <p className="stat">
                Faces Detected: <strong>{detectedFaces.length}</strong>
              </p>
              <div className="live-feed-list">
                {detectedFaces.length === 0 && (
                  <p className="muted">No active faces detected yet.</p>
                )}
                {detectedFaces.map((face, idx) => (
                  <FaceCard 
                    key={`feed-${face.id}`} 
                    face={face} 
                    delay={idx * 0.1} 
                    onHover={setHoveredFaceId}
                    isHighlighted={hoveredFaceId === face.id}
                  />
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