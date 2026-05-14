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
  Menu,
  X,
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
    <div className="dual-legend glass-card fade-in-up" style={{ animationDelay: '0.3s' }}>
      <div className="legend-section">
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <h4 className="legend-title">Emotion (Color)</h4>
          <HelpCircle size={14} style={{ opacity: 0.5, cursor: 'help' }} title="Each emotion is mapped to a unique color border." />
        </div>
        <div className="legend-items">
          {Object.entries(emotionColors).map(([emotion, color]) => (
            <div key={emotion} className="legend-item" title={`${emotion} detection color`}>
              <div className="legend-color" style={{ backgroundColor: color, boxShadow: `0 0 10px ${color}` }}></div>
              <span className="legend-text">{emotion}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="legend-divider"></div>
      <div className="legend-section">
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <h4 className="legend-title">Identity (Style)</h4>
          <HelpCircle size={14} style={{ opacity: 0.5, cursor: 'help' }} title="Solid lines for known people, dashed for unknowns." />
        </div>
        <div className="legend-items">
          <div className="legend-item">
            <div className="legend-style-box solid"></div>
            <span className="legend-text">Known Member</span>
          </div>
          <div className="legend-item">
            <div className="legend-style-box dashed"></div>
            <span className="legend-text">Unknown Subject</span>
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
        borderColor: isHighlighted ? '#fff' : `${color}44`,
        borderStyle: isUnknown ? 'dashed' : 'solid',
        boxShadow: isHighlighted ? `0 0 25px ${color}` : 'none',
        animationDelay: delay ? `${delay}s` : '0s',
        zIndex: isHighlighted ? 10 : 1,
        background: isHighlighted ? `linear-gradient(135deg, rgba(30, 41, 59, 0.8), ${color}11)` : 'var(--card)'
      }}
    >
      <div className="face-card-left">
        <FaceAvatar face={face} color={color} />
      </div>
      <div className="face-card-middle">
        <p style={{ fontWeight: 600, color: isUnknown ? 'var(--muted)' : 'var(--text)' }}>
          {isUnknown ? 'IDENTIFYING...' : face.name}
        </p>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ 
            fontSize: '0.75rem', 
            padding: '2px 8px', 
            borderRadius: '4px', 
            backgroundColor: `${color}22`,
            color: color,
            border: `1px solid ${color}44`,
            textTransform: 'uppercase',
            fontWeight: 700
          }}>
            {face.emotion}
          </span>
        </div>
      </div>
      <div className="face-card-right">
        {isUnknown ? (
          <HelpCircle size={18} style={{ color: 'var(--muted)', opacity: 0.5 }} />
        ) : (
          <div className="status-icon-ring" style={{ width: '28px', height: '28px', backgroundColor: 'transparent' }}>
             <User size={14} style={{ color: 'var(--success)' }} />
          </div>
        )}
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
      borderColor: variant === "warning" ? "rgba(250, 204, 21, 0.2)" : "rgba(59, 130, 246, 0.2)",
      background: variant === "warning" ? "rgba(250, 204, 21, 0.05)" : "rgba(30, 41, 59, 0.4)"
    }}>
      <div className="status-icon-ring" style={{ 
        color: variant === "warning" ? "var(--warning)" : "var(--accent-blue)",
        backgroundColor: variant === "warning" ? "rgba(250, 204, 21, 0.1)" : "rgba(59, 130, 246, 0.1)"
      }}>
        <Icon size={24} className={variant === "default" ? "float" : ""} />
      </div>
      <div style={{ textAlign: 'center' }}>
        <h4 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 600 }}>{title}</h4>
        <p style={{ margin: '4px 0 0', fontSize: '0.9rem', color: 'var(--muted)', maxWidth: '240px' }}>{subtitle}</p>
      </div>
    </div>
  )
}

function AppShell({ children }) {
  const { apiKey, setApiKey, showSettings, setShowSettings, logout } = useApiKey()
  const [tempKey, setTempKey] = useState(apiKey)
  const [isValidating, setIsValidating] = useState(false)
  const [errorMsg, setErrorMsg] = useState('')

  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  useEffect(() => {
    if (showSettings) {
      setTempKey(apiKey)
      setErrorMsg('')
    }
  }, [showSettings, apiKey])

  const toggleMobileMenu = () => setIsMobileMenuOpen(!isMobileMenuOpen)
  const closeMobileMenu = () => setIsMobileMenuOpen(false)

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
        setErrorMsg('Incorrect Password')
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
        <Link className="brand" to="/" onClick={closeMobileMenu}>
          <Sparkles size={18} />
          <span>VisionX</span>
        </Link>
        
        <button className="mobile-toggle" onClick={toggleMobileMenu}>
          {isMobileMenuOpen ? <X size={22} /> : <Menu size={22} />}
        </button>

        <nav className={`nav ${isMobileMenuOpen ? 'mobile-open' : ''}`}>
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              onClick={closeMobileMenu}
            >
              {item.label}
            </NavLink>
          ))}
          {apiKey ? (
            <button className="secondary-btn" onClick={() => { logout(); closeMobileMenu(); }} style={{ marginLeft: '0.5rem', padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}>
              Log Out
            </button>
          ) : (
            <button className="primary-btn" onClick={() => { setShowSettings(true); closeMobileMenu(); }} style={{ marginTop: 0, marginLeft: '0.5rem', padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}>
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
            {errorMsg && <p style={{ color: '#FF4C4C', fontSize: '0.9rem', marginTop: '0.5rem' }}>{errorMsg}</p>}
            <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem', justifyContent: 'flex-end' }}>
              <button className="secondary-btn" onClick={() => setShowSettings(false)}>Cancel</button>
              <button className="primary-btn" style={{ marginTop: 0 }} onClick={saveSettings} disabled={isValidating}>
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
    fetch(`${API_BASE}/metrics`, {
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
      .catch(() => { })

    // Fetch model info (members)
    fetch(`${API_BASE}/model/info`, {
      headers: { 'X-API-Key': apiKey }
    })
      .then(res => res.json())
      .then(data => {
        if (data.members) setMembers(data.members)
      })
      .catch(() => { })
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
  const [isConverting, setIsConverting] = useState(false)
  const [previewError, setPreviewError] = useState(false)

  const MAX_SIZE_MB = import.meta.env.VITE_MAX_UPLOAD_SIZE_MB || 5

  useEffect(() => {
    if (!file) {
      setPreviewUrl('')
      setPreviewError(false)
      return undefined
    }
    setPreviewError(false)
    const nextPreview = URL.createObjectURL(file)
    setPreviewUrl(nextPreview)
    return () => URL.revokeObjectURL(nextPreview)
  }, [file])

  const handleSelection = async (selectedFile) => {
    setResult(null)
    setError('')
    if (!selectedFile) return

    if (!isSupportedImageFile(selectedFile)) {
      setError('Please upload a valid image file (JPG, PNG, HEIC).')
      return
    }

    if (selectedFile.size > MAX_SIZE_MB * 1024 * 1024) {
      window.alert(`Error: File is too large. Maximum size allowed is ${MAX_SIZE_MB}MB.`)
      return
    }

    const lowerName = selectedFile.name?.toLowerCase() || ''
    if (lowerName.endsWith('.heic') || lowerName.endsWith('.heif')) {
      if (window.heic2any) {
        setIsConverting(true)
        try {
          // Set original file first so it's ready for upload even if preview fails
          setFile(selectedFile)

          const convertedBlob = await window.heic2any({
            blob: selectedFile,
            toType: 'image/jpeg'
          })
          
          // Create a new file for PREVIEW purposes mainly, but also for upload
          const newFile = new File(
            [Array.isArray(convertedBlob) ? convertedBlob[0] : convertedBlob],
            selectedFile.name.replace(/\.(heic|heif)$/i, '.jpg'),
            { type: 'image/jpeg' }
          )
          setFile(newFile)
          console.log('HEIC converted successfully for preview')
        } catch (err) {
          console.warn('HEIC preview conversion failed, proceeding with original file:', err)
          // File is already set to original above
        } finally {
          setIsConverting(false)
        }
      } else {
        setFile(selectedFile)
      }
    } else {
      setFile(selectedFile)
    }
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
      setError(`🚀 Ready for detection!\n
        📸 Upload or capture an image to continue.`)
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
        className={`upload-zone glass-card ${dragActive ? 'drag-active' : ''} ${isConverting ? 'processing' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setDragActive(true)
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => !isConverting && document.getElementById('file-input').click()}
      >
        {isConverting ? (
          <div className="conversion-status">
            <div className="shimmer-loader" style={{ width: '40px', height: '40px', borderRadius: '50%', margin: '0 auto 1rem auto' }}></div>
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
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }} onClick={(e) => e.stopPropagation()}>
              <label className="secondary-btn" onClick={(e) => e.stopPropagation()}>
                Choose Image
                <input
                  id="file-input"
                  type="file"
                  accept=".jpg,.jpeg,.png,.webp,.heic,.heif"
                  hidden
                  onChange={(event) => handleSelection(event.target.files?.[0])}
                />
              </label>
              <button className="secondary-btn" onClick={(e) => { e.stopPropagation(); startCamera(); }}>
                <Camera size={18} style={{ marginRight: '8px' }} />
                Open Camera
              </button>
            </div>
          </>
        )}
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
        <article className="glass-card preview-panel" style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
            <h3 style={{ fontSize: '1.1rem' }}>Source Preview</h3>
            {previewUrl && (
               <span className="badge" style={{ background: 'rgba(59, 130, 246, 0.1)', color: 'var(--accent-blue)', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
                  {imgDims.w}x{imgDims.h}px
               </span>
            )}
          </div>

          <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            {previewUrl ? (
              <div className="preview-image-container fade-in" style={{ boxShadow: '0 0 40px rgba(0,0,0,0.5)', borderRadius: '1rem', overflow: 'hidden' }}>
                {previewError ? (
                  <div className="preview-fallback glass-card" style={{ padding: '3rem 1.5rem', textAlign: 'center', background: 'rgba(15, 23, 42, 0.4)' }}>
                    <div style={{ background: 'rgba(59, 130, 246, 0.1)', width: '60px', height: '60px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1rem auto' }}>
                      <ShieldCheck size={30} style={{ color: 'var(--accent-blue)' }} />
                    </div>
                    <p style={{ fontWeight: 600, color: 'var(--text)', marginBottom: '4px' }}>Apple Format Detected</p>
                    <p className="muted" style={{ fontSize: '0.85rem' }}>Visual preview unavailable in browser,<br/>but AI analysis is fully supported.</p>
                  </div>
                ) : (
                  <img
                    src={previewUrl}
                    alt="Selected preview"
                    className={`preview-image ${isSimulating && simulatedProgress < 90 ? 'processing' : ''}`}
                    onLoad={(e) => setImgDims({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
                    onError={() => setPreviewError(true)}
                    style={{ width: '100%', height: 'auto', borderRadius: '0' }}
                  />
                )}
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
              <div className="premium-empty-state">
                <StatusCard 
                  icon={CloudUpload} 
                  title="No Image Selected" 
                  subtitle="Upload a JPG or PNG to begin AI processing." 
                />
              </div>
            )}
          </div>
          <button className="primary-btn" onClick={runPrediction} disabled={loading || !file} style={{ width: '100%', padding: '0.8rem' }}>
            {loading ? (
              <>
                <Zap size={18} className="pulse" />
                Processing...
              </>
            ) : (
              <>
                <Cpu size={18} />
                Detect Face & Emotion
              </>
            )}
          </button>
        </article>

        <article className="glass-card result-panel">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
             <h3 style={{ fontSize: '1.1rem' }}>AI Analysis Results</h3>
             {result && !isSimulating && (
                <span className="badge" style={{ background: 'rgba(74, 222, 128, 0.1)', color: 'var(--success)', border: '1px solid rgba(74, 222, 128, 0.2)' }}>
                  COMPLETED
                </span>
             )}
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
               <StatusCard 
                 icon={Zap} 
                 title="System Notice" 
                 subtitle={error} 
                 variant="warning"
               />
            </div>
          )}

          {isSimulating && simulatedProgress >= 30 && !result && (
            <div className="results-list">
              <SkeletonCard />
              <SkeletonCard />
              <SkeletonCard />
            </div>
          )}

          {!isSimulating && !error && result && result.results.length === 0 && (
            <div className="premium-empty-state">
              <StatusCard 
                icon={User} 
                title="Zero Matches" 
                subtitle="The AI detector could not identify any clear faces in this image." 
                variant="warning"
              />
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
                const known = result.results.filter(p => p.name !== "UNKNOWN" && p.name.toLowerCase() !== "unknown");
                const unknown = result.results.filter(p => p.name === "UNKNOWN" || p.name.toLowerCase() === "unknown");
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
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                           <HelpCircle size={16} style={{ color: 'var(--danger)' }} />
                           <h4 className="section-title">Unknown Subjects</h4>
                        </div>
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
            <div className="premium-empty-state">
               <StatusCard 
                icon={Cpu} 
                title="Ready for Analysis" 
                subtitle="Upload an image and click the button to trigger AI face & emotion detection." 
              />
            </div>
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
  
  // PRODUCTION LIFECYCLE MANAGEMENT
  const activeSessionRef = useRef(0)
  const isProcessingRef = useRef(false)
  const abortControllerRef = useRef(null)
  const canvasRef = useRef(document.createElement('canvas'))
  const lifecycleRef = useRef('IDLE') // IDLE, STARTING, RUNNING, STOPPING

  const log = (msg, data = '') => {
    const timestamp = new Date().toISOString().split('T')[1].split('Z')[0];
    console.log(`[VisionX][${timestamp}] ${msg}`, data);
  };

  const startCamera = async () => {
    if (lifecycleRef.current === 'STARTING' || lifecycleRef.current === 'RUNNING') {
      log('startCamera ignored - already in progress or running');
      return;
    }

    log('--- STARTING CAMERA ---');
    if (!apiKey) {
      log('Aborted: No API Key');
      setShowSettings(true)
      return
    }

    lifecycleRef.current = 'STARTING';
    setPermissionError('')

    try {
      // 1. Cleanup any ghost tracks first
      log('Step 1: Disposing old session...');
      stopCamera();

      // 2. Request fresh stream
      log('Step 2: Requesting getUserMedia...');
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 }, 
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        } 
      });
      
      if (!videoRef.current) throw new Error('Video element unmounted');

      // 3. Attach and Play
      log('Step 3: Attaching stream and playing...');
      videoRef.current.srcObject = stream;
      
      // We wrap play() in a promise to ensure we wait for the hardware to wake up
      await new Promise((resolve, reject) => {
        if (!videoRef.current) return reject();
        videoRef.current.onplaying = () => {
          log('Event: video.onplaying fired');
          resolve();
        };
        videoRef.current.onerror = reject;
        videoRef.current.play().catch(reject);
      });

      // 4. Finalize session
      activeSessionRef.current += 1;
      lifecycleRef.current = 'RUNNING';
      setStreaming(true);
      log(`--- CAMERA READY (Session #${activeSessionRef.current}) ---`);
    } catch (err) {
      log('--- CAMERA FAILED ---', err);
      lifecycleRef.current = 'IDLE';
      setPermissionError('Camera error: ' + (err.message || 'Unknown failure'));
    }
  }

  const stopCamera = () => {
    log('--- STOPPING CAMERA ---');
    lifecycleRef.current = 'STOPPING';

    // 1. Kill Inference
    if (abortControllerRef.current) {
      log('Killing inference loop AbortController');
      abortControllerRef.current.abort();
    }
    activeSessionRef.current += 1; // Increment again to be sure
    
    // 2. Kill Hardware
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject;
      stream.getTracks().forEach(track => {
        log(`Stopping track: ${track.label}`);
        track.stop();
      });
      videoRef.current.srcObject = null;
      videoRef.current.load(); // Force reset video element
    }
    
    setStreaming(false);
    setDetectedFaces([]);
    setSmoothedFaces([]);
    targetFacesRef.current = [];
    lifecycleRef.current = 'IDLE';
    log('--- CAMERA STOPPED ---');
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

      setSmoothedFaces(prev => {
        // Map current smoothed faces to target faces by ID
        const targets = targetFacesRef.current
        
        // If no targets, fade out or clear (optional)
        if (targets.length === 0) return []

        return targets.map(target => {
          const existing = prev.find(p => p.id === target.id)
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

    log(`[Inference] Initializing loop for Session #${activeSessionRef.current}`);
    const currentSession = activeSessionRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    
    const controller = new AbortController()
    abortControllerRef.current = controller

    const processFrame = async () => {
      // 1. SESSION GUARD
      if (activeSessionRef.current !== currentSession || controller.signal.aborted) {
        log(`[Inference] Terminating loop (Session mismatch or aborted)`);
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
      const frameStart = Date.now();

      try {
        const MAX_INF_WIDTH = 640
        const scale = Math.min(1, MAX_INF_WIDTH / video.videoWidth)
        canvas.width = video.videoWidth * scale
        canvas.height = video.videoHeight * scale
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

        const base64Image = canvas.toDataURL('image/jpeg', 0.6)
        const image_b64 = base64Image.replace(/^data:image\/[a-z]+;base64,/, '')

        log(`[Inference] API Request started (Session #${currentSession})`);
        const response = await fetch(`${API_BASE}/predict/base64`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': apiKey,
          },
          body: JSON.stringify({ image_b64, filename: 'live.jpg' }),
          signal: controller.signal
        })

        if (!response.ok) throw new Error(`API returned ${response.status}`);

        // 2. POST-AWAIT SESSION GUARD
        if (activeSessionRef.current !== currentSession || controller.signal.aborted) {
          log(`[Inference] API returned but session expired. Discarding.`);
          return
        }

        const data = await response.json()
        log(`[Inference] API Success: ${data.results.length} faces detected in ${Date.now() - frameStart}ms`);
        
        const items = data.results.map((face) => {
          let localCrop = null;
          try {
            const cropCanvas = document.createElement('canvas');
            const cropCtx = cropCanvas.getContext('2d');
            const cropSize = 120;
            cropCanvas.width = cropSize;
            cropCanvas.height = cropSize;
            cropCtx.drawImage(canvas, face.x, face.y, face.w, face.h, 0, 0, cropSize, cropSize);
            localCrop = cropCanvas.toDataURL('image/jpeg', 0.85);
          } catch (e) { /* ignore crop error */ }

          return {
            id: face.face_idx,
            name: face.name,
            emotion: face.emotion,
            face_image: localCrop,
            x: (face.x / canvas.width) * 100,
            y: (face.y / canvas.height) * 100,
            w: (face.w / canvas.width) * 100,
            h: (face.h / canvas.height) * 100,
          };
        })

        targetFacesRef.current = items
        setDetectedFaces(items)
      } catch (err) {
        if (err.name === 'AbortError') {
          log('[Inference] Fetch aborted intentionally');
        } else {
          log(`[Inference] Cycle Error: ${err.message}`);
        }
      } finally {
        isProcessingRef.current = false
        if (activeSessionRef.current === currentSession && !controller.signal.aborted) {
          // Stable 150ms throttle
          setTimeout(() => {
            if (activeSessionRef.current === currentSession && !controller.signal.aborted) {
              requestAnimationFrame(processFrame)
            }
          }, 150)
        }
      }
    }

    log(`[Inference] Scheduling loop initialization...`);
    const initTimeout = setTimeout(processFrame, 500)

    return () => {
      log(`[Inference] Cleaning up Session #${currentSession}`);
      controller.abort()
      clearTimeout(initTimeout)
      isProcessingRef.current = false
    }
  }, [streaming, apiKey])

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
                    objectFit: 'cover' 
                  }} 
                />
                {!streaming && (
                  <div className="camera-placeholder">
                    <UserRoundSearch size={48} />
                    <p>Camera feed ready</p>
                  </div>
                )}
                {streaming && smoothedFaces.map((face) => {
                  const isUnknown = !face.name || face.name.toLowerCase() === 'unknown'
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
                        willChange: 'left, top, width, height'
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