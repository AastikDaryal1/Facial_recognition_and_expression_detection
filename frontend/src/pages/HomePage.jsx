/**
 * src/pages/HomePage.jsx
 * ───────────────────────
 * Home/landing page — shown to regular users after login.
 * Extracted from old App.jsx LandingPage component.
 *
 * Changes from old version:
 *   - Uses fetchModelInfo() and fetchMetrics() from api.js (JWT Bearer)
 *   - Removed API key checks — auth is handled globally by AuthContext
 *   - Shows welcome message with user's email
 */

import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Activity,
  Zap,
  Cpu,
  Users,
  ImagePlus,
  ScanFace,
  ShieldCheck,
} from 'lucide-react'
import { useAuth } from '../AuthContext'
import { fetchModelInfo, fetchMetrics } from '../api'

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

export default function HomePage() {
  const { user } = useAuth()
  const [stats,   setStats]   = useState({ uptime: '0s', latency: '0s', count: 0 })
  const [members, setMembers] = useState([])

  useEffect(() => {
    // Fetch metrics
    fetchMetrics()
      .then(data => {
        setStats({
          uptime  : `${data.uptime_s || 0}s`,
          latency : `${data.avg_latency_s || 0}s`,
          count   : data.request_count || 0,
        })
      })
      .catch(() => {})

    // Fetch model info
    fetchModelInfo()
      .then(data => {
        if (data.members) setMembers(data.members)
      })
      .catch(() => {})
  }, [])

  return (
    <section className="fade-in">
      <div className="hero glass-card">
        <p className="kicker">Computer Vision + Emotion AI</p>
        <h1>Face & Emotion Detection System</h1>
        <p className="hero-subtitle">
          Detect faces, identify known people, and estimate real-time emotions
          from uploaded images or a live camera feed.
        </p>

        {/* Welcome message */}
        {user && (
          <p style={{
            color      : '#64748b',
            fontSize   : '0.9rem',
            marginTop  : '0.5rem',
          }}>
            Welcome back, <span style={{ color: '#a5b4fc' }}>{user.email}</span>
          </p>
        )}

        {/* Stats */}
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

      {/* Action cards */}
      <div className="action-grid">
        <ActionCard
          icon        = {ImagePlus}
          title       = "Upload Photo"
          description = "Drop an image and receive face, identity, and emotion predictions in seconds."
          buttonText  = "Open Upload"
          to          = "/upload"
        />
        <ActionCard
          icon        = {ScanFace}
          title       = "Live Detection"
          description = "Use webcam-based real-time analysis with dynamic overlays and live updates."
          buttonText  = "Open Live View"
          to          = "/live"
        />
      </div>

      {/* Core capabilities */}
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
              <p>JWT-based authentication protecting your analytics and results.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Recognition library */}
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
