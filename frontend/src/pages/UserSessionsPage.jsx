/**
 * src/pages/UserSessionsPage.jsx
 * ────────────────────────────────
 * Session history for regular users — view only.
 * Shows their own sessions with full results_json detail.
 * No delete, no note editing.
 */

import { useEffect, useState } from 'react'
import { RefreshCw, ClipboardList } from 'lucide-react'
import { fetchSessions } from '../api'

export default function UserSessionsPage() {
  const [sessions, setSessions]         = useState([])
  const [loading, setLoading]           = useState(false)
  const [error, setError]               = useState('')
  const [expandedSession, setExpanded]  = useState(null)

  const load = async () => {
    setLoading(true); setError('')
    try {
      const data = await fetchSessions()
      setSessions(data)
    } catch (e) {
      setError('Could not load session history. Is the backend running?')
    }
    setLoading(false)
  }

  useEffect(() => { load() }, [])

  return (
    <section className="fade-in">

      {/* Page header */}
      <div style={{ marginBottom: '2rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <ClipboardList size={22} color="#6366f1" />
          <div>
            <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700, color: '#f1f5f9' }}>My Session History</h2>
            <p style={{ margin: '0.15rem 0 0', color: '#64748b', fontSize: '0.85rem' }}>
              All detection runs you have performed
            </p>
          </div>
        </div>
        <button
          onClick={load}
          title="Refresh"
          style={{
            background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.3)',
            borderRadius: '8px', padding: '0.5rem 0.9rem',
            color: '#a5b4fc', cursor: 'pointer',
            display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem',
          }}
        >
          <RefreshCw size={14} style={loading ? { animation: 'spin 0.8s linear infinite' } : {}} />
          Refresh
          <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
        </button>
      </div>

      {/* Error */}
      {error && (
        <div style={{
          background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
          borderRadius: '8px', padding: '0.75rem 1rem', marginBottom: '1.5rem', color: '#fca5a5',
        }}>
          {error}
        </div>
      )}

      {/* Loading skeleton */}
      {loading && sessions.length === 0 && (
        <div style={{ display: 'grid', gap: '0.75rem' }}>
          {[1, 2, 3].map(i => (
            <div key={i} style={{
              background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(71,85,105,0.2)',
              borderRadius: '10px', padding: '1rem 1.25rem', height: '64px',
              opacity: 0.5,
            }} />
          ))}
        </div>
      )}

      {/* Empty state */}
      {!loading && sessions.length === 0 && !error && (
        <div style={{
          background: 'rgba(15,23,42,0.5)', border: '1px solid rgba(71,85,105,0.2)',
          borderRadius: '12px', padding: '3rem 2rem', textAlign: 'center',
        }}>
          <ClipboardList size={40} color="#334155" style={{ marginBottom: '1rem' }} />
          <p style={{ color: '#64748b', margin: 0, fontSize: '0.95rem' }}>
            No sessions yet. Run a detection from{' '}
            <a href="/upload" style={{ color: '#6366f1', textDecoration: 'none' }}>Upload</a>
            {' '}or{' '}
            <a href="/live" style={{ color: '#6366f1', textDecoration: 'none' }}>Live</a>
            {' '}to get started.
          </p>
        </div>
      )}

      {/* Session list */}
      {sessions.length > 0 && (
        <div style={{ display: 'grid', gap: '0.75rem' }}>
          {sessions.map(s => {
            const isExpanded = expandedSession === s.id
            const results    = s.results_json?.results || (Array.isArray(s.results_json) ? s.results_json : [])
            const known      = results.filter(f => f.name && f.name.toUpperCase() !== 'UNKNOWN')
            const unknown    = results.filter(f => !f.name || f.name.toUpperCase() === 'UNKNOWN')

            return (
              <div key={s.id} style={{
                background: 'rgba(15,23,42,0.6)',
                border: `1px solid ${isExpanded ? 'rgba(99,102,241,0.4)' : 'rgba(71,85,105,0.3)'}`,
                borderRadius: '10px', overflow: 'hidden',
                transition: 'border-color 0.2s',
              }}>

                {/* ── Header row ── */}
                <div style={{
                  padding: '1rem 1.25rem',
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem',
                }}>
                  <div style={{ flex: 1 }}>
                    <p style={{ margin: 0, fontWeight: 600, fontSize: '0.9rem', color: '#f1f5f9' }}>
                      {s.n_faces} face{s.n_faces !== 1 ? 's' : ''} detected
                      <span style={{ color: '#34d399', marginLeft: '0.5rem' }}>· {s.n_identified} identified</span>
                      {s.n_faces - s.n_identified > 0 && (
                        <span style={{ color: '#f87171', marginLeft: '0.5rem' }}>
                          · {s.n_faces - s.n_identified} unknown
                        </span>
                      )}
                    </p>
                    <p style={{ margin: '0.25rem 0 0', color: '#64748b', fontSize: '0.78rem' }}>
                      {s.created_at ? new Date(s.created_at).toLocaleString() : ''}
                      {' · '}{s.elapsed_s}s
                      {s.note && (
                        <span style={{ marginLeft: '0.75rem', color: '#a5b4fc' }}>📝 {s.note}</span>
                      )}
                    </p>
                  </div>

                  {/* Expand button — only show if there's detail to show */}
                  {results.length > 0 && (
                    <button
                      onClick={() => setExpanded(isExpanded ? null : s.id)}
                      style={{
                        background: isExpanded ? 'rgba(99,102,241,0.2)' : 'rgba(71,85,105,0.15)',
                        border: `1px solid ${isExpanded ? 'rgba(99,102,241,0.4)' : 'rgba(71,85,105,0.3)'}`,
                        borderRadius: '6px', padding: '0.3rem 0.75rem',
                        color: isExpanded ? '#a5b4fc' : '#94a3b8',
                        cursor: 'pointer', fontSize: '0.75rem', fontWeight: 600,
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {isExpanded ? '▲ Hide' : '▼ Details'}
                    </button>
                  )}
                </div>

                {/* ── Expanded results ── */}
                {isExpanded && (
                  <div style={{
                    borderTop: '1px solid rgba(71,85,105,0.25)',
                    padding: '1rem 1.25rem',
                    background: 'rgba(15,23,42,0.4)',
                  }}>

                    {/* Annotated image */}
                    {s.annotated_image && (
                      <div style={{ marginBottom: '1rem' }}>
                        <p style={{ margin: '0 0 0.5rem', fontSize: '0.78rem', color: '#94a3b8', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                          📸 Annotated Image
                        </p>
                        <img
                          src={`data:image/jpeg;base64,${s.annotated_image}`}
                          alt="Annotated detection"
                          style={{
                            width: '100%', maxWidth: '480px', borderRadius: '8px',
                            border: '1px solid rgba(99,102,241,0.3)',
                            display: 'block',
                          }}
                        />
                      </div>
                    )}

                    {/* Summary pills */}
                    <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
                      <span style={{ background: 'rgba(99,102,241,0.15)', border: '1px solid rgba(99,102,241,0.3)', borderRadius: '20px', padding: '0.2rem 0.75rem', fontSize: '0.75rem', color: '#a5b4fc', fontWeight: 600 }}>
                        Total: {results.length}
                      </span>
                      <span style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)', borderRadius: '20px', padding: '0.2rem 0.75rem', fontSize: '0.75rem', color: '#34d399', fontWeight: 600 }}>
                        Identified: {known.length}
                      </span>
                      <span style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '20px', padding: '0.2rem 0.75rem', fontSize: '0.75rem', color: '#f87171', fontWeight: 600 }}>
                        Unknown: {unknown.length}
                      </span>
                    </div>

                    {/* Known faces */}
                    {known.length > 0 && (
                      <div style={{ marginBottom: '0.75rem' }}>
                        <p style={{ margin: '0 0 0.5rem', fontSize: '0.78rem', color: '#34d399', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                          ✅ Identified
                        </p>
                        <div style={{ display: 'grid', gap: '0.4rem' }}>
                          {known.map((face, idx) => (
                            <div key={idx} style={{
                              background: 'rgba(16,185,129,0.07)', border: '1px solid rgba(16,185,129,0.2)',
                              borderRadius: '8px', padding: '0.5rem 0.75rem',
                              display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem',
                            }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                                <span style={{ fontSize: '1rem' }}>👤</span>
                                <span style={{ fontWeight: 600, fontSize: '0.875rem', color: '#f1f5f9' }}>{face.name}</span>
                              </div>
                              <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                                {face.emotion && (
                                  <span style={{ background: 'rgba(99,102,241,0.15)', borderRadius: '12px', padding: '0.15rem 0.6rem', fontSize: '0.72rem', color: '#a5b4fc', fontWeight: 600 }}>
                                    {face.emotion}
                                  </span>
                                )}
                                {face.confidence !== undefined && (
                                  <span style={{ fontSize: '0.72rem', color: '#64748b' }}>
                                    {(face.confidence * 100).toFixed(1)}% conf.
                                  </span>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Unknown faces */}
                    {unknown.length > 0 && (
                      <div>
                        <p style={{ margin: '0 0 0.5rem', fontSize: '0.78rem', color: '#f87171', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                          ❓ Unknown
                        </p>
                        <div style={{ display: 'grid', gap: '0.4rem' }}>
                          {unknown.map((face, idx) => (
                            <div key={idx} style={{
                              background: 'rgba(239,68,68,0.07)', border: '1px solid rgba(239,68,68,0.2)',
                              borderRadius: '8px', padding: '0.5rem 0.75rem',
                              display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem',
                            }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                                <span style={{ fontSize: '1rem' }}>❓</span>
                                <span style={{ fontWeight: 600, fontSize: '0.875rem', color: '#94a3b8' }}>Unknown Face #{idx + 1}</span>
                              </div>
                              {face.emotion && (
                                <span style={{ background: 'rgba(99,102,241,0.15)', borderRadius: '12px', padding: '0.15rem 0.6rem', fontSize: '0.72rem', color: '#a5b4fc', fontWeight: 600 }}>
                                  {face.emotion}
                                </span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}
