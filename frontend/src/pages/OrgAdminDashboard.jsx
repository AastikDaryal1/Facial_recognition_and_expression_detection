/**
 * src/pages/OrgAdminDashboard.jsx
 * ─────────────────────────────────
 * Org Admin dashboard — manage own organisation's users and persons.
 * Tabs: Overview, Team Members, Enrolled Persons, Sessions
 */

import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Sparkles, LogOut, Users, UserCheck, UserX,
  Plus, X, Send, RefreshCw, ScanFace,
  ClipboardList, Activity
} from 'lucide-react'
import { useAuth } from '../AuthContext'
import {
  fetchUsers, deactivateUser, activateUser,
  fetchPersons, createPerson, deletePerson,
  fetchSessions, deleteSession, updateSessionNote,
  inviteUser,
  uploadPersonPhotos, retrainPerson,
} from '../api'

const TABS = ['Overview', 'Team Members', 'Enrolled Persons', 'Sessions']

const ROLE_BADGE = {
  super_admin: { label: 'Super Admin', color: '#f59e0b', bg: 'rgba(245,158,11,0.1)' },
  org_admin: { label: 'Org Admin', color: '#6366f1', bg: 'rgba(99,102,241,0.1)' },
  user: { label: 'Member', color: '#10b981', bg: 'rgba(16,185,129,0.1)' },
}

function Badge({ role }) {
  const b = ROLE_BADGE[role] || { label: role, color: '#94a3b8', bg: 'rgba(148,163,184,0.1)' }
  return (
    <span style={{
      background: b.bg, color: b.color,
      border: `1px solid ${b.color}40`,
      borderRadius: '20px', padding: '0.2rem 0.7rem',
      fontSize: '0.75rem', fontWeight: 600,
    }}>{b.label}</span>
  )
}

function StatCard({ icon: Icon, label, value, color = '#6366f1' }) {
  return (
    <div style={{
      background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(99,102,241,0.2)',
      borderRadius: '12px', padding: '1.25rem',
      display: 'flex', alignItems: 'center', gap: '1rem',
    }}>
      <div style={{ background: `${color}20`, borderRadius: '10px', padding: '0.75rem', color }}>
        <Icon size={20} />
      </div>
      <div>
        <p style={{ margin: 0, color: '#64748b', fontSize: '0.8rem' }}>{label}</p>
        <p style={{ margin: 0, color: '#f1f5f9', fontSize: '1.4rem', fontWeight: 700 }}>{value}</p>
      </div>
    </div>
  )
}

export default function OrgAdminDashboard() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const [tab, setTab] = useState('Overview')
  const [users, setUsers] = useState([])
  const [persons, setPersons] = useState([])
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Invite modal
  const [showInvite, setShowInvite] = useState(false)
  const [inviteEmail, setInviteEmail] = useState('')
  const [inviteRole] = useState('user') // Default to 'user' for Org Admin invites
  const [inviteResult, setInviteResult] = useState(null)
  const [inviting, setInviting] = useState(false)

  // Session expand + note editing
  const [expandedSession, setExpandedSession] = useState(null)
  const [editingNoteId, setEditingNoteId] = useState(null)
  const [noteText, setNoteText] = useState('')
  const [savingNote, setSavingNote] = useState(false)

  // New person modal
  const [showNewPerson, setShowNewPerson] = useState(false)
  const [personName, setPersonName] = useState('')
  const [personEmpId, setPersonEmpId] = useState('')
  const [personDept, setPersonDept] = useState('')
  const [creatingPerson, setCreatingPerson] = useState(false)

  // Photo upload state
  const [uploadingPersonId, setUploadingPersonId] = useState(null)
  const [uploadFiles, setUploadFiles] = useState([])
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState('')

  // Retrain state
  const [retrainingPersonId, setRetrainingPersonId] = useState(null)

  const load = async () => {
    setLoading(true); setError('')
    try {
      const [uRes, pRes, sRes] = await Promise.allSettled([
        fetchUsers(), fetchPersons(), fetchSessions()
      ])
      if (uRes.status === 'fulfilled') setUsers(uRes.value)
      else console.error('fetchUsers failed:', uRes.reason)

      if (pRes.status === 'fulfilled') setPersons(pRes.value)
      else console.error('fetchPersons failed:', pRes.reason)

      if (sRes.status === 'fulfilled') setSessions(sRes.value)
      else console.error('fetchSessions failed:', sRes.reason)

      // Only show error banner if ALL three failed (total outage)
      const allFailed = [uRes, pRes, sRes].every(r => r.status === 'rejected')
      if (allFailed) setError('Failed to connect to the server. Is the backend running?')

    } catch (e) { setError(e.message) }
    setLoading(false)
  }

  useEffect(() => { load() }, [])

  const handleInvite = async () => {
    if (!inviteEmail) return
    setInviting(true); setInviteResult('')
    try {
      const res = await inviteUser(inviteEmail, inviteRole)
      setInviteResult({
        success: true,
        email: inviteEmail,
        token: res.invite_token,
        emailSent: res.email_sent,
      })
      setInviteEmail('')
    } catch (e) {
      setInviteResult({ success: false, message: e.message })
    }
    setInviting(false)
  }

  const handleCreatePerson = async () => {
    if (!personName) return
    setCreatingPerson(true)
    try {
      await createPerson(personName, personEmpId, personDept)
      setPersonName(''); setPersonEmpId(''); setPersonDept('')
      setShowNewPerson(false)
      load()
    } catch (e) { setError(e.message) }
    setCreatingPerson(false)
  }

  const handleDeactivate = async (id) => {
    try { await deactivateUser(id); load() } catch (e) { setError(e.message) }
  }

  const handleActivate = async (id) => {
    try { await activateUser(id); load() } catch (e) { setError(e.message) }
  }

  const handleDeletePerson = async (id) => {
    if (!window.confirm('Remove this person from the enrolment list?')) return
    try { await deletePerson(id); load() } catch (e) { setError(e.message) }
  }

  const handleUploadPhotos = async (personId) => {
    if (!uploadFiles.length) return
    setUploading(true)
    setUploadError('')
    try {
      await uploadPersonPhotos(personId, uploadFiles)
      setUploadingPersonId(null)
      setUploadFiles([])
      load()
    } catch (e) {
      setUploadError(e.message)
    }
    setUploading(false)
  }

  const handleRetrain = async (personId) => {
    if (!window.confirm('This will run the full ML pipeline and may take a few minutes. Continue?')) return
    setRetrainingPersonId(personId)
    try {
      await retrainPerson(personId)
      load()
    } catch (e) {
      setError(e.message)
    }
    setRetrainingPersonId(null)
  }

  const handleSaveNote = async (sessionId) => {
    setSavingNote(true)
    try {
      const updated = await updateSessionNote(sessionId, noteText)
      setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, note: updated.note } : s))
      setEditingNoteId(null)
    } catch (e) { setError(e.message) }
    setSavingNote(false)
  }

  const handleDeleteSession = async (id) => {
    if (!window.confirm('Delete this session record?')) return
    try { await deleteSession(id); load() } catch (e) { setError(e.message) }
  }

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)', color: '#f1f5f9' }}>

      {/* Navbar */}
      <header style={{
        background: 'rgba(15,23,42,0.8)', backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(99,102,241,0.2)',
        padding: '0 2rem', height: '60px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        position: 'sticky', top: 0, zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Sparkles size={18} color="#6366f1" />
          <span style={{ fontWeight: 700, fontSize: '1.1rem' }}>VisionX</span>
          <span style={{ color: '#64748b', fontSize: '0.85rem', marginLeft: '0.5rem' }}>Org Admin</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span style={{ color: '#64748b', fontSize: '0.85rem' }}>{user?.email}</span>
          <Badge role="org_admin" />
          <button onClick={() => navigate('/upload')} style={{
            background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.3)',
            borderRadius: '8px', padding: '0.4rem 0.8rem', color: '#a5b4fc',
            cursor: 'pointer', fontSize: '0.85rem',
          }}>
            Detection
          </button>
          <button onClick={logout} style={{
            background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
            borderRadius: '8px', padding: '0.4rem 0.8rem', color: '#f87171',
            cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem',
          }}>
            <LogOut size={14} /> Log Out
          </button>
        </div>
      </header>

      <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '2rem' }}>

        <div style={{ marginBottom: '2rem' }}>
          <h1 style={{ margin: 0, fontSize: '1.8rem', fontWeight: 700 }}>Organisation Dashboard</h1>
          <p style={{ margin: '0.25rem 0 0', color: '#64748b' }}>Manage your team, enrolments and sessions</p>
        </div>

        {error && (
          <div style={{
            background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
            borderRadius: '8px', padding: '0.75rem 1rem', marginBottom: '1.5rem', color: '#fca5a5',
          }}>{error}</div>
        )}

        {/* Tabs */}
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '2rem', borderBottom: '1px solid rgba(71,85,105,0.3)' }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              background: 'none', border: 'none', cursor: 'pointer',
              padding: '0.6rem 1.2rem', fontSize: '0.9rem', fontWeight: 600,
              color: tab === t ? '#6366f1' : '#64748b',
              borderBottom: tab === t ? '2px solid #6366f1' : '2px solid transparent',
              marginBottom: '-1px',
            }}>{t}</button>
          ))}
          <button onClick={load} style={{ marginLeft: 'auto', background: 'none', border: 'none', color: '#64748b', cursor: 'pointer', padding: '0.6rem' }}>
            <RefreshCw size={16} />
          </button>
        </div>

        {/* ── Overview ──────────────────────────────────────────────────── */}
        {tab === 'Overview' && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
              <StatCard icon={Users} label="Team Members" value={users.length} color="#6366f1" />
              <StatCard icon={ScanFace} label="Enrolled Persons" value={persons.length} color="#10b981" />
              <StatCard icon={ClipboardList} label="Sessions" value={sessions.length} color="#f59e0b" />
            </div>
            <div style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(99,102,241,0.2)', borderRadius: '12px', padding: '1.5rem' }}>
              <h3 style={{ margin: '0 0 1rem' }}>Quick Actions</h3>
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                <button onClick={() => setShowInvite(true)} style={{
                  background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                  border: 'none', borderRadius: '8px', padding: '0.7rem 1.2rem',
                  color: '#fff', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600,
                }}>
                  <Send size={16} /> Invite Team Member
                </button>
                <button onClick={() => setShowNewPerson(true)} style={{
                  background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)',
                  borderRadius: '8px', padding: '0.7rem 1.2rem',
                  color: '#34d399', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600,
                }}>
                  <Plus size={16} /> Enrol Person
                </button>
                <button onClick={() => navigate('/upload')} style={{
                  background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.3)',
                  borderRadius: '8px', padding: '0.7rem 1.2rem',
                  color: '#a5b4fc', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600,
                }}>
                  <Activity size={16} /> Run Detection
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── Team Members ──────────────────────────────────────────────── */}
        {tab === 'Team Members' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0 }}>Team Members ({users.length})</h3>
              <button onClick={() => setShowInvite(true)} style={{
                background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                border: 'none', borderRadius: '8px', padding: '0.6rem 1rem',
                color: '#fff', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.875rem', fontWeight: 600,
              }}>
                <Send size={14} /> Invite Member
              </button>
            </div>
            <div style={{ display: 'grid', gap: '0.75rem' }}>
              {users.map(u => (
                <div key={u.id} style={{
                  background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(71,85,105,0.3)',
                  borderRadius: '10px', padding: '1rem 1.25rem',
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem',
                  opacity: u.is_active ? 1 : 0.6,
                }}>
                  <div style={{ flex: 1 }}>
                    <p style={{ margin: 0, fontWeight: 600, fontSize: '0.95rem' }}>{u.email}</p>
                    <p style={{ margin: '0.2rem 0 0', color: '#64748b', fontSize: '0.78rem' }}>ID: {u.id}</p>
                  </div>
                  <Badge role={u.role} />
                  <span style={{
                    background: u.is_active ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)',
                    color: u.is_active ? '#34d399' : '#f87171',
                    border: `1px solid ${u.is_active ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)'}`,
                    borderRadius: '20px', padding: '0.2rem 0.7rem', fontSize: '0.75rem', fontWeight: 600,
                  }}>
                    {u.is_active ? 'Active' : 'Inactive'}
                  </span>
                  {u.email !== user?.email && u.role !== 'super_admin' && (
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      {u.is_active
                        ? <button onClick={() => handleDeactivate(u.id)} title="Deactivate" style={{ background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.3)', borderRadius: '6px', padding: '0.3rem 0.5rem', color: '#fbbf24', cursor: 'pointer' }}>
                          <UserX size={14} />
                        </button>
                        : <button onClick={() => handleActivate(u.id)} title="Activate" style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)', borderRadius: '6px', padding: '0.3rem 0.5rem', color: '#34d399', cursor: 'pointer' }}>
                          <UserCheck size={14} />
                        </button>
                      }
                    </div>
                  )}
                </div>
              ))}
              {users.length === 0 && <p style={{ color: '#64748b' }}>No team members yet. Invite someone to get started.</p>}
            </div>
          </div>
        )}

        {/* ── Enrolled Persons ──────────────────────────────────────────── */}
        {tab === 'Enrolled Persons' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0 }}>Enrolled Persons ({persons.length})</h3>
              <button onClick={() => setShowNewPerson(true)} style={{
                background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                border: 'none', borderRadius: '8px', padding: '0.6rem 1rem',
                color: '#fff', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.875rem', fontWeight: 600,
              }}>
                <Plus size={14} /> Add Person
              </button>
            </div>
            <div style={{ display: 'grid', gap: '0.75rem' }}>
              {persons.map(p => (
                <div key={p.id} style={{
                  background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(71,85,105,0.3)',
                  borderRadius: '10px', padding: '1rem 1.25rem',
                }}>
                  {/* ── Person header row ── */}
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem' }}>
                    <div style={{ flex: 1 }}>
                      <p style={{ margin: 0, fontWeight: 600 }}>{p.full_name}</p>
                      <p style={{ margin: '0.2rem 0 0', color: '#64748b', fontSize: '0.78rem' }}>
                        {p.employee_id && `EMP: ${p.employee_id}`}{p.department && ` | ${p.department}`}
                        {' · '}{p.photo_count} photo{p.photo_count !== 1 ? 's' : ''}
                        {p.gcs_path && <span style={{ color: '#6366f1', marginLeft: 6 }}>☁ GCS</span>}
                      </p>
                    </div>

                    {/* Status badge */}
                    <span style={{
                      background: p.is_enrolled ? 'rgba(16,185,129,0.1)' : 'rgba(245,158,11,0.1)',
                      color: p.is_enrolled ? '#34d399' : '#fbbf24',
                      border: `1px solid ${p.is_enrolled ? 'rgba(16,185,129,0.3)' : 'rgba(245,158,11,0.3)'}`,
                      borderRadius: '20px', padding: '0.2rem 0.7rem', fontSize: '0.75rem', fontWeight: 600,
                      whiteSpace: 'nowrap',
                    }}>
                      {p.is_enrolled ? '✓ Enrolled' : '⏳ Pending'}
                    </span>

                    {/* Upload photos button */}
                    <button
                      onClick={() => {
                        setUploadingPersonId(uploadingPersonId === p.id ? null : p.id)
                        setUploadFiles([])
                        setUploadError('')
                      }}
                      title="Upload photos"
                      style={{
                        background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.3)',
                        borderRadius: '6px', padding: '0.3rem 0.6rem', color: '#818cf8',
                        cursor: 'pointer', fontSize: '0.75rem', fontWeight: 600,
                      }}>
                      📷 Photos
                    </button>

                    {/* Retrain button — only show when photos exist */}
                    {p.photo_count >= 5 && (
                      <button
                        onClick={() => handleRetrain(p.id)}
                        disabled={retrainingPersonId === p.id}
                        title="Run ML pipeline and enrol"
                        style={{
                          background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)',
                          borderRadius: '6px', padding: '0.3rem 0.6rem', color: '#34d399',
                          cursor: retrainingPersonId === p.id ? 'wait' : 'pointer',
                          fontSize: '0.75rem', fontWeight: 600,
                          opacity: retrainingPersonId === p.id ? 0.6 : 1,
                        }}>
                        {retrainingPersonId === p.id ? '⏳ Training…' : '🔁 Retrain'}
                      </button>
                    )}

                    {/* Delete */}
                    <button onClick={() => handleDeletePerson(p.id)} style={{
                      background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
                      borderRadius: '6px', padding: '0.3rem 0.5rem', color: '#f87171', cursor: 'pointer',
                    }}>
                      <X size={14} />
                    </button>
                  </div>

                  {/* ── Photo upload panel (expands inline) ── */}
                  {uploadingPersonId === p.id && (
                    <div style={{
                      marginTop: '0.75rem', paddingTop: '0.75rem',
                      borderTop: '1px solid rgba(71,85,105,0.3)',
                    }}>
                      <p style={{ margin: '0 0 0.5rem', fontSize: '0.8rem', color: '#94a3b8' }}>
                        Select face photos for <strong>{p.full_name}</strong> (jpeg/png/webp, min 5 for training):
                      </p>
                      <input
                        type="file"
                        multiple
                        accept="image/jpeg,image/png,image/webp"
                        onChange={e => setUploadFiles(Array.from(e.target.files))}
                        style={{ color: '#f1f5f9', fontSize: '0.8rem', marginBottom: '0.5rem', display: 'block' }}
                      />
                      {uploadFiles.length > 0 && (
                        <p style={{ margin: '0 0 0.5rem', color: '#6366f1', fontSize: '0.78rem' }}>
                          {uploadFiles.length} file{uploadFiles.length !== 1 ? 's' : ''} selected
                        </p>
                      )}
                      {uploadError && (
                        <p style={{ margin: '0 0 0.5rem', color: '#f87171', fontSize: '0.78rem' }}>{uploadError}</p>
                      )}
                      <div style={{ display: 'flex', gap: '0.5rem' }}>
                        <button
                          onClick={() => handleUploadPhotos(p.id)}
                          disabled={uploading || !uploadFiles.length}
                          style={{
                            background: uploading || !uploadFiles.length
                              ? 'rgba(99,102,241,0.3)' : 'linear-gradient(135deg,#6366f1,#8b5cf6)',
                            border: 'none', borderRadius: '6px', padding: '0.4rem 0.9rem',
                            color: '#fff', cursor: uploading || !uploadFiles.length ? 'not-allowed' : 'pointer',
                            fontSize: '0.8rem', fontWeight: 600,
                          }}>
                          {uploading ? 'Uploading…' : 'Upload'}
                        </button>
                        <button
                          onClick={() => { setUploadingPersonId(null); setUploadFiles([]); setUploadError('') }}
                          style={{
                            background: 'rgba(71,85,105,0.2)', border: '1px solid rgba(71,85,105,0.3)',
                            borderRadius: '6px', padding: '0.4rem 0.7rem',
                            color: '#94a3b8', cursor: 'pointer', fontSize: '0.8rem',
                          }}>
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
              {persons.length === 0 && <p style={{ color: '#64748b' }}>No persons added yet. Click "Add Person" to get started.</p>}
            </div>
          </div>
        )}


        {/* ── Sessions ──────────────────────────────────────────────────── */}
        {tab === 'Sessions' && (
          <div>
            <h3 style={{ margin: '0 0 1rem' }}>Session History ({sessions.length})</h3>
            {sessions.length === 0
              ? <p style={{ color: '#64748b' }}>No sessions yet. Sessions will appear here after face detection runs.</p>
              : (
                <div style={{ display: 'grid', gap: '0.75rem' }}>
                  {sessions.map(s => {
                    const isExpanded = expandedSession === s.id
                    const isEditingNote = editingNoteId === s.id
                    const results = s.results_json?.results || (Array.isArray(s.results_json) ? s.results_json : [])
                    const known = results.filter(f => f.name && f.name.toUpperCase() !== 'UNKNOWN')
                    const unknown = results.filter(f => !f.name || f.name.toUpperCase() === 'UNKNOWN')

                    return (
                      <div key={s.id} style={{
                        background: 'rgba(15,23,42,0.6)', border: `1px solid ${isExpanded ? 'rgba(99,102,241,0.4)' : 'rgba(71,85,105,0.3)'}`,
                        borderRadius: '10px', overflow: 'hidden',
                        transition: 'border-color 0.2s',
                      }}>
                        {/* ── Session header row ── */}
                        <div style={{
                          padding: '1rem 1.25rem',
                          display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem',
                        }}>
                          <div style={{ flex: 1 }}>
                            <p style={{ margin: 0, fontWeight: 600, fontSize: '0.9rem' }}>
                              {s.n_faces} face{s.n_faces !== 1 ? 's' : ''} detected
                              <span style={{ color: '#34d399', marginLeft: '0.5rem' }}>· {s.n_identified} identified</span>
                              {s.n_faces - s.n_identified > 0 && (
                                <span style={{ color: '#f87171', marginLeft: '0.5rem' }}>· {s.n_faces - s.n_identified} unknown</span>
                              )}
                            </p>
                            <p style={{ margin: '0.2rem 0 0', color: '#64748b', fontSize: '0.78rem' }}>
                              {s.created_at ? new Date(s.created_at).toLocaleString() : ''} · {s.elapsed_s}s
                              {s.note && <span style={{ marginLeft: '0.75rem', color: '#a5b4fc' }}>📝 {s.note}</span>}
                            </p>
                          </div>

                          {/* Action buttons */}
                          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                            {/* Expand / collapse */}
                            <button
                              onClick={() => setExpandedSession(isExpanded ? null : s.id)}
                              title={isExpanded ? 'Collapse' : 'View Details'}
                              style={{
                                background: isExpanded ? 'rgba(99,102,241,0.2)' : 'rgba(71,85,105,0.15)',
                                border: `1px solid ${isExpanded ? 'rgba(99,102,241,0.4)' : 'rgba(71,85,105,0.3)'}`,
                                borderRadius: '6px', padding: '0.3rem 0.6rem',
                                color: isExpanded ? '#a5b4fc' : '#94a3b8', cursor: 'pointer',
                                fontSize: '0.75rem', fontWeight: 600,
                              }}
                            >
                              {isExpanded ? '▲ Hide' : '▼ Details'}
                            </button>
                            {/* Edit note */}
                            <button
                              onClick={() => { setEditingNoteId(s.id); setNoteText(s.note || '') }}
                              title="Add / Edit Note"
                              style={{
                                background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.3)',
                                borderRadius: '6px', padding: '0.3rem 0.5rem', color: '#a5b4fc', cursor: 'pointer',
                              }}
                            >
                              📝
                            </button>
                            {/* Delete */}
                            <button
                              onClick={() => handleDeleteSession(s.id)}
                              title="Delete Session"
                              style={{
                                background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
                                borderRadius: '6px', padding: '0.3rem 0.5rem', color: '#f87171', cursor: 'pointer',
                              }}
                            >
                              <X size={14} />
                            </button>
                          </div>
                        </div>

                        {/* ── Note editor ── */}
                        {isEditingNote && (
                          <div style={{
                            padding: '0 1.25rem 1rem',
                            borderTop: '1px solid rgba(71,85,105,0.2)',
                          }}>
                            <label style={{ display: 'block', color: '#94a3b8', fontSize: '0.8rem', margin: '0.75rem 0 0.4rem' }}>
                              Session Note
                            </label>
                            <textarea
                              value={noteText}
                              onChange={e => setNoteText(e.target.value)}
                              placeholder="Add a note about this session..."
                              rows={3}
                              style={{
                                width: '100%', background: 'rgba(30,41,59,0.8)',
                                border: '1px solid rgba(99,102,241,0.4)', borderRadius: '8px',
                                padding: '0.6rem', color: '#f1f5f9', fontSize: '0.85rem',
                                outline: 'none', resize: 'vertical', boxSizing: 'border-box',
                              }}
                            />
                            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
                              <button
                                onClick={() => handleSaveNote(s.id)}
                                disabled={savingNote}
                                style={{
                                  background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                                  border: 'none', borderRadius: '6px', padding: '0.45rem 1rem',
                                  color: '#fff', fontWeight: 600, fontSize: '0.8rem',
                                  cursor: savingNote ? 'not-allowed' : 'pointer', opacity: savingNote ? 0.7 : 1,
                                }}
                              >
                                {savingNote ? 'Saving...' : 'Save Note'}
                              </button>
                              <button
                                onClick={() => setEditingNoteId(null)}
                                style={{
                                  background: 'rgba(71,85,105,0.2)', border: '1px solid rgba(71,85,105,0.3)',
                                  borderRadius: '6px', padding: '0.45rem 1rem',
                                  color: '#94a3b8', fontSize: '0.8rem', cursor: 'pointer',
                                }}
                              >
                                Cancel
                              </button>
                            </div>
                          </div>
                        )}

                        {/* ── Expanded detail: results_json ── */}
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

                            {results.length === 0 ? (
                              <p style={{ color: '#64748b', fontSize: '0.85rem', margin: 0 }}>No detailed results stored for this session.</p>
                            ) : (
                              <>
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
                              </>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )
            }
          </div>
        )}
      </div>

      {/* ── Invite Modal ──────────────────────────────────────────────────── */}
      {showInvite && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 200 }}>
          <div style={{ background: '#1e293b', border: '1px solid rgba(99,102,241,0.3)', borderRadius: '16px', padding: '2rem', width: '100%', maxWidth: '400px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h3 style={{ margin: 0 }}>Invite Team Member</h3>
              <button onClick={() => { setShowInvite(false); setInviteResult('') }} style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}><X size={20} /></button>
            </div>
            {/* Success state */}
            {inviteResult?.success ? (
              <div>
                <div style={{
                  background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)',
                  borderRadius: '8px', padding: '1rem', marginBottom: '1.2rem',
                }}>
                  <p style={{ color: '#34d399', fontWeight: 600, margin: '0 0 0.5rem', fontSize: '0.9rem' }}>
                    ✅ {inviteResult.emailSent
                      ? `Invite email sent to ${inviteResult.email}`
                      : `Invite generated for ${inviteResult.email} (email not sent)`}
                  </p>
                  <p style={{ color: '#64748b', fontSize: '0.78rem', margin: '0 0 0.75rem' }}>
                    Copy this token and share it with the user. They should go to:<br />
                    <span style={{ color: '#a5b4fc' }}>
                      {window.location.origin}/signup?token=...
                    </span>
                  </p>
                  <div style={{
                    background: 'rgba(15,23,42,0.8)', borderRadius: '6px',
                    padding: '0.75rem', fontSize: '0.72rem', color: '#94a3b8',
                    wordBreak: 'break-all', marginBottom: '0.75rem',
                    maxHeight: '80px', overflowY: 'auto',
                    border: '1px solid rgba(71,85,105,0.3)',
                  }}>
                    {inviteResult.token}
                  </div>
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(
                        `${window.location.origin}/signup?token=${inviteResult.token}`
                      )
                      alert('Signup link copied to clipboard!')
                    }}
                    style={{
                      width: '100%', background: 'rgba(99,102,241,0.2)',
                      border: '1px solid rgba(99,102,241,0.4)',
                      borderRadius: '8px', padding: '0.6rem',
                      color: '#a5b4fc', cursor: 'pointer', fontWeight: 600, fontSize: '0.875rem',
                    }}
                  >
                    📋 Copy Signup Link
                  </button>
                </div>
                <button
                  onClick={() => { setShowInvite(false); setInviteResult(null) }}
                  style={{
                    width: '100%', background: 'rgba(71,85,105,0.2)',
                    border: '1px solid rgba(71,85,105,0.3)',
                    borderRadius: '8px', padding: '0.7rem',
                    color: '#94a3b8', cursor: 'pointer', fontWeight: 600,
                  }}
                >
                  Close
                </button>
              </div>
            ) : (
              /* Input state */
              <div>
                <div style={{ marginBottom: '1.5rem' }}>
                  <label style={{ display: 'block', color: '#94a3b8', fontSize: '0.85rem', marginBottom: '0.4rem' }}>Email</label>
                  <input type="email" value={inviteEmail} onChange={e => setInviteEmail(e.target.value)} placeholder="employee@company.com"
                    style={{ width: '100%', background: 'rgba(30,41,59,0.8)', border: '1px solid rgba(71,85,105,0.5)', borderRadius: '8px', padding: '0.7rem', color: '#f1f5f9', fontSize: '0.9rem', outline: 'none', boxSizing: 'border-box' }} />
                  <p style={{ color: '#64748b', fontSize: '0.78rem', marginTop: '0.3rem' }}>They will be added as a regular team member (user role).</p>
                </div>
                {inviteResult?.success === false && (
                  <div style={{
                    background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
                    borderRadius: '8px', padding: '0.75rem', marginBottom: '1rem',
                    color: '#fca5a5', fontSize: '0.85rem',
                  }}>{inviteResult.message}</div>
                )}
                <button onClick={handleInvite} disabled={inviting} style={{
                  width: '100%', background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                  border: 'none', borderRadius: '8px', padding: '0.8rem',
                  color: '#fff', fontWeight: 600, cursor: inviting ? 'not-allowed' : 'pointer', opacity: inviting ? 0.7 : 1,
                }}>{inviting ? 'Generating...' : 'Send Invite'}</button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── New Person Modal ──────────────────────────────────────────────── */}
      {showNewPerson && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 200 }}>
          <div style={{ background: '#1e293b', border: '1px solid rgba(99,102,241,0.3)', borderRadius: '16px', padding: '2rem', width: '100%', maxWidth: '400px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h3 style={{ margin: 0 }}>Enrol New Person</h3>
              <button onClick={() => setShowNewPerson(false)} style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}><X size={20} /></button>
            </div>
            {[
              { label: 'Full Name *', value: personName, set: setPersonName, placeholder: 'e.g. John Doe' },
              { label: 'Employee ID', value: personEmpId, set: setPersonEmpId, placeholder: 'e.g. EMP001' },
              { label: 'Department', value: personDept, set: setPersonDept, placeholder: 'e.g. Engineering' },
            ].map(field => (
              <div key={field.label} style={{ marginBottom: '1rem' }}>
                <label style={{ display: 'block', color: '#94a3b8', fontSize: '0.85rem', marginBottom: '0.4rem' }}>{field.label}</label>
                <input type="text" value={field.value} onChange={e => field.set(e.target.value)} placeholder={field.placeholder}
                  style={{ width: '100%', background: 'rgba(30,41,59,0.8)', border: '1px solid rgba(71,85,105,0.5)', borderRadius: '8px', padding: '0.7rem', color: '#f1f5f9', fontSize: '0.9rem', outline: 'none', boxSizing: 'border-box' }} />
              </div>
            ))}
            <button onClick={handleCreatePerson} disabled={creatingPerson} style={{
              width: '100%', background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
              border: 'none', borderRadius: '8px', padding: '0.8rem', marginTop: '0.5rem',
              color: '#fff', fontWeight: 600, cursor: creatingPerson ? 'not-allowed' : 'pointer',
            }}>{creatingPerson ? 'Enrolling...' : 'Enrol Person'}</button>
          </div>
        </div>
      )}
    </div>
  )
}
