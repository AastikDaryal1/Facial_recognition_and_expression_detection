/**
 * src/pages/OrgAdminDashboard.jsx
 * ────────────────────────────────
 * Dashboard for org_admin role (IT Team at Netsmartz).
 * Features:
 *   - Team members list (users in their org)
 *   - Invite users to org
 *   - Persons (face enrolment records) — create, view, manage
 *   - Session history for the org (view, delete)
 *   - Audit log (read-only, own org only)
 */

import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Activity, AlertCircle, Clock, FileText, ImagePlus, LogOut, PlusCircle,
  RefreshCw, ScanFace, Shield, Sparkles, Trash2, UserCheck, UserMinus,
  UserPlus, Users,
} from 'lucide-react'
import { useAuth } from '../AuthContext'
import {
  fetchUsers, inviteUser, deactivateUser, activateUser,
  fetchPersons, createPerson,
  fetchSessions, deleteSession,
  fetchAuditLogs,
} from '../api'

// ── helpers ─────────────────────────────────────────────────────────────────

function SectionHeader({ icon: Icon, title, accent = '#8b5cf6' }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1rem' }}>
      <Icon size={18} color={accent} />
      <h2 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700, color: '#f1f5f9' }}>{title}</h2>
    </div>
  )
}

function StatCard({ icon: Icon, label, value, accent }) {
  return (
    <div style={{
      background: 'rgba(15,23,42,0.6)',
      border: `1px solid ${accent}33`,
      borderRadius: '12px',
      padding: '1.25rem 1.5rem',
      display: 'flex',
      alignItems: 'center',
      gap: '1rem',
    }}>
      <div style={{ width: 44, height: 44, borderRadius: '10px', background: `${accent}22`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
        <Icon size={20} color={accent} />
      </div>
      <div>
        <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '2px' }}>{label}</div>
        <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f1f5f9' }}>{value}</div>
      </div>
    </div>
  )
}

// ── main component ───────────────────────────────────────────────────────────

export default function OrgAdminDashboard() {
  const { user, logout } = useAuth()

  const [tab, setTab] = useState('overview')

  // team (users in own org)
  const [teamMembers, setTeamMembers] = useState([])
  const [inviteEmail, setInviteEmail] = useState('')
  const [inviting, setInviting] = useState(false)
  const [inviteToken, setInviteToken] = useState('')
  const [inviteMsg, setInviteMsg] = useState({ type: '', text: '' })

  // persons
  const [persons, setPersons] = useState([])
  const [newPerson, setNewPerson] = useState({ full_name: '', employee_id: '', department: '' })
  const [creatingPerson, setCreatingPerson] = useState(false)
  const [personMsg, setPersonMsg] = useState({ type: '', text: '' })

  // sessions
  const [sessions, setSessions] = useState([])

  // audit
  const [auditLogs, setAuditLogs] = useState([])

  // ── load data ──────────────────────────────────────────────────────────────

  const loadTeam    = async () => { try { setTeamMembers(await fetchUsers()) } catch { /* silent */ } }
  const loadPersons = async () => { try { setPersons(await fetchPersons()) } catch { /* silent */ } }
  const loadSessions = async () => { try { setSessions(await fetchSessions()) } catch { /* silent */ } }
  const loadAudit   = async () => { try { setAuditLogs(await fetchAuditLogs()) } catch { /* silent */ } }

  useEffect(() => { loadTeam(); loadPersons() }, [])

  useEffect(() => {
    if (tab === 'sessions') loadSessions()
    if (tab === 'audit') loadAudit()
  }, [tab])

  // ── actions ────────────────────────────────────────────────────────────────

  const handleInvite = async () => {
    if (!inviteEmail.trim()) return
    setInviting(true)
    setInviteMsg({ type: '', text: '' })
    setInviteToken('')
    try {
      const res = await inviteUser(inviteEmail.trim(), 'user')
      setInviteToken(res.invite_token)
      setInviteMsg({ type: 'success', text: `Invite token generated for ${inviteEmail}.` })
      setInviteEmail('')
    } catch (e) {
      setInviteMsg({ type: 'error', text: e.message })
    }
    setInviting(false)
  }

  const handleToggleActive = async (u) => {
    try {
      if (u.is_active) await deactivateUser(u.id)
      else await activateUser(u.id)
      loadTeam()
    } catch (e) { alert(`Action failed: ${e.message}`) }
  }

  const handleCreatePerson = async () => {
    if (!newPerson.full_name.trim()) return
    setCreatingPerson(true)
    setPersonMsg({ type: '', text: '' })
    try {
      await createPerson(newPerson)
      setNewPerson({ full_name: '', employee_id: '', department: '' })
      setPersonMsg({ type: 'success', text: 'Person record created.' })
      loadPersons()
    } catch (e) {
      setPersonMsg({ type: 'error', text: e.message })
    }
    setCreatingPerson(false)
  }

  const handleDeleteSession = async (id) => {
    if (!window.confirm('Delete this session record?')) return
    try { await deleteSession(id); loadSessions() }
    catch (e) { alert(`Failed: ${e.message}`) }
  }

  // ── tabs ───────────────────────────────────────────────────────────────────

  const tabs = [
    { id: 'overview',  label: 'Overview',  icon: Activity },
    { id: 'team',      label: 'Team',      icon: Users },
    { id: 'persons',   label: 'Persons',   icon: UserCheck },
    { id: 'sessions',  label: 'Sessions',  icon: FileText },
    { id: 'audit',     label: 'Audit Log', icon: Shield },
  ]

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', color: '#f1f5f9' }}>

      {/* Top bar */}
      <header style={{
        background: 'rgba(15,23,42,0.95)',
        borderBottom: '1px solid #1e293b',
        padding: '0 2rem',
        height: '60px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        backdropFilter: 'blur(12px)',
        position: 'sticky',
        top: 0,
        zIndex: 50,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <Sparkles size={20} color="#8b5cf6" />
          <span style={{ fontWeight: 800, fontSize: '1.05rem', letterSpacing: '-0.02em' }}>VisionX</span>
          <span style={{
            background: '#8b5cf622', color: '#8b5cf6', border: '1px solid #8b5cf655',
            borderRadius: '4px', padding: '2px 8px', fontSize: '0.7rem', fontWeight: 700,
            textTransform: 'uppercase', letterSpacing: '0.08em', marginLeft: '0.5rem',
          }}>Org Admin</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>{user?.email}</span>
          <Link to="/upload" style={{ color: '#8b5cf6', fontSize: '0.85rem', textDecoration: 'none' }}>
            <ImagePlus size={16} style={{ verticalAlign: 'middle', marginRight: '4px' }} />Upload
          </Link>
          <Link to="/live" style={{ color: '#8b5cf6', fontSize: '0.85rem', textDecoration: 'none' }}>
            <ScanFace size={16} style={{ verticalAlign: 'middle', marginRight: '4px' }} />Live
          </Link>
          <button onClick={logout} style={{ background: 'none', border: '1px solid #334155', borderRadius: '6px', color: '#94a3b8', padding: '5px 12px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.83rem' }}>
            <LogOut size={14} /> Logout
          </button>
        </div>
      </header>

      {/* Tab nav */}
      <div style={{ borderBottom: '1px solid #1e293b', background: 'rgba(15,23,42,0.8)', padding: '0 2rem' }}>
        <div style={{ display: 'flex' }}>
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              style={{
                background: 'none', border: 'none',
                borderBottom: tab === id ? '2px solid #8b5cf6' : '2px solid transparent',
                color: tab === id ? '#f1f5f9' : '#64748b',
                padding: '1rem 1.25rem', cursor: 'pointer',
                display: 'flex', alignItems: 'center', gap: '6px',
                fontSize: '0.88rem', fontWeight: tab === id ? 600 : 400,
                transition: 'all 0.15s',
              }}
            >
              <Icon size={15} /> {label}
            </button>
          ))}
        </div>
      </div>

      <main style={{ maxWidth: '1100px', margin: '0 auto', padding: '2rem' }}>

        {/* ── OVERVIEW ── */}
        {tab === 'overview' && (
          <div>
            <div style={{ marginBottom: '1.5rem' }}>
              <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 800 }}>Organisation Dashboard</h1>
              <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: '0.88rem' }}>Manage your team and face enrolments.</p>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
              <StatCard icon={Users}     label="Team Members"       value={teamMembers.filter(m => m.role !== 'super_admin').length} accent="#8b5cf6" />
              <StatCard icon={UserCheck} label="Enrolled Persons"   value={persons.filter(p => p.is_enrolled).length}                accent="#10b981" />
              <StatCard icon={FileText}  label="Total Persons"      value={persons.length}                                           accent="#06b6d4" />
            </div>
            <div style={{ background: 'rgba(139,92,246,0.08)', border: '1px solid #8b5cf633', borderRadius: '12px', padding: '1.5rem' }}>
              <SectionHeader icon={Activity} title="Quick Actions" accent="#8b5cf6" />
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                <button onClick={() => setTab('team')} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#f1f5f9', padding: '0.6rem 1.2rem', cursor: 'pointer', fontSize: '0.88rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <UserPlus size={15} /> Invite Team Member
                </button>
                <button onClick={() => setTab('persons')} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#f1f5f9', padding: '0.6rem 1.2rem', cursor: 'pointer', fontSize: '0.88rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <PlusCircle size={15} /> Add Person Record
                </button>
                <button onClick={() => setTab('sessions')} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#f1f5f9', padding: '0.6rem 1.2rem', cursor: 'pointer', fontSize: '0.88rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <FileText size={15} /> View Sessions
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── TEAM ── */}
        {tab === 'team' && (
          <div>
            <div style={{ marginBottom: '1.5rem' }}>
              <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 800 }}>Team Members</h1>
              <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: '0.88rem' }}>
                {teamMembers.filter(m => m.role !== 'super_admin').length} members in your organisation
              </p>
            </div>

            {/* Invite panel */}
            <div style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid #1e293b', borderRadius: '12px', padding: '1.25rem', marginBottom: '1.5rem' }}>
              <SectionHeader icon={UserPlus} title="Invite Team Member" accent="#10b981" />
              <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                <input
                  type="email"
                  placeholder="Email address…"
                  value={inviteEmail}
                  onChange={(e) => setInviteEmail(e.target.value)}
                  style={{ flex: 1, minWidth: '200px', background: '#0f172a', border: '1px solid #334155', borderRadius: '8px', color: '#f1f5f9', padding: '0.6rem 1rem', fontSize: '0.88rem' }}
                />
                <button onClick={handleInvite} disabled={inviting || !inviteEmail.trim()} style={{ background: '#10b981', border: 'none', borderRadius: '8px', color: '#fff', padding: '0.6rem 1.25rem', cursor: 'pointer', fontWeight: 600, fontSize: '0.88rem', opacity: inviting || !inviteEmail.trim() ? 0.5 : 1 }}>
                  {inviting ? 'Generating…' : 'Send Invite'}
                </button>
              </div>
              {inviteMsg.text && <p style={{ margin: '0.5rem 0 0', fontSize: '0.83rem', color: inviteMsg.type === 'success' ? '#10b981' : '#ff4c4c' }}>{inviteMsg.text}</p>}
              {inviteToken && (
                <div style={{ marginTop: '0.75rem', background: '#0f172a', border: '1px solid #334155', borderRadius: '8px', padding: '0.75rem 1rem' }}>
                  <div style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: '4px' }}>Share this signup link:</div>
                  <code style={{ fontSize: '0.78rem', color: '#a5b4fc', wordBreak: 'break-all' }}>
                    {window.location.origin}/signup?token={inviteToken}
                  </code>
                </div>
              )}
            </div>

            {/* Members table */}
            <div style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid #1e293b', borderRadius: '12px', overflow: 'hidden' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
                <thead>
                  <tr style={{ background: '#1e293b', color: '#94a3b8' }}>
                    <th style={{ textAlign: 'left', padding: '0.75rem 1rem', fontWeight: 600 }}>Email</th>
                    <th style={{ textAlign: 'left', padding: '0.75rem 1rem', fontWeight: 600 }}>Role</th>
                    <th style={{ textAlign: 'left', padding: '0.75rem 1rem', fontWeight: 600 }}>Status</th>
                    <th style={{ textAlign: 'right', padding: '0.75rem 1rem', fontWeight: 600 }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {teamMembers.filter(m => m.role !== 'super_admin').length === 0 && (
                    <tr><td colSpan={4} style={{ padding: '1.5rem', textAlign: 'center', color: '#64748b' }}>No team members yet. Invite someone above.</td></tr>
                  )}
                  {teamMembers.filter(m => m.role !== 'super_admin').map((u, idx) => (
                    <tr key={u.id} style={{ borderTop: '1px solid #1e293b', background: idx % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)' }}>
                      <td style={{ padding: '0.75rem 1rem', color: '#f1f5f9' }}>{u.email}</td>
                      <td style={{ padding: '0.75rem 1rem' }}>
                        <span style={{ fontSize: '0.78rem', color: u.role === 'org_admin' ? '#8b5cf6' : '#10b981' }}>
                          {u.role === 'org_admin' ? 'Org Admin' : 'Member'}
                        </span>
                      </td>
                      <td style={{ padding: '0.75rem 1rem' }}>
                        <span style={{ fontSize: '0.78rem', color: u.is_active ? '#10b981' : '#ef4444' }}>
                          {u.is_active ? '● Active' : '● Inactive'}
                        </span>
                      </td>
                      <td style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>
                        <button
                          onClick={() => handleToggleActive(u)}
                          style={{ background: 'none', border: '1px solid #334155', borderRadius: '6px', color: u.is_active ? '#ef4444' : '#10b981', padding: '4px 10px', cursor: 'pointer', fontSize: '0.78rem', display: 'inline-flex', alignItems: 'center', gap: '4px' }}
                        >
                          {u.is_active ? <><UserMinus size={12} /> Deactivate</> : <><UserCheck size={12} /> Activate</>}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── PERSONS ── */}
        {tab === 'persons' && (
          <div>
            <div style={{ marginBottom: '1.5rem' }}>
              <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 800 }}>Person Records</h1>
              <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: '0.88rem' }}>Face enrolment records for your organisation.</p>
            </div>

            {/* Create person */}
            <div style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid #1e293b', borderRadius: '12px', padding: '1.25rem', marginBottom: '1.5rem' }}>
              <SectionHeader icon={PlusCircle} title="Add Person Record" accent="#06b6d4" />
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem', marginBottom: '0.75rem' }}>
                {[
                  { key: 'full_name',   placeholder: 'Full name *' },
                  { key: 'employee_id', placeholder: 'Employee ID' },
                  { key: 'department',  placeholder: 'Department' },
                ].map(({ key, placeholder }) => (
                  <input
                    key={key}
                    type="text"
                    placeholder={placeholder}
                    value={newPerson[key]}
                    onChange={(e) => setNewPerson(p => ({ ...p, [key]: e.target.value }))}
                    style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px', color: '#f1f5f9', padding: '0.6rem 1rem', fontSize: '0.88rem' }}
                  />
                ))}
              </div>
              <button onClick={handleCreatePerson} disabled={creatingPerson || !newPerson.full_name.trim()} style={{ background: '#06b6d4', border: 'none', borderRadius: '8px', color: '#fff', padding: '0.6rem 1.25rem', cursor: 'pointer', fontWeight: 600, fontSize: '0.88rem', opacity: creatingPerson || !newPerson.full_name.trim() ? 0.5 : 1 }}>
                {creatingPerson ? 'Creating…' : 'Add Person'}
              </button>
              {personMsg.text && <p style={{ margin: '0.5rem 0 0', fontSize: '0.83rem', color: personMsg.type === 'success' ? '#10b981' : '#ff4c4c' }}>{personMsg.text}</p>}
            </div>

            {/* Person list */}
            {persons.length === 0 ? (
              <div style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid #1e293b', borderRadius: '12px', padding: '2rem', textAlign: 'center', color: '#64748b' }}>
                No person records yet. Add one above.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {persons.map((p) => (
                  <div key={p.id} style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid #1e293b', borderRadius: '10px', padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
                    <div>
                      <div style={{ fontWeight: 600, fontSize: '0.95rem', color: '#f1f5f9' }}>{p.full_name}</div>
                      <div style={{ fontSize: '0.78rem', color: '#64748b', marginTop: '2px' }}>
                        {p.employee_id && `ID: ${p.employee_id}`}
                        {p.employee_id && p.department && ' · '}
                        {p.department && `Dept: ${p.department}`}
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                      <span style={{ fontSize: '0.75rem', padding: '3px 8px', borderRadius: '4px', background: p.is_enrolled ? '#10b98122' : '#f59e0b22', color: p.is_enrolled ? '#10b981' : '#f59e0b', border: `1px solid ${p.is_enrolled ? '#10b98155' : '#f59e0b55'}` }}>
                        {p.is_enrolled ? '✓ Enrolled' : '⏳ Pending'}
                      </span>
                      {p.photo_count > 0 && (
                        <span style={{ fontSize: '0.75rem', color: '#64748b' }}>{p.photo_count} photo{p.photo_count !== 1 ? 's' : ''}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── SESSIONS ── */}
        {tab === 'sessions' && (
          <div>
            <div style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 800 }}>Session History</h1>
                <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: '0.88rem' }}>All detection sessions in your organisation.</p>
              </div>
              <button onClick={loadSessions} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#94a3b8', padding: '0.5rem 1rem', cursor: 'pointer', fontSize: '0.83rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <RefreshCw size={13} /> Refresh
              </button>
            </div>

            {sessions.length === 0 ? (
              <div style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid #1e293b', borderRadius: '12px', padding: '2rem', textAlign: 'center', color: '#64748b' }}>
                No sessions yet. Sessions are recorded when users run detections.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {sessions.map((s) => (
                  <div key={s.id} style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid #1e293b', borderRadius: '10px', padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
                    <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                      <div>
                        <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Faces / Identified</div>
                        <div style={{ fontWeight: 700, color: '#f1f5f9' }}>{s.n_faces} / {s.n_identified}</div>
                      </div>
                      {s.elapsed_s && (
                        <div>
                          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Duration</div>
                          <div style={{ fontWeight: 600, color: '#f1f5f9' }}>{s.elapsed_s.toFixed(2)}s</div>
                        </div>
                      )}
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                      <span style={{ fontSize: '0.75rem', color: '#475569' }}>
                        <Clock size={11} style={{ verticalAlign: 'middle', marginRight: '3px' }} />
                        {new Date(s.created_at).toLocaleString()}
                      </span>
                      <button onClick={() => handleDeleteSession(s.id)} style={{ background: 'none', border: '1px solid #334155', borderRadius: '6px', color: '#ef4444', padding: '4px 8px', cursor: 'pointer', fontSize: '0.78rem', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                        <Trash2 size={12} /> Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── AUDIT LOG ── */}
        {tab === 'audit' && (
          <div>
            <div style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 800 }}>Audit Log</h1>
                <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: '0.88rem' }}>Actions recorded for your organisation.</p>
              </div>
              <button onClick={loadAudit} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#94a3b8', padding: '0.5rem 1rem', cursor: 'pointer', fontSize: '0.83rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <RefreshCw size={13} /> Refresh
              </button>
            </div>

            {auditLogs.length === 0 ? (
              <div style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid #1e293b', borderRadius: '12px', padding: '2rem', textAlign: 'center', color: '#64748b' }}>
                No audit log entries yet.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {auditLogs.map((log) => (
                  <div key={log.id} style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid #1e293b', borderRadius: '8px', padding: '0.75rem 1rem', display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
                    <Shield size={14} color="#8b5cf6" style={{ marginTop: '2px', flexShrink: 0 }} />
                    <div style={{ flex: 1 }}>
                      <span style={{ fontWeight: 600, color: '#f1f5f9', fontSize: '0.88rem' }}>{log.action}</span>
                      {log.detail && <span style={{ color: '#64748b', fontSize: '0.83rem', marginLeft: '0.5rem' }}>{log.detail}</span>}
                    </div>
                    <span style={{ fontSize: '0.75rem', color: '#475569', flexShrink: 0 }}>
                      <Clock size={11} style={{ verticalAlign: 'middle', marginRight: '3px' }} />
                      {new Date(log.created_at).toLocaleString()}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

      </main>
    </div>
  )
}
