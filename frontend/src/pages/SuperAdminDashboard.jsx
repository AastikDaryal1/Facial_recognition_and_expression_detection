/**
 * src/pages/SuperAdminDashboard.jsx
 * ───────────────────────────────────
 * Super Admin dashboard — full system management.
 * Tabs: Overview, Organisations, Users, Audit Logs
 */

import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Sparkles, LogOut, LayoutDashboard, Building2, Users,
  ClipboardList, Activity, Zap, Clock, Hash,
  UserCheck, UserX, Trash2, ShieldAlert, RefreshCw,
  Plus, X, Send, ChevronDown
} from 'lucide-react'
import { useAuth } from '../AuthContext'
import {
  fetchMetrics, fetchOrganisations, createOrganisation,
  fetchUsers, deactivateUser, activateUser, changeUserRole,
  deleteUser, inviteUser, fetchAuditLogs,
} from '../api'

const TABS = ['Overview', 'Organisations', 'Users', 'Audit Logs']

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
      borderRadius: '12px', padding: '1.25rem', display: 'flex',
      alignItems: 'center', gap: '1rem',
    }}>
      <div style={{
        background: `${color}20`, borderRadius: '10px',
        padding: '0.75rem', color,
      }}>
        <Icon size={20} />
      </div>
      <div>
        <p style={{ margin: 0, color: '#64748b', fontSize: '0.8rem' }}>{label}</p>
        <p style={{ margin: 0, color: '#f1f5f9', fontSize: '1.4rem', fontWeight: 700 }}>{value}</p>
      </div>
    </div>
  )
}

export default function SuperAdminDashboard() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const [tab, setTab] = useState('Overview')
  const [metrics, setMetrics] = useState(null)
  const [orgs, setOrgs] = useState([])
  const [users, setUsers] = useState([])
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Invite modal
  const [showInvite, setShowInvite] = useState(false)
  const [inviteEmail, setInviteEmail] = useState('')
  const [inviteRole, setInviteRole] = useState('org_admin')
  const [inviteResult, setInviteResult] = useState('')
  const [inviting, setInviting] = useState(false)

  // New org modal
  const [showNewOrg, setShowNewOrg] = useState(false)
  const [newOrgName, setNewOrgName] = useState('')
  const [creatingOrg, setCreatingOrg] = useState(false)

  const load = async () => {
    setLoading(true); setError('')
    try {
      const [m, o, u, l] = await Promise.all([
        fetchMetrics(), fetchOrganisations(), fetchUsers(), fetchAuditLogs()
      ])
      setMetrics(m); setOrgs(o); setUsers(u); setLogs(l)
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

  const handleCreateOrg = async () => {
    if (!newOrgName) return
    setCreatingOrg(true)
    try {
      await createOrganisation(newOrgName)
      setNewOrgName(''); setShowNewOrg(false)
      load()
    } catch (e) { setError(e.message) }
    setCreatingOrg(false)
  }

  const handleDeactivate = async (id) => {
    try { await deactivateUser(id); load() } catch (e) { setError(e.message) }
  }

  const handleActivate = async (id) => {
    try { await activateUser(id); load() } catch (e) { setError(e.message) }
  }

  const handleDelete = async (id) => {
    if (!window.confirm('Permanently delete this user?')) return
    try { await deleteUser(id); load() } catch (e) { setError(e.message) }
  }

  const handleRoleChange = async (id, role) => {
    try { await changeUserRole(id, role); load() } catch (e) { setError(e.message) }
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
          <span style={{ color: '#64748b', fontSize: '0.85rem', marginLeft: '0.5rem' }}>Super Admin</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span style={{ color: '#64748b', fontSize: '0.85rem' }}>{user?.email}</span>
          <Badge role="super_admin" />
          <button onClick={logout} style={{
            background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
            borderRadius: '8px', padding: '0.4rem 0.8rem', color: '#f87171',
            cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem',
          }}>
            <LogOut size={14} /> Log Out
          </button>
        </div>
      </header>

      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>

        {/* Page title */}
        <div style={{ marginBottom: '2rem' }}>
          <h1 style={{ margin: 0, fontSize: '1.8rem', fontWeight: 700 }}>System Dashboard</h1>
          <p style={{ margin: '0.25rem 0 0', color: '#64748b' }}>Full system overview and management</p>
        </div>

        {/* Error */}
        {error && (
          <div style={{
            background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
            borderRadius: '8px', padding: '0.75rem 1rem', marginBottom: '1.5rem', color: '#fca5a5',
          }}>{error}</div>
        )}

        {/* Tabs */}
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '2rem', borderBottom: '1px solid rgba(71,85,105,0.3)', paddingBottom: '0' }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              background: 'none', border: 'none', cursor: 'pointer',
              padding: '0.6rem 1.2rem', fontSize: '0.9rem', fontWeight: 600,
              color: tab === t ? '#6366f1' : '#64748b',
              borderBottom: tab === t ? '2px solid #6366f1' : '2px solid transparent',
              marginBottom: '-1px', transition: 'color 0.2s',
            }}>{t}</button>
          ))}
          <button onClick={load} style={{
            marginLeft: 'auto', background: 'none', border: 'none',
            color: '#64748b', cursor: 'pointer', padding: '0.6rem',
          }}>
            <RefreshCw size={16} className={loading ? 'spinning' : ''} />
          </button>
        </div>

        {/* ── Overview Tab ─────────────────────────────────────────────── */}
        {tab === 'Overview' && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
              <StatCard icon={Activity} label="Request Count" value={metrics?.request_count ?? '—'} color="#6366f1" />
              <StatCard icon={Zap} label="Avg Latency" value={metrics ? `${metrics.avg_latency_s}s` : '—'} color="#10b981" />
              <StatCard icon={Clock} label="Uptime" value={metrics ? `${Math.round(metrics.uptime_s)}s` : '—'} color="#f59e0b" />
              <StatCard icon={Building2} label="Organisations" value={orgs.length} color="#8b5cf6" />
              <StatCard icon={Users} label="Total Users" value={users.length} color="#06b6d4" />
              <StatCard icon={ClipboardList} label="Audit Entries" value={logs.length} color="#f43f5e" />
            </div>

            {/* Quick actions */}
            <div style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(99,102,241,0.2)', borderRadius: '12px', padding: '1.5rem' }}>
              <h3 style={{ margin: '0 0 1rem', fontSize: '1rem' }}>Quick Actions</h3>
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                <button onClick={() => setShowInvite(true)} style={{
                  background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                  border: 'none', borderRadius: '8px', padding: '0.7rem 1.2rem',
                  color: '#fff', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600,
                }}>
                  <Send size={16} /> Invite User
                </button>
                <button onClick={() => setShowNewOrg(true)} style={{
                  background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.3)',
                  borderRadius: '8px', padding: '0.7rem 1.2rem',
                  color: '#a5b4fc', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600,
                }}>
                  <Plus size={16} /> New Organisation
                </button>
                <button onClick={() => navigate('/upload')} style={{
                  background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)',
                  borderRadius: '8px', padding: '0.7rem 1.2rem',
                  color: '#34d399', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600,
                }}>
                  <LayoutDashboard size={16} /> Go to Detection
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── Organisations Tab ─────────────────────────────────────────── */}
        {tab === 'Organisations' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0 }}>All Organisations ({orgs.length})</h3>
              <button onClick={() => setShowNewOrg(true)} style={{
                background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                border: 'none', borderRadius: '8px', padding: '0.6rem 1rem',
                color: '#fff', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.875rem', fontWeight: 600,
              }}>
                <Plus size={14} /> New Organisation
              </button>
            </div>
            <div style={{ display: 'grid', gap: '0.75rem' }}>
              {orgs.map(org => (
                <div key={org.id} style={{
                  background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(71,85,105,0.3)',
                  borderRadius: '10px', padding: '1rem 1.25rem',
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                }}>
                  <div>
                    <p style={{ margin: 0, fontWeight: 600, fontSize: '0.95rem' }}>{org.name}</p>
                    <p style={{ margin: '0.2rem 0 0', color: '#64748b', fontSize: '0.78rem' }}>ID: {org.id}</p>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <span style={{
                      background: org.is_active ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)',
                      color: org.is_active ? '#34d399' : '#f87171',
                      border: `1px solid ${org.is_active ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)'}`,
                      borderRadius: '20px', padding: '0.2rem 0.7rem', fontSize: '0.75rem', fontWeight: 600,
                    }}>
                      {org.is_active ? 'Active' : 'Inactive'}
                    </span>
                    <span style={{ color: '#64748b', fontSize: '0.8rem' }}>
                      {users.filter(u => u.org_id === org.id).length} users
                    </span>
                  </div>
                </div>
              ))}
              {orgs.length === 0 && <p style={{ color: '#64748b' }}>No organisations yet.</p>}
            </div>
          </div>
        )}

        {/* ── Users Tab ─────────────────────────────────────────────────── */}
        {tab === 'Users' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0 }}>All Users ({users.length})</h3>
              <button onClick={() => setShowInvite(true)} style={{
                background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                border: 'none', borderRadius: '8px', padding: '0.6rem 1rem',
                color: '#fff', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.875rem', fontWeight: 600,
              }}>
                <Send size={14} /> Invite User
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
                  {/* Role change */}
                  {u.role !== 'super_admin' && u.email !== user?.email && (
                    <select
                      value={u.role}
                      onChange={(e) => handleRoleChange(u.id, e.target.value)}
                      style={{
                        background: 'rgba(30,41,59,0.8)', border: '1px solid rgba(71,85,105,0.5)',
                        borderRadius: '6px', padding: '0.3rem 0.5rem', color: '#f1f5f9', fontSize: '0.8rem', cursor: 'pointer',
                      }}
                    >
                      <option value="org_admin">Org Admin</option>
                      <option value="user">User</option>
                    </select>
                  )}
                  {/* Actions */}
                  {u.email !== user?.email && (
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      {u.is_active
                        ? <button onClick={() => handleDeactivate(u.id)} title="Deactivate" style={{ background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.3)', borderRadius: '6px', padding: '0.3rem 0.5rem', color: '#fbbf24', cursor: 'pointer' }}>
                          <UserX size={14} />
                        </button>
                        : <button onClick={() => handleActivate(u.id)} title="Activate" style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)', borderRadius: '6px', padding: '0.3rem 0.5rem', color: '#34d399', cursor: 'pointer' }}>
                          <UserCheck size={14} />
                        </button>
                      }
                      <button onClick={() => handleDelete(u.id)} title="Delete" style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '6px', padding: '0.3rem 0.5rem', color: '#f87171', cursor: 'pointer' }}>
                        <Trash2 size={14} />
                      </button>
                    </div>
                  )}
                </div>
              ))}
              {users.length === 0 && <p style={{ color: '#64748b' }}>No users yet.</p>}
            </div>
          </div>
        )}

        {/* ── Audit Logs Tab ────────────────────────────────────────────── */}
        {tab === 'Audit Logs' && (
          <div>
            <h3 style={{ margin: '0 0 1rem' }}>Audit Logs ({logs.length})</h3>
            {logs.length === 0
              ? <p style={{ color: '#64748b' }}>No audit logs yet. Actions like invites, deactivations, and deletions will appear here.</p>
              : (
                <div style={{ display: 'grid', gap: '0.5rem' }}>
                  {logs.map((log, idx) => (
                    <div key={idx} style={{
                      background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(71,85,105,0.3)',
                      borderRadius: '8px', padding: '0.75rem 1rem',
                      display: 'flex', alignItems: 'center', gap: '1rem',
                    }}>
                      <ShieldAlert size={16} color="#f59e0b" style={{ flexShrink: 0 }} />
                      <div style={{ flex: 1 }}>
                        <p style={{ margin: 0, fontSize: '0.875rem', fontWeight: 600 }}>{log.action}</p>
                        <p style={{ margin: '0.2rem 0 0', color: '#64748b', fontSize: '0.78rem' }}>
                          {log.actor_email || log.actor_id} → {log.target_type} {log.target_id}
                        </p>
                      </div>
                      <span style={{ color: '#475569', fontSize: '0.78rem', flexShrink: 0 }}>
                        {log.created_at ? new Date(log.created_at).toLocaleString() : ''}
                      </span>
                    </div>
                  ))}
                </div>
              )
            }
          </div>
        )}
      </div>

      {/* ── Invite Modal ──────────────────────────────────────────────────── */}
      {showInvite && (
        <div style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 200,
        }}>
          <div style={{
            background: '#1e293b', border: '1px solid rgba(99,102,241,0.3)',
            borderRadius: '16px', padding: '2rem', width: '100%', maxWidth: '420px',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h3 style={{ margin: 0 }}>Invite User</h3>
              <button onClick={() => { setShowInvite(false); setInviteResult('') }}
                style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}>
                <X size={20} />
              </button>
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
                  onClick={() => { setShowInvite(false); setInviteResult('') }}
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
                <div style={{ marginBottom: '1rem' }}>
                  <label style={{ display: 'block', color: '#94a3b8', fontSize: '0.85rem', marginBottom: '0.4rem' }}>Email</label>
                  <input type="email" value={inviteEmail} onChange={e => setInviteEmail(e.target.value)}
                    placeholder="user@example.com"
                    style={{ width: '100%', background: 'rgba(30,41,59,0.8)', border: '1px solid rgba(71,85,105,0.5)', borderRadius: '8px', padding: '0.7rem', color: '#f1f5f9', fontSize: '0.9rem', outline: 'none', boxSizing: 'border-box' }} />
                </div>
                <div style={{ marginBottom: '1.5rem' }}>
                  <label style={{ display: 'block', color: '#94a3b8', fontSize: '0.85rem', marginBottom: '0.4rem' }}>Role</label>
                  <select value={inviteRole} onChange={e => setInviteRole(e.target.value)}
                    style={{ width: '100%', background: 'rgba(30,41,59,0.8)', border: '1px solid rgba(71,85,105,0.5)', borderRadius: '8px', padding: '0.7rem', color: '#f1f5f9', fontSize: '0.9rem', outline: 'none', boxSizing: 'border-box' }}>
                    <option value="org_admin">Org Admin</option>
                    <option value="user">User</option>
                  </select>
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
                  color: '#fff', fontWeight: 600, cursor: inviting ? 'not-allowed' : 'pointer',
                  opacity: inviting ? 0.7 : 1,
                }}>
                  {inviting ? 'Generating...' : 'Send Invite'}
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── New Org Modal ─────────────────────────────────────────────────── */}
      {showNewOrg && (
        <div style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 200,
        }}>
          <div style={{
            background: '#1e293b', border: '1px solid rgba(99,102,241,0.3)',
            borderRadius: '16px', padding: '2rem', width: '100%', maxWidth: '380px',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h3 style={{ margin: 0 }}>New Organisation</h3>
              <button onClick={() => setShowNewOrg(false)} style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}>
                <X size={20} />
              </button>
            </div>
            <div style={{ marginBottom: '1.5rem' }}>
              <label style={{ display: 'block', color: '#94a3b8', fontSize: '0.85rem', marginBottom: '0.4rem' }}>Organisation Name</label>
              <input type="text" value={newOrgName} onChange={e => setNewOrgName(e.target.value)}
                placeholder="e.g. IT Team"
                style={{ width: '100%', background: 'rgba(30,41,59,0.8)', border: '1px solid rgba(71,85,105,0.5)', borderRadius: '8px', padding: '0.7rem', color: '#f1f5f9', fontSize: '0.9rem', outline: 'none', boxSizing: 'border-box' }} />
            </div>
            <button onClick={handleCreateOrg} disabled={creatingOrg} style={{
              width: '100%', background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
              border: 'none', borderRadius: '8px', padding: '0.8rem',
              color: '#fff', fontWeight: 600, cursor: creatingOrg ? 'not-allowed' : 'pointer',
            }}>
              {creatingOrg ? 'Creating...' : 'Create Organisation'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
