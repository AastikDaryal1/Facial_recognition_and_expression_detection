/**
 * src/pages/SuperAdminDashboard.jsx
 * ────────────────────────────────────
 * Dashboard for super_admin role (Netsmartz / company level).
 * Features:
 *   - System metrics (requests, latency, uptime)
 *   - All organisations list + create new org
 *   - All users across all orgs + invite org_admin + change role + deactivate
 *   - Audit log viewer
 */

import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Activity, AlertCircle, Building2, CheckCircle2, ChevronDown, Clock,
  Cpu, ImagePlus, LogOut, PlusCircle, RefreshCw, ScanFace, Shield,
  Sparkles, Trash2, UserCheck, UserMinus, UserPlus, Users, Zap,
} from 'lucide-react'
import { useAuth } from '../AuthContext'
import {
  fetchMetrics, fetchOrganisations, createOrganisation,
  fetchUsers, inviteUser, changeUserRole, deactivateUser, activateUser,
  fetchAuditLogs,
} from '../api'

// ── helpers ─────────────────────────────────────────────────────────────────

const roleColors = {
  super_admin: '#f59e0b',
  org_admin:   '#8b5cf6',
  user:        '#10b981',
}

const roleBadge = (role) => (
  <span style={{
    background: `${roleColors[role]}22`,
    color: roleColors[role],
    border: `1px solid ${roleColors[role]}55`,
    borderRadius: '4px',
    padding: '2px 8px',
    fontSize: '0.75rem',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  }}>
    {role.replace('_', ' ')}
  </span>
)

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
      boxShadow: `0 0 20px ${accent}11`,
    }}>
      <div style={{
        width: 44, height: 44, borderRadius: '10px',
        background: `${accent}22`, display: 'flex',
        alignItems: 'center', justifyContent: 'center', flexShrink: 0,
      }}>
        <Icon size={20} color={accent} />
      </div>
      <div>
        <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '2px' }}>{label}</div>
        <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f1f5f9' }}>{value}</div>
      </div>
    </div>
  )
}

function SectionHeader({ icon: Icon, title, accent = '#6366f1' }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1rem' }}>
      <Icon size={18} color={accent} />
      <h2 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700, color: '#f1f5f9' }}>{title}</h2>
    </div>
  )
}

// ── main component ───────────────────────────────────────────────────────────

export default function SuperAdminDashboard() {
  const { user, logout } = useAuth()

  // metrics
  const [metrics, setMetrics] = useState(null)
  const [metricsError, setMetricsError] = useState('')

  // organisations
  const [orgs, setOrgs] = useState([])
  const [newOrgName, setNewOrgName] = useState('')
  const [creatingOrg, setCreatingOrg] = useState(false)
  const [orgMsg, setOrgMsg] = useState({ type: '', text: '' })

  // users
  const [users, setUsers] = useState([])
  const [inviteEmail, setInviteEmail] = useState('')
  const [inviteRole, setInviteRole] = useState('org_admin')
  const [inviting, setInviting] = useState(false)
  const [inviteToken, setInviteToken] = useState('')
  const [inviteMsg, setInviteMsg] = useState({ type: '', text: '' })

  // audit
  const [auditLogs, setAuditLogs] = useState([])
  const [auditPage, setAuditPage] = useState(1)

  // active tab
  const [tab, setTab] = useState('overview')

  // ── load data ──────────────────────────────────────────────────────────────

  const loadMetrics = async () => {
    try {
      const data = await fetchMetrics()
      setMetrics(data)
    } catch (e) {
      setMetricsError(e.message)
    }
  }

  const loadOrgs = async () => {
    try { setOrgs(await fetchOrganisations()) } catch { /* silent */ }
  }

  const loadUsers = async () => {
    try { setUsers(await fetchUsers()) } catch { /* silent */ }
  }

  const loadAudit = async () => {
    try { setAuditLogs(await fetchAuditLogs(auditPage)) } catch { /* silent */ }
  }

  useEffect(() => {
    loadMetrics()
    loadOrgs()
    loadUsers()
  }, [])

  useEffect(() => { if (tab === 'audit') loadAudit() }, [tab, auditPage])

  // ── actions ────────────────────────────────────────────────────────────────

  const handleCreateOrg = async () => {
    if (!newOrgName.trim()) return
    setCreatingOrg(true)
    setOrgMsg({ type: '', text: '' })
    try {
      await createOrganisation(newOrgName.trim())
      setNewOrgName('')
      setOrgMsg({ type: 'success', text: 'Organisation created successfully.' })
      loadOrgs()
    } catch (e) {
      setOrgMsg({ type: 'error', text: e.message })
    }
    setCreatingOrg(false)
  }

  const handleInvite = async () => {
    if (!inviteEmail.trim()) return
    setInviting(true)
    setInviteMsg({ type: '', text: '' })
    setInviteToken('')
    try {
      const res = await inviteUser(inviteEmail.trim(), inviteRole)
      setInviteToken(res.invite_token)
      setInviteMsg({ type: 'success', text: `Invite token generated for ${inviteEmail}. Copy and share it.` })
      setInviteEmail('')
    } catch (e) {
      setInviteMsg({ type: 'error', text: e.message })
    }
    setInviting(false)
  }

  const handleRoleChange = async (userId, newRole) => {
    try {
      await changeUserRole(userId, newRole)
      loadUsers()
    } catch (e) {
      alert(`Role change failed: ${e.message}`)
    }
  }

  const handleToggleActive = async (u) => {
    try {
      if (u.is_active) await deactivateUser(u.id)
      else await activateUser(u.id)
      loadUsers()
    } catch (e) {
      alert(`Action failed: ${e.message}`)
    }
  }

  // ── UI ─────────────────────────────────────────────────────────────────────

  const tabs = [
    { id: 'overview',       label: 'Overview',       icon: Activity },
    { id: 'organisations',  label: 'Organisations',  icon: Building2 },
    { id: 'users',          label: 'Users',          icon: Users },
    { id: 'audit',          label: 'Audit Log',      icon: Shield },
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
          <Sparkles size={20} color="#f59e0b" />
          <span style={{ fontWeight: 800, fontSize: '1.05rem', letterSpacing: '-0.02em' }}>VisionX</span>
          <span style={{
            background: '#f59e0b22', color: '#f59e0b', border: '1px solid #f59e0b55',
            borderRadius: '4px', padding: '2px 8px', fontSize: '0.7rem', fontWeight: 700,
            textTransform: 'uppercase', letterSpacing: '0.08em', marginLeft: '0.5rem',
          }}>Super Admin</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>{user?.email}</span>
          <Link to="/upload" style={{ color: '#6366f1', fontSize: '0.85rem', textDecoration: 'none' }}>
            <ImagePlus size={16} style={{ verticalAlign: 'middle', marginRight: '4px' }} />Upload
          </Link>
          <Link to="/live" style={{ color: '#6366f1', fontSize: '0.85rem', textDecoration: 'none' }}>
            <ScanFace size={16} style={{ verticalAlign: 'middle', marginRight: '4px' }} />Live
          </Link>
          <button onClick={logout} style={{ background: 'none', border: '1px solid #334155', borderRadius: '6px', color: '#94a3b8', padding: '5px 12px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.83rem' }}>
            <LogOut size={14} /> Logout
          </button>
        </div>
      </header>

      {/* Tab nav */}
      <div style={{ borderBottom: '1px solid #1e293b', background: 'rgba(15,23,42,0.8)', padding: '0 2rem' }}>
        <div style={{ display: 'flex', gap: '0' }}>
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              style={{
                background: 'none',
                border: 'none',
                borderBottom: tab === id ? '2px solid #6366f1' : '2px solid transparent',
                color: tab === id ? '#f1f5f9' : '#64748b',
                padding: '1rem 1.25rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '0.88rem',
                fontWeight: tab === id ? 600 : 400,
                transition: 'all 0.15s',
              }}
            >
              <Icon size={15} /> {label}
            </button>
          ))}
        </div>
      </div>

      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>

        {/* ── OVERVIEW ── */}
        {tab === 'overview' && (
          <div>
            <div style={{ marginBottom: '1.5rem' }}>
              <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 800 }}>System Overview</h1>
              <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: '0.88rem' }}>
                Real-time metrics and system health.
              </p>
            </div>

            {metricsError && (
              <div style={{ background: '#ff4c4c22', border: '1px solid #ff4c4c55', borderRadius: '8px', padding: '0.75rem 1rem', marginBottom: '1.5rem', color: '#ff4c4c', fontSize: '0.88rem' }}>
                <AlertCircle size={14} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
                {metricsError}
              </div>
            )}

            {metrics && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
                <StatCard icon={Activity}  label="Total Requests"   value={metrics.request_count ?? 0}                   accent="#6366f1" />
                <StatCard icon={Zap}       label="Avg. Latency"     value={`${(metrics.avg_latency_s ?? 0).toFixed(3)}s`} accent="#10b981" />
                <StatCard icon={Cpu}       label="Uptime"           value={`${Math.floor((metrics.uptime_s ?? 0) / 60)}m`} accent="#f59e0b" />
                <StatCard icon={Users}     label="Organisations"    value={orgs.length}                                  accent="#8b5cf6" />
                <StatCard icon={UserCheck} label="Total Users"      value={users.length}                                 accent="#06b6d4" />
              </div>
            )}

            <div style={{ background: 'rgba(99,102,241,0.08)', border: '1px solid #6366f133', borderRadius: '12px', padding: '1.5rem' }}>
              <SectionHeader icon={Activity} title="Quick Actions" accent="#6366f1" />
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                <button onClick={() => setTab('organisations')} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#f1f5f9', padding: '0.6rem 1.2rem', cursor: 'pointer', fontSize: '0.88rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Building2 size={15} /> Manage Organisations
                </button>
                <button onClick={() => setTab('users')} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#f1f5f9', padding: '0.6rem 1.2rem', cursor: 'pointer', fontSize: '0.88rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Users size={15} /> Manage Users
                </button>
                <button onClick={loadMetrics} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#64748b', padding: '0.6rem 1.2rem', cursor: 'pointer', fontSize: '0.88rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <RefreshCw size={14} /> Refresh Metrics
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── ORGANISATIONS ── */}
        {tab === 'organisations' && (
          <div>
            <div style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 800 }}>Organisations</h1>
                <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: '0.88rem' }}>{orgs.length} total</p>
              </div>
            </div>

            {/* Create org */}
            <div style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid #1e293b', borderRadius: '12px', padding: '1.25rem', marginBottom: '1.5rem' }}>
              <SectionHeader icon={PlusCircle} title="Create Organisation" accent="#10b981" />
              <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                <input
                  type="text"
                  placeholder="Organisation name…"
                  value={newOrgName}
                  onChange={(e) => setNewOrgName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleCreateOrg()}
                  style={{ flex: 1, minWidth: '200px', background: '#0f172a', border: '1px solid #334155', borderRadius: '8px', color: '#f1f5f9', padding: '0.6rem 1rem', fontSize: '0.88rem' }}
                />
                <button onClick={handleCreateOrg} disabled={creatingOrg || !newOrgName.trim()} style={{ background: '#10b981', border: 'none', borderRadius: '8px', color: '#fff', padding: '0.6rem 1.25rem', cursor: 'pointer', fontWeight: 600, fontSize: '0.88rem', opacity: creatingOrg || !newOrgName.trim() ? 0.5 : 1 }}>
                  {creatingOrg ? 'Creating…' : 'Create'}
                </button>
              </div>
              {orgMsg.text && (
                <p style={{ margin: '0.5rem 0 0', fontSize: '0.83rem', color: orgMsg.type === 'success' ? '#10b981' : '#ff4c4c' }}>
                  {orgMsg.text}
                </p>
              )}
            </div>

            {/* Org list */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {orgs.length === 0 && <p style={{ color: '#64748b' }}>No organisations yet.</p>}
              {orgs.map((org) => (
                <div key={org.id} style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid #1e293b', borderRadius: '10px', padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <Building2 size={18} color="#8b5cf6" />
                    <div>
                      <div style={{ fontWeight: 600, fontSize: '0.95rem' }}>{org.name}</div>
                      <div style={{ fontSize: '0.75rem', color: '#64748b' }}>ID: {org.id.slice(0, 8)}…</div>
                    </div>
                  </div>
                  <span style={{ fontSize: '0.75rem', color: org.is_active ? '#10b981' : '#ef4444' }}>
                    {org.is_active ? <><CheckCircle2 size={12} style={{ verticalAlign: 'middle', marginRight: '4px' }} />Active</> : 'Inactive'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── USERS ── */}
        {tab === 'users' && (
          <div>
            <div style={{ marginBottom: '1.5rem' }}>
              <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 800 }}>Users</h1>
              <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: '0.88rem' }}>{users.length} total across all organisations</p>
            </div>

            {/* Invite panel */}
            <div style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid #1e293b', borderRadius: '12px', padding: '1.25rem', marginBottom: '1.5rem' }}>
              <SectionHeader icon={UserPlus} title="Invite User" accent="#8b5cf6" />
              <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                <input
                  type="email"
                  placeholder="Email address…"
                  value={inviteEmail}
                  onChange={(e) => setInviteEmail(e.target.value)}
                  style={{ flex: 1, minWidth: '200px', background: '#0f172a', border: '1px solid #334155', borderRadius: '8px', color: '#f1f5f9', padding: '0.6rem 1rem', fontSize: '0.88rem' }}
                />
                <select
                  value={inviteRole}
                  onChange={(e) => setInviteRole(e.target.value)}
                  style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px', color: '#f1f5f9', padding: '0.6rem 1rem', fontSize: '0.88rem', cursor: 'pointer' }}
                >
                  <option value="org_admin">Org Admin</option>
                  <option value="user">User</option>
                </select>
                <button onClick={handleInvite} disabled={inviting || !inviteEmail.trim()} style={{ background: '#8b5cf6', border: 'none', borderRadius: '8px', color: '#fff', padding: '0.6rem 1.25rem', cursor: 'pointer', fontWeight: 600, fontSize: '0.88rem', opacity: inviting || !inviteEmail.trim() ? 0.5 : 1 }}>
                  {inviting ? 'Generating…' : 'Send Invite'}
                </button>
              </div>
              {inviteMsg.text && (
                <p style={{ margin: '0.5rem 0 0', fontSize: '0.83rem', color: inviteMsg.type === 'success' ? '#10b981' : '#ff4c4c' }}>
                  {inviteMsg.text}
                </p>
              )}
              {inviteToken && (
                <div style={{ marginTop: '0.75rem', background: '#0f172a', border: '1px solid #334155', borderRadius: '8px', padding: '0.75rem 1rem' }}>
                  <div style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: '4px' }}>Invite Token (share this link):</div>
                  <code style={{ fontSize: '0.78rem', color: '#a5b4fc', wordBreak: 'break-all' }}>
                    {window.location.origin}/signup?token={inviteToken}
                  </code>
                </div>
              )}
            </div>

            {/* Users table */}
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
                  {users.length === 0 && (
                    <tr><td colSpan={4} style={{ padding: '1.5rem', textAlign: 'center', color: '#64748b' }}>No users found.</td></tr>
                  )}
                  {users.map((u, idx) => (
                    <tr key={u.id} style={{ borderTop: '1px solid #1e293b', background: idx % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)' }}>
                      <td style={{ padding: '0.75rem 1rem', color: '#f1f5f9' }}>{u.email}</td>
                      <td style={{ padding: '0.75rem 1rem' }}>
                        {u.role === 'super_admin' ? roleBadge(u.role) : (
                          <select
                            value={u.role}
                            onChange={(e) => handleRoleChange(u.id, e.target.value)}
                            style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '6px', color: '#f1f5f9', padding: '3px 8px', fontSize: '0.8rem', cursor: 'pointer' }}
                          >
                            <option value="org_admin">Org Admin</option>
                            <option value="user">User</option>
                          </select>
                        )}
                      </td>
                      <td style={{ padding: '0.75rem 1rem' }}>
                        <span style={{ fontSize: '0.78rem', color: u.is_active ? '#10b981' : '#ef4444' }}>
                          {u.is_active ? '● Active' : '● Inactive'}
                        </span>
                      </td>
                      <td style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>
                        {u.role !== 'super_admin' && (
                          <button
                            onClick={() => handleToggleActive(u)}
                            title={u.is_active ? 'Deactivate' : 'Activate'}
                            style={{ background: 'none', border: '1px solid #334155', borderRadius: '6px', color: u.is_active ? '#ef4444' : '#10b981', padding: '4px 10px', cursor: 'pointer', fontSize: '0.78rem', display: 'inline-flex', alignItems: 'center', gap: '4px' }}
                          >
                            {u.is_active ? <><UserMinus size={12} /> Deactivate</> : <><UserCheck size={12} /> Activate</>}
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── AUDIT LOG ── */}
        {tab === 'audit' && (
          <div>
            <div style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 800 }}>Audit Log</h1>
                <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: '0.88rem' }}>All system actions recorded here.</p>
              </div>
              <button onClick={loadAudit} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#94a3b8', padding: '0.5rem 1rem', cursor: 'pointer', fontSize: '0.83rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <RefreshCw size={13} /> Refresh
              </button>
            </div>

            {auditLogs.length === 0 ? (
              <div style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid #1e293b', borderRadius: '12px', padding: '2rem', textAlign: 'center', color: '#64748b' }}>
                No audit log entries yet. Actions like inviting users, changing roles, and deactivating accounts will appear here.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {auditLogs.map((log) => (
                  <div key={log.id} style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid #1e293b', borderRadius: '8px', padding: '0.75rem 1rem', display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
                    <Shield size={14} color="#6366f1" style={{ marginTop: '2px', flexShrink: 0 }} />
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
