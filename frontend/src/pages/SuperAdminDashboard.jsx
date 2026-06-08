/**
 * src/pages/SuperAdminDashboard.jsx
 * ───────────────────────────────────
 * Super Admin dashboard — full system management.
 * Tabs: Overview (KPI), Organisations, People, Sessions, Audit Logs
 */

import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Sparkles, LogOut, LayoutDashboard, Building2, Users,
  ClipboardList, Activity, Zap, Clock, Hash,
  UserCheck, UserX, Trash2, ShieldAlert, RefreshCw,
  Plus, X, Send, ChevronDown, ScanFace, ChevronUp, TrendingUp
} from 'lucide-react'
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import { useAuth } from '../AuthContext'
import {
  fetchMetrics, fetchOrganisations, createOrganisation, deleteOrganisation,
  fetchUsers, deactivateUser, activateUser, changeUserRole,
  deleteUser, inviteUser, fetchAuditLogs, fetchSessions, triggerSync,
} from '../api'

const TABS = ['Overview', 'Organisations', 'People', 'Sessions', 'Audit Logs']

const ROLE_BADGE = {
  super_admin : { label: 'Super Admin', color: '#f59e0b', bg: 'rgba(245,158,11,0.1)'  },
  org_admin   : { label: 'Org Admin',   color: '#6366f1', bg: 'rgba(99,102,241,0.1)'  },
  member      : { label: 'Member',      color: '#10b981', bg: 'rgba(16,185,129,0.1)'  },
}

const CHART_COLORS = ['#6366f1','#10b981','#f59e0b','#f43f5e','#06b6d4','#8b5cf6']

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

function StatCard({ icon: Icon, label, value, color = '#6366f1', sub }) {
  return (
    <div style={{
      background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(99,102,241,0.2)',
      borderRadius: '12px', padding: '1.25rem', display: 'flex',
      alignItems: 'center', gap: '1rem',
    }}>
      <div style={{ background: `${color}20`, borderRadius: '10px', padding: '0.75rem', color }}>
        <Icon size={20} />
      </div>
      <div>
        <p style={{ margin: 0, color: '#64748b', fontSize: '0.8rem' }}>{label}</p>
        <p style={{ margin: 0, color: '#f1f5f9', fontSize: '1.4rem', fontWeight: 700 }}>{value}</p>
        {sub && <p style={{ margin: 0, color: '#475569', fontSize: '0.72rem' }}>{sub}</p>}
      </div>
    </div>
  )
}

function ChartCard({ title, children }) {
  return (
    <div style={{
      background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(99,102,241,0.2)',
      borderRadius: '12px', padding: '1.25rem',
    }}>
      <p style={{ margin: '0 0 1rem', fontWeight: 600, fontSize: '0.9rem', color: '#94a3b8' }}>{title}</p>
      {children}
    </div>
  )
}

// Build sessions-per-day data for last 7 days
function buildSessionTrend(sessions) {
  const days = []
  for (let i = 6; i >= 0; i--) {
    const d = new Date()
    d.setDate(d.getDate() - i)
    const label = d.toLocaleDateString('en', { weekday: 'short' })
    const dateStr = d.toISOString().slice(0, 10)
    const count = sessions.filter(s => s.created_at?.slice(0, 10) === dateStr).length
    days.push({ day: label, sessions: count })
  }
  return days
}

// Build users-per-org data
function buildUsersPerOrg(orgs, users) {
  return orgs.map(o => ({
    name: o.name.length > 12 ? o.name.slice(0, 12) + '…' : o.name,
    members: users.filter(u => u.org_id === o.id && u.role !== 'super_admin').length,
  }))
}

// Build emotion distribution from sessions
function buildEmotionDist(sessions) {
  const counts = {}
  sessions.forEach(s => {
    const results = s.results_json?.results || (Array.isArray(s.results_json) ? s.results_json : [])
    results.forEach(f => {
      if (f.emotion) counts[f.emotion] = (counts[f.emotion] || 0) + 1
    })
  })
  return Object.entries(counts).map(([name, value]) => ({ name, value }))
}

// Build active vs inactive users
function buildUserStatus(users) {
  const active   = users.filter(u => u.is_active).length
  const inactive = users.filter(u => !u.is_active).length
  return [
    { name: 'Active',   value: active   },
    { name: 'Inactive', value: inactive },
  ]
}

export default function SuperAdminDashboard() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const [tab,      setTab]      = useState('Overview')
  const [metrics,  setMetrics]  = useState(null)
  const [orgs,     setOrgs]     = useState([])
  const [users,    setUsers]    = useState([])
  const [logs,     setLogs]     = useState([])
  const [sessions, setSessions] = useState([])
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState('')

  const [expandedSession,  setExpandedSession]  = useState(null)
  const [sessionOrgFilter, setSessionOrgFilter] = useState('all')

  // Invite modal
  const [showInvite,  setShowInvite]  = useState(false)
  const [inviteEmail, setInviteEmail] = useState('')
  const [inviteRole,  setInviteRole]  = useState('org_admin')
  const [inviteOrgId, setInviteOrgId] = useState('')
  const [inviteResult,setInviteResult]= useState(null)
  const [inviting,    setInviting]    = useState(false)

  // New org modal
  const [showNewOrg,  setShowNewOrg]  = useState(false)
  const [newOrgName,  setNewOrgName]  = useState('')
  const [creatingOrg, setCreatingOrg] = useState(false)

  // Cloud sync
  const [syncing, setSyncing] = useState(false)

  const load = async () => {
    setLoading(true); setError('')
    try {
      const [m, o, u, l, s] = await Promise.all([
        fetchMetrics(), fetchOrganisations(), fetchUsers(), fetchAuditLogs(), fetchSessions()
      ])
      setMetrics(m); setOrgs(o); setUsers(u); setLogs(l); setSessions(s)
    } catch (e) { setError(e.message) }
    setLoading(false)
  }

  useEffect(() => { load() }, [])

  const handleInvite = async () => {
    if (!inviteEmail) return
    if (!inviteOrgId) { alert('Please select an organisation.'); return }
    setInviting(true); setInviteResult(null)
    try {
      const res = await inviteUser(inviteEmail, inviteRole, inviteOrgId)
      setInviteResult({ success: true, email: inviteEmail, token: res.invite_token, emailSent: res.email_sent })
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
      setNewOrgName(''); setShowNewOrg(false); load()
    } catch (e) { setError(e.message) }
    setCreatingOrg(false)
  }

  const handleDeactivate  = async (id) => { try { await deactivateUser(id);  load() } catch (e) { setError(e.message) } }
  const handleActivate    = async (id) => { try { await activateUser(id);    load() } catch (e) { setError(e.message) } }
  const handleDelete      = async (id) => {
    if (!window.confirm('Permanently delete this user?')) return
    try { await deleteUser(id); load() } catch (e) { setError(e.message) }
  }
  const handleDeleteOrg   = async (id) => {
    if (!window.confirm('Permanently delete this organisation? All users in it will be unassigned.')) return
    try { await deleteOrganisation(id); load() } catch (e) { setError(e.message) }
  }
  const handleRoleChange = async (id, newRole, currentRole) => {
    const roleLabel = { org_admin: 'Org Admin', member: 'Member' }
    if (!window.confirm(`Change role from "${roleLabel[currentRole]}" to "${roleLabel[newRole]}"?\nThis affects permissions immediately.`)) { load(); return }
    try { await changeUserRole(id, newRole); load() } catch (e) { setError(e.message) }
  }

  // KPI computed values
  const totalFaces      = sessions.reduce((a, s) => a + (s.n_faces || 0), 0)
  const totalIdentified = sessions.reduce((a, s) => a + (s.n_identified || 0), 0)
  const accuracy        = totalFaces > 0 ? ((totalIdentified / totalFaces) * 100).toFixed(1) : '—'
  const sessionTrend    = buildSessionTrend(sessions)
  const usersPerOrg     = buildUsersPerOrg(orgs, users)
  const emotionDist     = buildEmotionDist(sessions)
  const userStatus      = buildUserStatus(users)

  const inp = { width: '100%', background: 'rgba(30,41,59,0.8)', border: '1px solid rgba(71,85,105,0.5)', borderRadius: '8px', padding: '0.7rem', color: '#f1f5f9', fontSize: '0.9rem', outline: 'none', boxSizing: 'border-box' }

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)', color: '#f1f5f9' }}>
      <style>{`
        @keyframes pulse {
          0% { opacity: 0.4; transform: scale(0.9); }
          50% { opacity: 1; transform: scale(1.15); }
          100% { opacity: 0.4; transform: scale(0.9); }
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        .spin {
          animation: spin 1s linear infinite;
        }
      `}</style>

      {/* Navbar */}
      <header style={{ background: 'rgba(15,23,42,0.8)', backdropFilter: 'blur(20px)', borderBottom: '1px solid rgba(99,102,241,0.2)', padding: '0 2rem', height: '60px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', position: 'sticky', top: 0, zIndex: 100 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Sparkles size={18} color="#6366f1" />
          <span style={{ fontWeight: 700, fontSize: '1.1rem' }}>VisionX</span>
          <span style={{ color: '#64748b', fontSize: '0.85rem', marginLeft: '0.5rem' }}>Super Admin</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span style={{ color: '#64748b', fontSize: '0.85rem' }}>{user?.email}</span>
          <Badge role="super_admin" />
          <button onClick={logout} style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '8px', padding: '0.4rem 0.8rem', color: '#f87171', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem' }}>
            <LogOut size={14} /> Log Out
          </button>
        </div>
      </header>

      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>

        <div style={{ marginBottom: '2rem' }}>
          <h1 style={{ margin: 0, fontSize: '1.8rem', fontWeight: 700 }}>System Dashboard</h1>
          <p style={{ margin: '0.25rem 0 0', color: '#64748b' }}>Full system overview and management</p>
        </div>

        {error && <div style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '8px', padding: '0.75rem 1rem', marginBottom: '1.5rem', color: '#fca5a5' }}>{error}</div>}

        {/* Tabs */}
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '2rem', borderBottom: '1px solid rgba(71,85,105,0.3)' }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0.6rem 1.2rem', fontSize: '0.9rem', fontWeight: 600, color: tab === t ? '#6366f1' : '#64748b', borderBottom: tab === t ? '2px solid #6366f1' : '2px solid transparent', marginBottom: '-1px' }}>{t}</button>
          ))}
          <button onClick={load} style={{ marginLeft: 'auto', background: 'none', border: 'none', color: '#64748b', cursor: 'pointer', padding: '0.6rem' }}>
            <RefreshCw size={16} />
          </button>
        </div>

        {/* ── Overview (KPI) ──────────────────────────────────────────────── */}
        {tab === 'Overview' && (
          <div>
            {/* Stat cards row */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
              <StatCard icon={Building2}    label="Organisations"    value={orgs.length}                                             color="#8b5cf6" />
              <StatCard icon={Users}        label="Total People"     value={users.length}                                            color="#06b6d4" />
              <StatCard icon={ScanFace}     label="Total Scans"      value={sessions.length}                                         color="#6366f1" />
              <StatCard icon={TrendingUp}   label="Recognition Rate" value={accuracy === '—' ? '—' : `${accuracy}%`}                color="#10b981" sub={`${totalIdentified}/${totalFaces} faces`} />
              <StatCard icon={Activity}     label="Avg Latency"      value={metrics ? `${metrics.avg_latency_s}s` : '—'}             color="#f59e0b" />
              <StatCard icon={ClipboardList}label="Audit Entries"    value={logs.length}                                             color="#f43f5e" />
              <div onClick={async () => { if (syncing) return; setSyncing(true); try { await triggerSync(); await load() } catch (e) { setError(e.message) } setSyncing(false) }} style={{ cursor: syncing ? 'wait' : 'pointer' }} title="Click to trigger Cloud Sync">
                <StatCard 
                  icon={RefreshCw} 
                  label="Cloud Sync" 
                  value={
                    <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      {syncing ? 'Syncing…' : (metrics?.gcs_watcher_status === 'Syncing' ? 'Syncing…' : 'Idle')}
                      <span style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        background: syncing || metrics?.gcs_watcher_status === 'Syncing' ? '#fbbf24' : '#10b981',
                        display: 'inline-block',
                        boxShadow: syncing || metrics?.gcs_watcher_status === 'Syncing' ? '0 0 8px #fbbf24' : '0 0 8px #10b981',
                        animation: syncing || metrics?.gcs_watcher_status === 'Syncing' ? 'pulse 1.5s infinite' : 'none'
                      }} />
                    </span>
                  }
                  color="#10b981" 
                />
              </div>
            </div>

            {/* Charts row 1 */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>

              <ChartCard title="Sessions — Last 7 Days">
                <ResponsiveContainer width="100%" height={180}>
                  <LineChart data={sessionTrend}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(71,85,105,0.3)" />
                    <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 11 }} />
                    <YAxis allowDecimals={false} tick={{ fill: '#64748b', fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(99,102,241,0.3)', borderRadius: '8px', color: '#f1f5f9' }} />
                    <Line type="monotone" dataKey="sessions" stroke="#6366f1" strokeWidth={2} dot={{ fill: '#6366f1', r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="People per Organisation">
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={usersPerOrg}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(71,85,105,0.3)" />
                    <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 11 }} />
                    <YAxis allowDecimals={false} tick={{ fill: '#64748b', fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(99,102,241,0.3)', borderRadius: '8px', color: '#f1f5f9' }} />
                    <Bar dataKey="members" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            </div>

            {/* Charts row 2 */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>

              <ChartCard title="Emotion Distribution (all scans)">
                {emotionDist.length === 0
                  ? <p style={{ color: '#475569', fontSize: '0.85rem', margin: '2rem 0', textAlign: 'center' }}>No scan data yet.</p>
                  : <ResponsiveContainer width="100%" height={180}>
                      <PieChart>
                        <Pie data={emotionDist} dataKey="value" nameKey="name" cx="50%" cy="40%" outerRadius={55} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`} labelLine={false} fontSize={10}>
                          {emotionDist.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                        </Pie>
                        <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(99,102,241,0.3)', borderRadius: '8px', color: '#f1f5f9' }} />
                        <Legend wrapperStyle={{ color: '#94a3b8', fontSize: '0.78rem', marginTop: '10px' }} />
                      </PieChart>
                    </ResponsiveContainer>
                }
              </ChartCard>

              <ChartCard title="Active vs Inactive Users">
                <ResponsiveContainer width="100%" height={180}>
                  <PieChart>
                    <Pie data={userStatus} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={45} outerRadius={65} label={({ name, value }) => `${name}: ${value}`} fontSize={11}>
                      <Cell fill="#10b981" />
                      <Cell fill="#f43f5e" />
                    </Pie>
                    <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(99,102,241,0.3)', borderRadius: '8px', color: '#f1f5f9' }} />
                    <Legend wrapperStyle={{ color: '#94a3b8', fontSize: '0.8rem' }} />
                  </PieChart>
                </ResponsiveContainer>
              </ChartCard>
            </div>

            {/* Quick actions */}
            <div style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(99,102,241,0.2)', borderRadius: '12px', padding: '1.5rem' }}>
              <h3 style={{ margin: '0 0 1rem', fontSize: '1rem' }}>Quick Actions</h3>
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                <button onClick={() => setShowInvite(true)} style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', border: 'none', borderRadius: '8px', padding: '0.7rem 1.2rem', color: '#fff', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600 }}>
                  <Send size={16} /> Invite User
                </button>
                <button onClick={() => setShowNewOrg(true)} style={{ background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.3)', borderRadius: '8px', padding: '0.7rem 1.2rem', color: '#a5b4fc', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600 }}>
                  <Plus size={16} /> New Organisation
                </button>
                <button onClick={() => navigate('/upload')} style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)', borderRadius: '8px', padding: '0.7rem 1.2rem', color: '#34d399', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600 }}>
                  <LayoutDashboard size={16} /> Go to Detection
                </button>
                <button onClick={async () => { if (syncing) return; setSyncing(true); try { await triggerSync(); await load() } catch (e) { setError(e.message) } setSyncing(false) }} disabled={syncing} style={{ background: syncing ? 'rgba(16,185,129,0.3)' : 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)', borderRadius: '8px', padding: '0.7rem 1.2rem', color: '#34d399', cursor: syncing ? 'wait' : 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600 }}>
                  <RefreshCw size={16} className={syncing ? 'spin' : ''} /> {syncing ? 'Syncing…' : 'Cloud Sync'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── Organisations Tab ────────────────────────────────────────────── */}
        {tab === 'Organisations' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0 }}>All Organisations ({orgs.length})</h3>
              <button onClick={() => setShowNewOrg(true)} style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', border: 'none', borderRadius: '8px', padding: '0.6rem 1rem', color: '#fff', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.875rem', fontWeight: 600 }}>
                <Plus size={14} /> New Organisation
              </button>
            </div>
            <div style={{ display: 'grid', gap: '0.75rem' }}>
              {orgs.map(org => (
                <div key={org.id} style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(71,85,105,0.3)', borderRadius: '10px', padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div>
                    <p style={{ margin: 0, fontWeight: 600, fontSize: '0.95rem' }}>{org.name}</p>
                    <p style={{ margin: '0.2rem 0 0', color: '#64748b', fontSize: '0.78rem' }}>ID: {org.id}</p>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <span style={{ background: org.is_active ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)', color: org.is_active ? '#34d399' : '#f87171', border: `1px solid ${org.is_active ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)'}`, borderRadius: '20px', padding: '0.2rem 0.7rem', fontSize: '0.75rem', fontWeight: 600 }}>
                      {org.is_active ? 'Active' : 'Inactive'}
                    </span>
                    <span style={{ color: '#64748b', fontSize: '0.8rem' }}>
                      {users.filter(u => u.org_id === org.id && u.role !== 'super_admin').length} members
                    </span>
                    <button onClick={() => handleDeleteOrg(org.id)} title="Delete Organisation" style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '6px', padding: '0.3rem 0.5rem', color: '#f87171', cursor: 'pointer', display: 'flex', alignItems: 'center', marginLeft: '0.5rem' }}>
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              ))}
              {orgs.length === 0 && <p style={{ color: '#64748b' }}>No organisations yet.</p>}
            </div>
          </div>
        )}

        {/* ── People Tab (renamed from Users) ──────────────────────────────── */}
        {tab === 'People' && (() => {
          const orgMap = Object.fromEntries(orgs.map(o => [o.id, o.name]))
          return (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h3 style={{ margin: 0 }}>All People ({users.length})</h3>
                <button onClick={() => setShowInvite(true)} style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', border: 'none', borderRadius: '8px', padding: '0.6rem 1rem', color: '#fff', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.875rem', fontWeight: 600 }}>
                  <Send size={14} /> Invite User
                </button>
              </div>
              <div style={{ display: 'grid', gap: '0.75rem' }}>
                {users.map(u => (
                  <div key={u.id} style={{ background: u.role === 'super_admin' ? 'rgba(245,158,11,0.04)' : 'rgba(15,23,42,0.6)', border: u.role === 'super_admin' ? '1px solid rgba(245,158,11,0.2)' : '1px solid rgba(71,85,105,0.3)', borderRadius: '10px', padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', opacity: u.is_active ? 1 : 0.6 }}>
                    <div style={{ flex: 1 }}>
                      <p style={{ margin: 0, fontWeight: 600, fontSize: '0.95rem' }}>{u.full_name || '—'}</p>
                      <p style={{ margin: '0.15rem 0 0', color: '#94a3b8', fontSize: '0.82rem' }}>{u.email}</p>
                      <p style={{ margin: '0.15rem 0 0', color: '#64748b', fontSize: '0.75rem' }}>ID: {u.id}</p>
                      {u.role === 'super_admin'
                        ? <p style={{ margin: '0.2rem 0 0', fontSize: '0.75rem', color: '#f59e0b' }}>🌐 Platform-wide</p>
                        : u.org_id
                          ? <p style={{ margin: '0.2rem 0 0', fontSize: '0.75rem', color: '#818cf8' }}>🏢 {orgMap[u.org_id] || u.org_id.slice(0, 8)}</p>
                          : <p style={{ margin: '0.2rem 0 0', fontSize: '0.75rem', color: '#475569', fontStyle: 'italic' }}>⚠ No organisation</p>
                      }
                    </div>
                    <Badge role={u.role} />
                    <span style={{ background: u.is_active ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)', color: u.is_active ? '#34d399' : '#f87171', border: `1px solid ${u.is_active ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)'}`, borderRadius: '20px', padding: '0.2rem 0.7rem', fontSize: '0.75rem', fontWeight: 600 }}>
                      {u.is_active ? 'Active' : 'Inactive'}
                    </span>
                    {u.role !== 'super_admin' && u.email !== user?.email && (
                      <select value={u.role} onChange={e => handleRoleChange(u.id, e.target.value, u.role)} style={{ background: 'rgba(30,41,59,0.8)', border: '1px solid rgba(71,85,105,0.5)', borderRadius: '6px', padding: '0.3rem 0.5rem', color: '#f1f5f9', fontSize: '0.8rem', cursor: 'pointer' }}>
                        <option value="org_admin">Org Admin</option>
                        <option value="member">Member</option>
                      </select>
                    )}
                    {u.email !== user?.email && (
                      <div style={{ display: 'flex', gap: '0.5rem' }}>
                        {u.is_active
                          ? <button onClick={() => handleDeactivate(u.id)} title="Deactivate" style={{ background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.3)', borderRadius: '6px', padding: '0.3rem 0.5rem', color: '#fbbf24', cursor: 'pointer' }}><UserX size={14} /></button>
                          : <button onClick={() => handleActivate(u.id)}   title="Activate"   style={{ background: 'rgba(16,185,129,0.1)',  border: '1px solid rgba(16,185,129,0.3)',  borderRadius: '6px', padding: '0.3rem 0.5rem', color: '#34d399', cursor: 'pointer' }}><UserCheck size={14} /></button>
                        }
                        <button onClick={() => handleDelete(u.id)} title="Delete" style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '6px', padding: '0.3rem 0.5rem', color: '#f87171', cursor: 'pointer' }}><Trash2 size={14} /></button>
                      </div>
                    )}
                  </div>
                ))}
                {users.length === 0 && <p style={{ color: '#64748b' }}>No users yet.</p>}
              </div>
            </div>
          )
        })()}

        {/* ── Sessions Tab ─────────────────────────────────────────────────── */}
        {tab === 'Sessions' && (() => {
          const orgMap  = Object.fromEntries(orgs.map(o => [o.id, o.name]))
          const filtered = sessionOrgFilter === 'all' ? sessions : sessions.filter(s => s.org_id === sessionOrgFilter)
          return (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
                <h3 style={{ margin: 0 }}>All Sessions ({filtered.length})</h3>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <label style={{ color: '#94a3b8', fontSize: '0.8rem' }}>Filter by org:</label>
                  <select value={sessionOrgFilter} onChange={e => setSessionOrgFilter(e.target.value)} style={{ background: 'rgba(15,23,42,0.8)', border: '1px solid rgba(71,85,105,0.4)', borderRadius: '6px', padding: '0.35rem 0.6rem', color: '#f1f5f9', fontSize: '0.8rem' }}>
                    <option value="all">All Organisations</option>
                    {orgs.map(o => <option key={o.id} value={o.id}>{o.name}</option>)}
                  </select>
                </div>
              </div>
              {filtered.length === 0
                ? <p style={{ color: '#64748b' }}>No sessions found.</p>
                : <div style={{ display: 'grid', gap: '0.75rem' }}>
                    {filtered.map(s => {
                      const isExpanded = expandedSession === s.id
                      const resultsData= s.results_json || {}
                      const results    = resultsData.results || (Array.isArray(s.results_json) ? s.results_json : [])
                      const method     = resultsData.detection_method || (s.annotated_image ? 'Upload' : 'Unknown')
                      const isKnown = (n) => n && n.toUpperCase() !== 'UNKNOWN' && n.toLowerCase() !== 'unknown subject' && n.toLowerCase() !== 'unrecognised'
                      const known      = results.filter(f => isKnown(f.name))
                      const unknown    = results.filter(f => !isKnown(f.name))
                      const methodBadge = {
                        'Upload':    { color: '#60a5fa', bg: 'rgba(96,165,250,0.12)',  border: 'rgba(96,165,250,0.35)',  icon: '📤' },
                        'Snapshot':  { color: '#fbbf24', bg: 'rgba(251,191,36,0.12)',  border: 'rgba(251,191,36,0.35)',  icon: '📸' },
                        'Live Feed': { color: '#34d399', bg: 'rgba(52,211,153,0.12)',  border: 'rgba(52,211,153,0.35)',  icon: '🎥' },
                      }[method] || { color: '#94a3b8', bg: 'rgba(148,163,184,0.1)', border: 'rgba(148,163,184,0.3)', icon: '❓' }

                      return (
                        <div key={s.id} style={{ background: 'rgba(15,23,42,0.6)', border: `1px solid ${isExpanded ? 'rgba(99,102,241,0.4)' : 'rgba(71,85,105,0.3)'}`, borderRadius: '10px', overflow: 'hidden' }}>
                          <div onClick={() => setExpandedSession(isExpanded ? null : s.id)} style={{ padding: '1rem 1.25rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
                            <ScanFace size={16} color="#6366f1" />
                            <div style={{ flex: 1 }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.25rem', flexWrap: 'wrap' }}>
                                <p style={{ margin: 0, fontWeight: 600, fontSize: '0.875rem' }}>
                                  {known.length} identified · {unknown.length} unknown · {s.n_faces} face{s.n_faces !== 1 ? 's' : ''}
                                </p>
                                <span style={{ background: methodBadge.bg, border: `1px solid ${methodBadge.border}`, borderRadius: '20px', padding: '0.15rem 0.6rem', fontSize: '0.7rem', color: methodBadge.color, fontWeight: 700, whiteSpace: 'nowrap' }}>
                                  {methodBadge.icon} {method}
                                </span>
                              </div>
                              <p style={{ margin: 0, color: '#64748b', fontSize: '0.75rem' }}>
                                {new Date(s.created_at).toLocaleString()} · {s.elapsed_s?.toFixed(2)}s
                              </p>
                            </div>
                            <span style={{ background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.3)', borderRadius: '20px', padding: '0.2rem 0.7rem', color: '#818cf8', fontSize: '0.73rem', fontWeight: 600 }}>🏢 {orgMap[s.org_id] || '—'}</span>
                            {isExpanded ? <ChevronUp size={14} color="#64748b" /> : <ChevronDown size={14} color="#64748b" />}
                          </div>
                          {isExpanded && (
                            <div style={{ padding: '0 1.25rem 1rem', borderTop: '1px solid rgba(71,85,105,0.2)', background: 'rgba(15,23,42,0.4)' }}>
                              {s.annotated_image ? (
                                <div style={{ marginBottom: '1rem', marginTop: '0.75rem' }}>
                                  <p style={{ margin: '0 0 0.5rem', fontSize: '0.78rem', color: '#94a3b8', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>📸 Annotated Image</p>
                                  <img src={`data:image/jpeg;base64,${s.annotated_image}`} alt="Annotated" style={{ width: '100%', maxWidth: '480px', borderRadius: '8px', border: `1px solid ${methodBadge.border}`, display: 'block' }} />
                                </div>
                              ) : method === 'Live Feed' ? (
                                <p style={{ margin: '0.75rem 0', fontSize: '0.78rem', color: '#475569', fontStyle: 'italic' }}>🎥 Live Feed frames are not saved as images.</p>
                              ) : null}
                              <div style={{ marginTop: '0.5rem' }}>
                                {results.map((f, i) => {
                                  const isFaceKnown = isKnown(f.name)
                                  return (
                                    <span key={i} style={{ display: 'inline-block', margin: '0.3rem 0.3rem 0 0', background: isFaceKnown ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)', border: `1px solid ${isFaceKnown ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)'}`, color: isFaceKnown ? '#34d399' : '#f87171', borderRadius: '6px', padding: '0.2rem 0.6rem', fontSize: '0.75rem' }}>
                                      {isFaceKnown ? f.name : 'Unrecognised'}{f.emotion ? ` · ${f.emotion}` : ''}
                                    </span>
                                  )
                                })}
                              </div>
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
              }
            </div>
          )
        })()}

        {/* ── Audit Logs Tab ───────────────────────────────────────────────── */}
        {tab === 'Audit Logs' && (
          <div>
            <h3 style={{ margin: '0 0 1rem' }}>Audit Logs ({logs.length})</h3>
            {logs.length === 0
              ? <p style={{ color: '#64748b' }}>No audit logs yet.</p>
              : <div style={{ display: 'grid', gap: '0.5rem' }}>
                  {logs.map((log, idx) => {
                    const detail = log.detail || {}
                    const targetLabel = detail.invited_email || detail.email || detail.name || (log.target_id ? log.target_id.slice(0, 8) + '…' : '—')
                    const ACTION_LABELS = {
                      'auth.signup'       : '🔐 Super admin created',
                      'auth.invite'       : '📨 Invite sent',
                      'auth.invite_accept': '✅ Invite accepted',
                      'user.role_change'  : '🔄 Role changed',
                      'user.deactivate'   : '🚫 User deactivated',
                      'user.activate'     : '✅ User reactivated',
                      'user.delete'       : '🗑️ User deleted',
                      'org.create'        : '🏢 Organisation created',
                    }
                    return (
                      <div key={idx} style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(71,85,105,0.3)', borderRadius: '8px', padding: '0.75rem 1rem', display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
                        <ShieldAlert size={16} color="#f59e0b" style={{ flexShrink: 0, marginTop: '2px' }} />
                        <div style={{ flex: 1 }}>
                          <p style={{ margin: 0, fontSize: '0.875rem', fontWeight: 600 }}>{ACTION_LABELS[log.action] || log.action}</p>
                          <p style={{ margin: '0.2rem 0 0', color: '#94a3b8', fontSize: '0.78rem' }}>
                            {targetLabel && <span>Target: <strong style={{ color: '#f1f5f9' }}>{targetLabel}</strong></span>}
                            {detail.role && <span style={{ marginLeft: '0.75rem' }}>Role: <strong style={{ color: '#a5b4fc' }}>{detail.role}</strong></span>}
                            {detail.old_role && detail.new_role && <span style={{ marginLeft: '0.75rem' }}><strong style={{ color: '#f87171' }}>{detail.old_role}</strong>{' → '}<strong style={{ color: '#34d399' }}>{detail.new_role}</strong></span>}
                          </p>
                        </div>
                        <span style={{ color: '#475569', fontSize: '0.78rem', flexShrink: 0 }}>{log.created_at ? new Date(log.created_at).toLocaleString() : ''}</span>
                      </div>
                    )
                  })}
                </div>
            }
          </div>
        )}
      </div>

      {/* ── Invite Modal ──────────────────────────────────────────────────── */}
      {showInvite && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 200 }}>
          <div style={{ background: '#1e293b', border: '1px solid rgba(99,102,241,0.3)', borderRadius: '16px', padding: '2rem', width: '100%', maxWidth: '420px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h3 style={{ margin: 0 }}>Invite User</h3>
              <button onClick={() => { setShowInvite(false); setInviteResult(null); setInviteOrgId('') }} style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}><X size={20} /></button>
            </div>
            {inviteResult?.success ? (
              <div>
                <div style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)', borderRadius: '8px', padding: '1rem', marginBottom: '1.2rem' }}>
                  <p style={{ color: '#34d399', fontWeight: 600, margin: '0 0 0.5rem', fontSize: '0.9rem' }}>
                    ✅ {inviteResult.emailSent ? `Invite email sent to ${inviteResult.email}` : `Invite generated for ${inviteResult.email} (email not sent)`}
                  </p>
                  <p style={{ color: '#64748b', fontSize: '0.78rem', margin: '0 0 0.75rem' }}>
                    Share this signup link:<br />
                    <span style={{ color: '#a5b4fc' }}>{import.meta.env.VITE_FRONTEND_URL || window.location.origin}/signup?token=…</span>
                  </p>
                  <div style={{ background: 'rgba(15,23,42,0.8)', borderRadius: '6px', padding: '0.75rem', fontSize: '0.72rem', color: '#94a3b8', wordBreak: 'break-all', marginBottom: '0.75rem', maxHeight: '80px', overflowY: 'auto', border: '1px solid rgba(71,85,105,0.3)' }}>
                    {inviteResult.token}
                  </div>
                  <button onClick={() => { navigator.clipboard.writeText(`${import.meta.env.VITE_FRONTEND_URL || window.location.origin}/signup?token=${inviteResult.token}`); alert('Signup link copied!') }} style={{ width: '100%', background: 'rgba(99,102,241,0.2)', border: '1px solid rgba(99,102,241,0.4)', borderRadius: '8px', padding: '0.6rem', color: '#a5b4fc', cursor: 'pointer', fontWeight: 600, fontSize: '0.875rem' }}>
                    📋 Copy Signup Link
                  </button>
                </div>
                <button onClick={() => { setShowInvite(false); setInviteResult(null); setInviteOrgId('') }} style={{ width: '100%', background: 'rgba(71,85,105,0.2)', border: '1px solid rgba(71,85,105,0.3)', borderRadius: '8px', padding: '0.7rem', color: '#94a3b8', cursor: 'pointer', fontWeight: 600 }}>Close</button>
              </div>
            ) : (
              <div>
                <div style={{ marginBottom: '1rem' }}>
                  <label style={{ display: 'block', color: '#94a3b8', fontSize: '0.85rem', marginBottom: '0.4rem' }}>Email</label>
                  <input type="email" value={inviteEmail} onChange={e => setInviteEmail(e.target.value)} placeholder="user@example.com" style={inp} />
                </div>
                <div style={{ marginBottom: '1rem' }}>
                  <label style={{ display: 'block', color: '#94a3b8', fontSize: '0.85rem', marginBottom: '0.4rem' }}>Role</label>
                  <select value={inviteRole} onChange={e => setInviteRole(e.target.value)} style={inp}>
                    <option value="org_admin">Org Admin</option>
                    <option value="member">Member</option>
                  </select>
                </div>
                <div style={{ marginBottom: '1.5rem' }}>
                  <label style={{ display: 'block', color: '#94a3b8', fontSize: '0.85rem', marginBottom: '0.4rem' }}>Organisation</label>
                  <select value={inviteOrgId} onChange={e => setInviteOrgId(e.target.value)} style={{ ...inp, color: inviteOrgId ? '#f1f5f9' : '#64748b' }}>
                    <option value="">— Select Organisation —</option>
                    {orgs.filter(o => o.is_active).map(o => <option key={o.id} value={o.id}>{o.name}</option>)}
                  </select>
                </div>
                {inviteResult?.success === false && <div style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '8px', padding: '0.75rem', marginBottom: '1rem', color: '#fca5a5', fontSize: '0.85rem' }}>{inviteResult.message}</div>}
                <button onClick={handleInvite} disabled={inviting} style={{ width: '100%', background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', border: 'none', borderRadius: '8px', padding: '0.8rem', color: '#fff', fontWeight: 600, cursor: inviting ? 'not-allowed' : 'pointer', opacity: inviting ? 0.7 : 1 }}>
                  {inviting ? 'Generating…' : 'Send Invite'}
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── New Org Modal ─────────────────────────────────────────────────── */}
      {showNewOrg && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 200 }}>
          <div style={{ background: '#1e293b', border: '1px solid rgba(99,102,241,0.3)', borderRadius: '16px', padding: '2rem', width: '100%', maxWidth: '380px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h3 style={{ margin: 0 }}>New Organisation</h3>
              <button onClick={() => setShowNewOrg(false)} style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}><X size={20} /></button>
            </div>
            <div style={{ marginBottom: '1.5rem' }}>
              <label style={{ display: 'block', color: '#94a3b8', fontSize: '0.85rem', marginBottom: '0.4rem' }}>Organisation Name</label>
              <input type="text" value={newOrgName} onChange={e => setNewOrgName(e.target.value)} placeholder="e.g. IT Team" style={inp} />
            </div>
            <button onClick={handleCreateOrg} disabled={creatingOrg} style={{ width: '100%', background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', border: 'none', borderRadius: '8px', padding: '0.8rem', color: '#fff', fontWeight: 600, cursor: creatingOrg ? 'not-allowed' : 'pointer' }}>
              {creatingOrg ? 'Creating…' : 'Create Organisation'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
