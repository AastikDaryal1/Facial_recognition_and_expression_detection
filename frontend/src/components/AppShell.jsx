/**
 * src/components/AppShell.jsx
 * ────────────────────────────
 * Top navigation bar + page layout wrapper.
 * Extracted from old App.jsx — same design, JWT auth replacing API key.
 *
 * Changes from old version:
 *   - No more API key modal
 *   - Login/logout uses AuthContext (JWT)
 *   - Nav items change based on role
 *   - Shows user email + role badge in navbar
 */

import { Link, NavLink } from 'react-router-dom'
import {
  Sparkles,
  LogOut,
  LayoutDashboard,
  Users,
  Upload,
  Camera,
  Home,
} from 'lucide-react'
import { useAuth } from '../AuthContext'

// Nav items per role
const NAV_ITEMS = {
  super_admin: [
    { to: '/admin',  label: 'Dashboard', icon: LayoutDashboard },
    { to: '/upload', label: 'Upload',    icon: Upload },
    { to: '/live',   label: 'Live',      icon: Camera },
  ],
  org_admin: [
    { to: '/org-dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { to: '/upload',        label: 'Upload',    icon: Upload },
    { to: '/live',          label: 'Live',      icon: Camera },
  ],
  user: [
    { to: '/',       label: 'Home',   icon: Home   },
    { to: '/upload', label: 'Upload', icon: Upload },
    { to: '/live',   label: 'Live',   icon: Camera },
  ],
}

// Role badge colors
const ROLE_BADGE = {
  super_admin : { label: 'Super Admin', color: '#f59e0b', bg: 'rgba(245,158,11,0.1)'  },
  org_admin   : { label: 'Org Admin',   color: '#6366f1', bg: 'rgba(99,102,241,0.1)'  },
  user        : { label: 'Member',      color: '#10b981', bg: 'rgba(16,185,129,0.1)'  },
}

export default function AppShell({ children }) {
  const { user, logout } = useAuth()

  const navItems  = user ? (NAV_ITEMS[user.role] || NAV_ITEMS.user) : NAV_ITEMS.user
  const roleBadge = user ? ROLE_BADGE[user.role] : null

  return (
    <div className="app-shell">
      <header className="topbar glass-card">

        {/* Brand */}
        <Link className="brand" to="/">
          <Sparkles size={18} />
          <span>VisionX</span>
        </Link>

        {/* Nav links */}
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

          {/* User info + logout */}
          {user && (
            <div style={{
              display    : 'flex',
              alignItems : 'center',
              gap        : '10px',
              marginLeft : '0.5rem',
            }}>

              {/* Role badge */}
              {roleBadge && (
                <span style={{
                  background   : roleBadge.bg,
                  color        : roleBadge.color,
                  border       : `1px solid ${roleBadge.color}40`,
                  borderRadius : '20px',
                  padding      : '0.2rem 0.7rem',
                  fontSize     : '0.75rem',
                  fontWeight   : 600,
                }}>
                  {roleBadge.label}
                </span>
              )}

              {/* Email */}
              <span style={{
                color    : '#64748b',
                fontSize : '0.8rem',
                maxWidth : '140px',
                overflow : 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace  : 'nowrap',
              }}>
                {user.email}
              </span>

              {/* Logout button */}
              <button
                onClick = {logout}
                className="secondary-btn"
                style={{
                  display    : 'flex',
                  alignItems : 'center',
                  gap        : '6px',
                  padding    : '0.4rem 0.8rem',
                  fontSize   : '0.85rem',
                }}
              >
                <LogOut size={14} />
                Log Out
              </button>
            </div>
          )}
        </nav>
      </header>

      <main className="content-wrap">
        {children}
      </main>
    </div>
  )
}
