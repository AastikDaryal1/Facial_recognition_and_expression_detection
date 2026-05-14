/**
 * src/components/AppShell.jsx
 * ────────────────────────────
 * Top navigation bar + page layout wrapper.
 *
 * Auth: JWT via AuthContext (veerojasvi)
 * Mobile menu: hamburger toggle (from Shubh's mobile responsiveness update)
 */

import { useState } from 'react'
import { Link, NavLink } from 'react-router-dom'
import {
  Sparkles,
  LogOut,
  LayoutDashboard,
  Upload,
  Camera,
  Home,
  ClipboardList,
  Menu,
  X,
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
    { to: '/',         label: 'Home',     icon: Home          },
    { to: '/upload',   label: 'Upload',   icon: Upload        },
    { to: '/live',     label: 'Live',     icon: Camera        },
    { to: '/sessions', label: 'Sessions', icon: ClipboardList },
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

  // Mobile menu state — Shubh's mobile responsiveness
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const toggleMobileMenu = () => setIsMobileMenuOpen(prev => !prev)
  const closeMobileMenu  = () => setIsMobileMenuOpen(false)

  const navItems  = user ? (NAV_ITEMS[user.role] || NAV_ITEMS.user) : NAV_ITEMS.user
  const roleBadge = user ? ROLE_BADGE[user.role] : null

  return (
    <div className="app-shell">
      <header className="topbar glass-card">

        {/* Brand */}
        <Link className="brand" to="/" onClick={closeMobileMenu}>
          <Sparkles size={18} />
          <span>VisionX</span>
        </Link>

        {/* Hamburger toggle — visible on mobile only (CSS controls display) */}
        <button className="mobile-toggle" onClick={toggleMobileMenu} aria-label="Toggle menu">
          {isMobileMenuOpen ? <X size={22} /> : <Menu size={22} />}
        </button>

        {/* Nav — gains 'mobile-open' class when hamburger is active */}
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
                color       : '#64748b',
                fontSize    : '0.8rem',
                maxWidth    : '140px',
                overflow    : 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace  : 'nowrap',
              }}>
                {user.email}
              </span>

              {/* Logout button */}
              <button
                onClick={() => { logout(); closeMobileMenu() }}
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
