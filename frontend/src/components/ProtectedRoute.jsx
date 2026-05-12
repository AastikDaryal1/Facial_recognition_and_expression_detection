/**
 * src/components/ProtectedRoute.jsx
 * ───────────────────────────────────
 * Wraps any route that requires authentication or a specific role.
 *
 * Usage:
 *   <ProtectedRoute>                          → any logged-in user
 *   <ProtectedRoute role="super_admin">       → super_admin only
 *   <ProtectedRoute role="org_admin">         → org_admin only
 *   <ProtectedRoute role={["super_admin","org_admin"]}> → either role
 *
 * What it does:
 *   - If still loading session → shows a spinner
 *   - If not logged in → redirects to /login
 *   - If logged in but wrong role → redirects to their own dashboard
 *   - If logged in and correct role → renders the page
 */

import { Navigate } from 'react-router-dom'
import { useAuth } from '../AuthContext'

// Maps each role to their home dashboard
const ROLE_HOME = {
  super_admin : '/admin',
  org_admin   : '/org-dashboard',
  user        : '/',
}

export default function ProtectedRoute({ children, role }) {
  const { user, isLoading } = useAuth()

  // ── Still checking stored tokens ───────────────────────────────────────────
  if (isLoading) {
    return (
      <div style={{
        display        : 'flex',
        alignItems     : 'center',
        justifyContent : 'center',
        height         : '100vh',
        background     : '#0f172a',
        color          : '#94a3b8',
        fontSize       : '1rem',
        gap            : '12px',
      }}>
        <div style={{
          width        : '20px',
          height       : '20px',
          border       : '2px solid #334155',
          borderTop    : '2px solid #6366f1',
          borderRadius : '50%',
          animation    : 'spin 0.8s linear infinite',
        }} />
        Loading...
        <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
      </div>
    )
  }

  // ── Not logged in → go to login ────────────────────────────────────────────
  if (!user) {
    return <Navigate to="/login" replace />
  }

  // ── Check role requirement ─────────────────────────────────────────────────
  if (role) {
    const allowedRoles = Array.isArray(role) ? role : [role]

    if (!allowedRoles.includes(user.role)) {
      // Logged in but wrong role — redirect to their own dashboard
      const home = ROLE_HOME[user.role] || '/'
      return <Navigate to={home} replace />
    }
  }

  // ── All good — render the page ─────────────────────────────────────────────
  return children
}
