/**
 * src/App.jsx
 * ────────────
 * Root of the app. Sets up:
 *   - AuthProvider (JWT auth state)
 *   - BrowserRouter (routing)
 *   - All routes with role-based protection
 *
 * Route map:
 *   /login          → LoginPage           (public)
 *   /signup         → SignupInvitePage    (public — invite token required)
 *   /               → HomePage            (any logged-in user)
 *   /upload         → UploadPage          (any logged-in user)
 *   /live           → LivePage            (any logged-in user)
 *   /admin          → SuperAdminDashboard (super_admin only)
 *   /org-dashboard  → OrgAdminDashboard   (org_admin only)
 *   *               → NotFound
 */

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './AuthContext'

// Pages
import LoginPage from './pages/LoginPage'
import SignupInvitePage from './pages/SignupInvitePage'
import SuperAdminDashboard from './pages/SuperAdminDashboard'
import OrgAdminDashboard from './pages/OrgAdminDashboard'

// Protected route wrapper
import ProtectedRoute from './components/ProtectedRoute'

// Existing pages — kept exactly as they were, just extracted into their own files
import HomePage from './pages/HomePage'
import UploadPage from './pages/UploadPage'
import LivePage from './pages/LivePage'
import UserSessionsPage from './pages/UserSessionsPage'

// App shell (navbar + layout) — kept exactly as before
import AppShell from './components/AppShell'

function NotFound() {
  return (
    <div style={{
      minHeight: '100vh',
      background: '#0f172a',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      flexDirection: 'column',
      gap: '1rem',
      color: '#94a3b8',
    }}>
      <h2 style={{ color: '#f1f5f9', margin: 0 }}>404 — Page not found</h2>
      <p style={{ margin: 0 }}>The page you're looking for doesn't exist.</p>
      <a href="/" style={{ color: '#6366f1', textDecoration: 'none' }}>Go home</a>
    </div>
  )
}

// Smart redirect — sends logged-in users to their dashboard
function RootRedirect() {
  const { user, isLoading } = useAuth()

  if (isLoading) return null

  if (!user) return <Navigate to="/login" replace />

  if (user.role === 'super_admin') return <Navigate to="/admin" replace />
  if (user.role === 'org_admin') return <Navigate to="/org-dashboard" replace />

  // Regular user stays at home
  return (
    <ProtectedRoute>
      <AppShell>
        <HomePage />
      </AppShell>
    </ProtectedRoute>
  )
}

function AppRoutes() {
  return (
    <Routes>

      {/* ── Public routes ───────────────────────────────────────────────── */}
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignupInvitePage />} />

      {/* ── Root — smart redirect based on role ─────────────────────────── */}
      <Route path="/" element={<RootRedirect />} />

      {/* ── Regular user routes ─────────────────────────────────────────── */}
      <Route
        path="/upload"
        element={
          <ProtectedRoute>
            <AppShell>
              <UploadPage />
            </AppShell>
          </ProtectedRoute>
        }
      />
      <Route
        path="/live"
        element={
          <ProtectedRoute>
            <AppShell>
              <LivePage />
            </AppShell>
          </ProtectedRoute>
        }
      />
      <Route
        path="/sessions"
        element={
          <ProtectedRoute>
            <AppShell>
              <UserSessionsPage />
            </AppShell>
          </ProtectedRoute>
        }
      />

      {/* ── Super admin dashboard ────────────────────────────────────────── */}
      <Route
        path="/admin"
        element={
          <ProtectedRoute role="super_admin">
            <SuperAdminDashboard />
          </ProtectedRoute>
        }
      />

      {/* ── Org admin dashboard ──────────────────────────────────────────── */}
      <Route
        path="/org-dashboard"
        element={
          <ProtectedRoute role="org_admin">
            <OrgAdminDashboard />
          </ProtectedRoute>
        }
      />

      {/* ── 404 ─────────────────────────────────────────────────────────── */}
      <Route path="*" element={<NotFound />} />

    </Routes>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  )
}