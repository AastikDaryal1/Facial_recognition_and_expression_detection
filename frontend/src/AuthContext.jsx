/**
 * src/AuthContext.jsx
 * ──────────────────────────────────────────────────────────────────────────
 * Provides auth state and login/logout to the whole app.
 *
 * user = { email, role, org_id }  or  null when not logged in
 * isLoading = true while restoring session on first page load
 */

import { createContext, useContext, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  login as apiLogin,
  logout as apiLogout,
  restoreSession,
  clearAccessToken,
  clearRefreshToken,
} from './api'

const AuthContext = createContext(null)

const INACTIVITY_MS = 24 * 60 * 1000   // 24 minutes

export function AuthProvider({ children }) {
  const [user, setUser]         = useState(null)
  const [isLoading, setLoading] = useState(true)
  const navigate                = useNavigate()
  const inactivityTimer         = useRef(null)

  // ── Restore session on first load ──────────────────────────────────────

  useEffect(() => {
    restoreSession()
      .then((restored) => {
        if (restored) setUser(restored)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  // ── Inactivity auto-logout ─────────────────────────────────────────────

  useEffect(() => {
    if (!user) return

    const reset = () => {
      clearTimeout(inactivityTimer.current)
      inactivityTimer.current = setTimeout(() => doLogout(), INACTIVITY_MS)
    }

    reset()
    window.addEventListener('mousemove', reset)
    window.addEventListener('keydown', reset)
    window.addEventListener('click', reset)

    return () => {
      clearTimeout(inactivityTimer.current)
      window.removeEventListener('mousemove', reset)
      window.removeEventListener('keydown', reset)
      window.removeEventListener('click', reset)
    }
  }, [user])

  // ── Actions ────────────────────────────────────────────────────────────

  const login = async (email, password) => {
    const userData = await apiLogin(email, password)
    setUser(userData)
    return userData   // caller uses role to redirect
  }

  const doLogout = async () => {
    await apiLogout()
    setUser(null)
    navigate('/login')
  }

  return (
    <AuthContext.Provider value={{ user, isLoading, login, logout: doLogout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}
