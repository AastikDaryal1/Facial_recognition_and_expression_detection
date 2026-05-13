/**
 * src/AuthContext.jsx
 * ────────────────────
 * Global auth state for the entire app.
 * Replaces the old ApiKeyContext which used a raw API key.
 *
 * Provides:
 *   - user        → { email, role } or null if not logged in
 *   - isLoading   → true while checking stored tokens on page load
 *   - login()     → call with email + password, sets user state
 *   - logout()    → clears tokens, redirects to /login
 *
 * Usage in any component:
 *   const { user, login, logout } = useAuth()
 */

import { createContext, useContext, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  login as apiLogin,
  logout as apiLogout,
  restoreSession as apiRestoreSession,
  getAccessToken,
  getRefreshToken,
  setAccessToken,
  setRefreshToken,
  setUserInfo,
  getUserRole,
  getUserEmail,
  decodeToken,
  clearTokens,
} from './api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)       // { email, role } or null
  const [isLoading, setIsLoading] = useState(true) // checking stored tokens

  // ── On page load — restore session from stored tokens ──────────────────────
  useEffect(() => {
    const init = async () => {
      try {
        const result = await apiRestoreSession()
        if (result) {
          setUser(result)
        }
      } catch {
        // Silently fail, user stays logged out
      }
      setIsLoading(false)
    }
    init()
  }, [])

  // ── Auto-logout on inactivity (24 minutes — matches JWT expiry) ────────────
  useEffect(() => {
    if (!user) return

    const TIMEOUT_MS = 24 * 60 * 1000
    let timeout

    const resetTimer = () => {
      clearTimeout(timeout)
      timeout = setTimeout(() => {
        handleLogout()
      }, TIMEOUT_MS)
    }

    resetTimer()

    window.addEventListener('mousemove', resetTimer)
    window.addEventListener('keydown',   resetTimer)
    window.addEventListener('click',     resetTimer)
    window.addEventListener('scroll',    resetTimer)

    return () => {
      clearTimeout(timeout)
      window.removeEventListener('mousemove', resetTimer)
      window.removeEventListener('keydown',   resetTimer)
      window.removeEventListener('click',     resetTimer)
      window.removeEventListener('scroll',    resetTimer)
    }
  }, [user])

  // ── Login ──────────────────────────────────────────────────────────────────
  const handleLogin = async (email, password) => {
    const result = await apiLogin(email, password)
    setUser({ email: result.email, role: result.role })
    return result
  }

  // ── Logout ─────────────────────────────────────────────────────────────────
  const handleLogout = async () => {
    await apiLogout()
    setUser(null)
    // Redirect to login — use window.location so it works outside Router context
    window.location.href = '/login'
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        login  : handleLogin,
        logout : handleLogout,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}
