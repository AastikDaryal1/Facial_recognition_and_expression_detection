/**
 * src/pages/LoginPage.jsx
 * ────────────────────────
 * Login page — same for all roles.
 * After login, redirects to the correct dashboard based on role:
 *   super_admin → /admin
 *   org_admin   → /org-dashboard
 *   user        → /
 */

import { useState } from 'react'
import { useNavigate, Link, useSearchParams } from 'react-router-dom'
import { Sparkles, Mail, Lock, LogIn, Eye, EyeOff } from 'lucide-react'
import { useAuth } from '../AuthContext'

export default function LoginPage() {
  const { login, user } = useAuth()
  const navigate        = useNavigate()
  const [searchParams]  = useSearchParams()

  const [email,       setEmail]       = useState('')
  const [password,    setPassword]    = useState('')
  const [showPass,    setShowPass]    = useState(false)
  const [loading,     setLoading]     = useState(false)
  const [error,       setError]       = useState('')

  // If already logged in, redirect immediately
  if (user) {
    const home = {
      super_admin : '/admin',
      org_admin   : '/org-dashboard',
      user        : '/',
    }
    navigate(home[user.role] || '/', { replace: true })
    return null
  }

  // Check if we came from an invite link
  const inviteToken = searchParams.get('token')

  const handleLogin = async (e) => {
    e.preventDefault()
    if (!email || !password) {
      setError('Please enter your email and password.')
      return
    }
    setLoading(true)
    setError('')

    try {
      const result = await login(email, password)

      // Redirect based on role
      if (result.role === 'super_admin') {
        navigate('/admin', { replace: true })
      } else if (result.role === 'org_admin') {
        navigate('/org-dashboard', { replace: true })
      } else {
        navigate('/', { replace: true })
      }
    } catch (err) {
      setError(err.message || 'Login failed. Please check your credentials.')
    }

    setLoading(false)
  }

  return (
    <div style={{
      minHeight       : '100vh',
      background      : 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)',
      display         : 'flex',
      alignItems      : 'center',
      justifyContent  : 'center',
      padding         : '1rem',
      fontFamily      : 'inherit',
    }}>

      {/* Glow blobs in background */}
      <div style={{
        position   : 'fixed',
        top        : '20%',
        left       : '10%',
        width      : '300px',
        height     : '300px',
        background : 'radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%)',
        borderRadius: '50%',
        pointerEvents: 'none',
      }} />
      <div style={{
        position   : 'fixed',
        bottom     : '20%',
        right      : '10%',
        width      : '250px',
        height     : '250px',
        background : 'radial-gradient(circle, rgba(139,92,246,0.12) 0%, transparent 70%)',
        borderRadius: '50%',
        pointerEvents: 'none',
      }} />

      {/* Login card */}
      <div style={{
        width        : '100%',
        maxWidth     : '420px',
        background   : 'rgba(15, 23, 42, 0.8)',
        backdropFilter: 'blur(20px)',
        border       : '1px solid rgba(99, 102, 241, 0.3)',
        borderRadius : '16px',
        padding      : '2.5rem',
        boxShadow    : '0 25px 50px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05)',
        position     : 'relative',
        zIndex       : 1,
      }}>

        {/* Brand */}
        <div style={{
          display        : 'flex',
          alignItems     : 'center',
          gap            : '10px',
          marginBottom   : '2rem',
          justifyContent : 'center',
        }}>
          <Sparkles size={22} color="#6366f1" />
          <span style={{
            fontSize   : '1.4rem',
            fontWeight : 700,
            color      : '#f1f5f9',
            letterSpacing: '0.5px',
          }}>VisionX</span>
        </div>

        {/* Heading */}
        <h1 style={{
          fontSize     : '1.6rem',
          fontWeight   : 700,
          color        : '#f1f5f9',
          marginBottom : '0.4rem',
          textAlign    : 'center',
          marginTop    : 0,
        }}>
          Welcome back
        </h1>
        <p style={{
          color        : '#64748b',
          textAlign    : 'center',
          marginBottom : '2rem',
          fontSize     : '0.9rem',
        }}>
          Sign in to your account to continue
        </p>

        {/* Invite notice */}
        {inviteToken && (
          <div style={{
            background   : 'rgba(99, 102, 241, 0.1)',
            border       : '1px solid rgba(99, 102, 241, 0.3)',
            borderRadius : '8px',
            padding      : '0.75rem 1rem',
            marginBottom : '1.5rem',
            fontSize     : '0.85rem',
            color        : '#a5b4fc',
          }}>
            You have an invite! <Link to={`/signup?token=${inviteToken}`} style={{ color: '#6366f1', textDecoration: 'underline' }}>
              Click here to create your account instead.
            </Link>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleLogin}>

          {/* Email field */}
          <div style={{ marginBottom: '1.2rem' }}>
            <label style={{
              display      : 'block',
              color        : '#94a3b8',
              fontSize     : '0.85rem',
              marginBottom : '0.5rem',
              fontWeight   : 500,
            }}>
              Email address
            </label>
            <div style={{ position: 'relative' }}>
              <Mail size={16} style={{
                position  : 'absolute',
                left      : '14px',
                top       : '50%',
                transform : 'translateY(-50%)',
                color     : '#475569',
              }} />
              <input
                type        = "email"
                value       = {email}
                onChange    = {(e) => setEmail(e.target.value)}
                placeholder = "you@example.com"
                autoComplete= "email"
                style={{
                  width          : '100%',
                  background     : 'rgba(30, 41, 59, 0.8)',
                  border         : '1px solid rgba(71, 85, 105, 0.5)',
                  borderRadius   : '8px',
                  padding        : '0.75rem 0.75rem 0.75rem 2.5rem',
                  color          : '#f1f5f9',
                  fontSize       : '0.95rem',
                  outline        : 'none',
                  boxSizing      : 'border-box',
                  transition     : 'border-color 0.2s',
                }}
                onFocus = {(e) => e.target.style.borderColor = 'rgba(99,102,241,0.6)'}
                onBlur  = {(e) => e.target.style.borderColor = 'rgba(71, 85, 105, 0.5)'}
              />
            </div>
          </div>

          {/* Password field */}
          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{
              display      : 'block',
              color        : '#94a3b8',
              fontSize     : '0.85rem',
              marginBottom : '0.5rem',
              fontWeight   : 500,
            }}>
              Password
            </label>
            <div style={{ position: 'relative' }}>
              <Lock size={16} style={{
                position  : 'absolute',
                left      : '14px',
                top       : '50%',
                transform : 'translateY(-50%)',
                color     : '#475569',
              }} />
              <input
                type        = {showPass ? 'text' : 'password'}
                value       = {password}
                onChange    = {(e) => setPassword(e.target.value)}
                placeholder = "Enter your password"
                autoComplete= "current-password"
                style={{
                  width          : '100%',
                  background     : 'rgba(30, 41, 59, 0.8)',
                  border         : '1px solid rgba(71, 85, 105, 0.5)',
                  borderRadius   : '8px',
                  padding        : '0.75rem 2.5rem 0.75rem 2.5rem',
                  color          : '#f1f5f9',
                  fontSize       : '0.95rem',
                  outline        : 'none',
                  boxSizing      : 'border-box',
                  transition     : 'border-color 0.2s',
                }}
                onFocus = {(e) => e.target.style.borderColor = 'rgba(99,102,241,0.6)'}
                onBlur  = {(e) => e.target.style.borderColor = 'rgba(71, 85, 105, 0.5)'}
              />
              <button
                type    = "button"
                onClick = {() => setShowPass(!showPass)}
                style={{
                  position   : 'absolute',
                  right      : '14px',
                  top        : '50%',
                  transform  : 'translateY(-50%)',
                  background : 'none',
                  border     : 'none',
                  color      : '#475569',
                  cursor     : 'pointer',
                  padding    : 0,
                  display    : 'flex',
                }}
              >
                {showPass ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          {/* Error message */}
          {error && (
            <div style={{
              background   : 'rgba(239, 68, 68, 0.1)',
              border       : '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius : '8px',
              padding      : '0.75rem 1rem',
              marginBottom : '1.2rem',
              color        : '#fca5a5',
              fontSize     : '0.875rem',
            }}>
              {error}
            </div>
          )}

          {/* Submit button */}
          <button
            type     = "submit"
            disabled = {loading}
            style={{
              width          : '100%',
              background     : loading
                ? 'rgba(99, 102, 241, 0.5)'
                : 'linear-gradient(135deg, #6366f1, #8b5cf6)',
              border         : 'none',
              borderRadius   : '8px',
              padding        : '0.85rem',
              color          : '#fff',
              fontSize       : '0.95rem',
              fontWeight     : 600,
              cursor         : loading ? 'not-allowed' : 'pointer',
              display        : 'flex',
              alignItems     : 'center',
              justifyContent : 'center',
              gap            : '8px',
              transition     : 'opacity 0.2s',
              boxShadow      : '0 4px 15px rgba(99, 102, 241, 0.3)',
            }}
          >
            {loading ? (
              <>
                <div style={{
                  width        : '16px',
                  height       : '16px',
                  border       : '2px solid rgba(255,255,255,0.3)',
                  borderTop    : '2px solid #fff',
                  borderRadius : '50%',
                  animation    : 'spin 0.8s linear infinite',
                }} />
                Signing in...
              </>
            ) : (
              <>
                <LogIn size={16} />
                Sign In
              </>
            )}
          </button>

          <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
        </form>

        {/* Divider */}
        <div style={{
          borderTop    : '1px solid rgba(71, 85, 105, 0.3)',
          marginTop    : '2rem',
          paddingTop   : '1.5rem',
          textAlign    : 'center',
        }}>
          <p style={{ color: '#475569', fontSize: '0.85rem', margin: 0 }}>
            Have an invite link?{' '}
            <Link to="/signup" style={{ color: '#6366f1', textDecoration: 'none', fontWeight: 500 }}>
              Create your account
            </Link>
          </p>
        </div>

      </div>
    </div>
  )
}
