/**
 * src/pages/LoginPage.jsx
 * ────────────────────────
 * Login page — same for all roles.
 * Automatically detects if no super admin exists yet and shows
 * a first-time setup form instead of the normal login form.
 *
 * Flow:
 * - On load: calls /auth/check-setup
 * - If setup not complete → shows "Create Super Admin" form
 * - If setup complete → shows normal login form
 * - After login/setup → redirects to correct dashboard based on role
 */

import { useState, useEffect } from 'react'
import { useNavigate, Link, useSearchParams } from 'react-router-dom'
import {
  Sparkles, Mail, Lock, LogIn, Eye, EyeOff,
  ShieldCheck, Building2, UserPlus,
} from 'lucide-react'
import { useAuth } from '../AuthContext'
import { checkSetup, signupSuperAdmin } from '../api'

const ROLE_HOME = {
  super_admin : '/admin',
  org_admin   : '/org-dashboard',
  member      : '/',
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared styles
// ─────────────────────────────────────────────────────────────────────────────

const inputStyle = {
  width        : '100%',
  background   : 'rgba(30, 41, 59, 0.8)',
  border       : '1px solid rgba(71, 85, 105, 0.5)',
  borderRadius : '8px',
  padding      : '0.75rem 0.75rem 0.75rem 2.5rem',
  color        : '#f1f5f9',
  fontSize     : '0.95rem',
  outline      : 'none',
  boxSizing    : 'border-box',
  transition   : 'border-color 0.2s',
}

const labelStyle = {
  display      : 'block',
  color        : '#94a3b8',
  fontSize     : '0.85rem',
  marginBottom : '0.5rem',
  fontWeight   : 500,
}

const iconStyle = {
  position  : 'absolute',
  left      : '14px',
  top       : '50%',
  transform : 'translateY(-50%)',
  color     : '#475569',
}

const eyeBtnStyle = {
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
}

function PasswordInput({ value, onChange, placeholder, autoComplete }) {
  const [show, setShow] = useState(false)
  return (
    <div style={{ position: 'relative' }}>
      <Lock size={16} style={iconStyle} />
      <input
        type         = {show ? 'text' : 'password'}
        value        = {value}
        onChange     = {onChange}
        placeholder  = {placeholder}
        autoComplete = {autoComplete}
        style        = {{ ...inputStyle, padding: '0.75rem 2.5rem 0.75rem 2.5rem' }}
        onFocus      = {(e) => e.target.style.borderColor = 'rgba(99,102,241,0.6)'}
        onBlur       = {(e) => e.target.style.borderColor = 'rgba(71,85,105,0.5)'}
      />
      <button type="button" onClick={() => setShow(!show)} style={eyeBtnStyle}>
        {show ? <EyeOff size={16} /> : <Eye size={16} />}
      </button>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// First-time setup form
// ─────────────────────────────────────────────────────────────────────────────

function SetupForm({ onSuccess }) {
  const [fullName, setFullName] = useState('')
  const [orgName,  setOrgName]  = useState('')
  const [email,    setEmail]    = useState('')
  const [password, setPassword] = useState('')
  const [confirm,  setConfirm]  = useState('')
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState('')

  const handleSetup = async (e) => {
    e.preventDefault()
    if (!fullName || !orgName || !email || !password) {
      setError('Please fill in all fields.')
      return
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters.')
      return
    }
    if (password !== confirm) {
      setError('Passwords do not match.')
      return
    }
    setLoading(true)
    setError('')
    try {
      const result = await signupSuperAdmin(fullName, email, password, orgName)
      onSuccess(result)
    } catch (err) {
      setError(err.message || 'Setup failed. Please try again.')
    }
    setLoading(false)
  }

  return (
    <>
      {/* Setup badge */}
      <div style={{
        display        : 'flex',
        alignItems     : 'center',
        justifyContent : 'center',
        gap            : '8px',
        background     : 'rgba(99,102,241,0.1)',
        border         : '1px solid rgba(99,102,241,0.3)',
        borderRadius   : '8px',
        padding        : '0.6rem 1rem',
        marginBottom   : '1.5rem',
        color          : '#a5b4fc',
        fontSize       : '0.85rem',
        fontWeight     : 600,
      }}>
        <ShieldCheck size={16} />
        First-time setup — create your Super Admin account
      </div>

      <h1 style={{
        fontSize     : '1.5rem',
        fontWeight   : 700,
        color        : '#f1f5f9',
        marginBottom : '0.4rem',
        textAlign    : 'center',
        marginTop    : 0,
      }}>
        Welcome to VisionX
      </h1>
      <p style={{
        color        : '#64748b',
        textAlign    : 'center',
        marginBottom : '1.75rem',
        fontSize     : '0.875rem',
      }}>
        Set up your organisation and admin account to get started.
        This can only be done once.
      </p>

      <form onSubmit={handleSetup}>

        {/* Full Name */}
        <div style={{ marginBottom: '1.2rem' }}>
          <label style={labelStyle}>Full Name</label>
          <div style={{ position: 'relative' }}>
            <Building2 size={16} style={iconStyle} />
            <input
              type        = "text"
              value       = {fullName}
              onChange    = {(e) => setFullName(e.target.value)}
              placeholder = "Your full name"
              style       = {inputStyle}
              onFocus     = {(e) => e.target.style.borderColor = 'rgba(99,102,241,0.6)'}
              onBlur      = {(e) => e.target.style.borderColor = 'rgba(71,85,105,0.5)'}
            />
          </div>
        </div>

        {/* Organisation name */}
        <div style={{ marginBottom: '1.2rem' }}>
          <label style={labelStyle}>Organisation Name</label>
          <div style={{ position: 'relative' }}>
            <Building2 size={16} style={iconStyle} />
            <input
              type        = "text"
              value       = {orgName}
              onChange    = {(e) => setOrgName(e.target.value)}
              placeholder = "e.g. Netsmartz"
              style       = {inputStyle}
              onFocus     = {(e) => e.target.style.borderColor = 'rgba(99,102,241,0.6)'}
              onBlur      = {(e) => e.target.style.borderColor = 'rgba(71,85,105,0.5)'}
            />
          </div>
        </div>

        {/* Email */}
        <div style={{ marginBottom: '1.2rem' }}>
          <label style={labelStyle}>Admin Email</label>
          <div style={{ position: 'relative' }}>
            <Mail size={16} style={iconStyle} />
            <input
              type        = "email"
              value       = {email}
              onChange    = {(e) => setEmail(e.target.value)}
              placeholder = "admin@yourcompany.com"
              autoComplete= "email"
              style       = {inputStyle}
              onFocus     = {(e) => e.target.style.borderColor = 'rgba(99,102,241,0.6)'}
              onBlur      = {(e) => e.target.style.borderColor = 'rgba(71,85,105,0.5)'}
            />
          </div>
        </div>

        {/* Password */}
        <div style={{ marginBottom: '1.2rem' }}>
          <label style={labelStyle}>Password</label>
          <PasswordInput
            value        = {password}
            onChange     = {(e) => setPassword(e.target.value)}
            placeholder  = "Minimum 8 characters"
            autoComplete = "new-password"
          />
        </div>

        {/* Confirm password */}
        <div style={{ marginBottom: '1.5rem' }}>
          <label style={labelStyle}>Confirm Password</label>
          <PasswordInput
            value        = {confirm}
            onChange     = {(e) => setConfirm(e.target.value)}
            placeholder  = "Re-enter your password"
            autoComplete = "new-password"
          />
          {confirm && confirm !== password && (
            <p style={{ color: '#f87171', fontSize: '0.78rem', marginTop: '0.3rem' }}>
              Passwords do not match.
            </p>
          )}
        </div>

        {/* Error */}
        {error && (
          <div style={{
            background   : 'rgba(239,68,68,0.1)',
            border       : '1px solid rgba(239,68,68,0.3)',
            borderRadius : '8px',
            padding      : '0.75rem 1rem',
            marginBottom : '1.2rem',
            color        : '#fca5a5',
            fontSize     : '0.875rem',
          }}>
            {error}
          </div>
        )}

        {/* Submit */}
        <button
          type     = "submit"
          disabled = {loading}
          style={{
            width          : '100%',
            background     : loading
              ? 'rgba(99,102,241,0.5)'
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
            boxShadow      : '0 4px 15px rgba(99,102,241,0.3)',
          }}
        >
          {loading ? (
            <>
              <div style={{
                width: '16px', height: '16px',
                border: '2px solid rgba(255,255,255,0.3)',
                borderTop: '2px solid #fff',
                borderRadius: '50%',
                animation: 'spin 0.8s linear infinite',
              }} />
              Creating account...
            </>
          ) : (
            <>
              <UserPlus size={16} />
              Create Super Admin Account
            </>
          )}
        </button>

        <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
      </form>
    </>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Normal login form
// ─────────────────────────────────────────────────────────────────────────────

function LoginForm({ onSuccess }) {
  const [email,    setEmail]    = useState('')
  const [password, setPassword] = useState('')
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState('')
  const { login }  = useAuth()
  const [searchParams] = useSearchParams()
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
      const result = await login(email.trim(), password)
      onSuccess(result)
    } catch (err) {
      setError(err.message || 'Login failed. Please check your credentials.')
    }
    setLoading(false)
  }

  return (
    <>
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
          background   : 'rgba(99,102,241,0.1)',
          border       : '1px solid rgba(99,102,241,0.3)',
          borderRadius : '8px',
          padding      : '0.75rem 1rem',
          marginBottom : '1.5rem',
          fontSize     : '0.85rem',
          color        : '#a5b4fc',
        }}>
          You have an invite!{' '}
          <Link to={`/signup?token=${inviteToken}`} style={{ color: '#6366f1', textDecoration: 'underline' }}>
            Click here to create your account instead.
          </Link>
        </div>
      )}

      <form onSubmit={handleLogin}>

        {/* Email */}
        <div style={{ marginBottom: '1.2rem' }}>
          <label style={labelStyle}>Email address</label>
          <div style={{ position: 'relative' }}>
            <Mail size={16} style={iconStyle} />
            <input
              type        = "email"
              value       = {email}
              onChange    = {(e) => setEmail(e.target.value)}
              placeholder = "you@example.com"
              autoComplete= "email"
              style       = {inputStyle}
              onFocus     = {(e) => e.target.style.borderColor = 'rgba(99,102,241,0.6)'}
              onBlur      = {(e) => e.target.style.borderColor = 'rgba(71,85,105,0.5)'}
            />
          </div>
        </div>

        {/* Password */}
        <div style={{ marginBottom: '1.5rem' }}>
          <label style={labelStyle}>Password</label>
          <PasswordInput
            value        = {password}
            onChange     = {(e) => setPassword(e.target.value)}
            placeholder  = "Enter your password"
            autoComplete = "current-password"
          />
        </div>

        {/* Error */}
        {error && (
          <div style={{
            background   : 'rgba(239,68,68,0.1)',
            border       : '1px solid rgba(239,68,68,0.3)',
            borderRadius : '8px',
            padding      : '0.75rem 1rem',
            marginBottom : '1.2rem',
            color        : '#fca5a5',
            fontSize     : '0.875rem',
          }}>
            {error}
          </div>
        )}

        {/* Submit */}
        <button
          type     = "submit"
          disabled = {loading}
          style={{
            width          : '100%',
            background     : loading
              ? 'rgba(99,102,241,0.5)'
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
            boxShadow      : '0 4px 15px rgba(99,102,241,0.3)',
          }}
        >
          {loading ? (
            <>
              <div style={{
                width: '16px', height: '16px',
                border: '2px solid rgba(255,255,255,0.3)',
                borderTop: '2px solid #fff',
                borderRadius: '50%',
                animation: 'spin 0.8s linear infinite',
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

      {/* Invite link */}
      <div style={{
        borderTop  : '1px solid rgba(71,85,105,0.3)',
        marginTop  : '2rem',
        paddingTop : '1.5rem',
        textAlign  : 'center',
      }}>
        <p style={{ color: '#475569', fontSize: '0.85rem', margin: 0 }}>
          Have an invite link?{' '}
          <Link to="/signup" style={{ color: '#6366f1', textDecoration: 'none', fontWeight: 500 }}>
            Create your account
          </Link>
        </p>
      </div>
    </>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Main LoginPage — detects setup state automatically
// ─────────────────────────────────────────────────────────────────────────────

export default function LoginPage() {
  const { user }  = useAuth()
  const navigate  = useNavigate()

  // null = checking, true = setup done, false = first time
  const [setupComplete, setSetupComplete] = useState(null)

  // If already logged in redirect immediately
  useEffect(() => {
    if (user) {
      navigate(ROLE_HOME[user.role] || '/', { replace: true })
    }
  }, [user])

  const [setupError, setSetupError] = useState(null)

  // Check if super admin exists
  useEffect(() => {
    checkSetup()
      .then(done => setSetupComplete(done))
      .catch((err) => {
        console.error("checkSetup error:", err)
        setSetupError(err.message || 'Failed to connect to backend.')
        setSetupComplete(false) // Default to setup screen so user isn't stuck on login
      })
  }, [])

  const handleSuccess = (result) => {
    navigate(ROLE_HOME[result.role] || '/', { replace: true })
  }

  return (
    <div style={{
      minHeight      : '100vh',
      background     : 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)',
      display        : 'flex',
      alignItems     : 'center',
      justifyContent : 'center',
      padding        : '1rem',
      fontFamily     : 'inherit',
    }}>

      {/* Glow blobs */}
      <div style={{
        position     : 'fixed', top: '20%', left: '10%',
        width: '300px', height: '300px',
        background   : 'radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%)',
        borderRadius : '50%', pointerEvents: 'none',
      }} />
      <div style={{
        position     : 'fixed', bottom: '20%', right: '10%',
        width: '250px', height: '250px',
        background   : 'radial-gradient(circle, rgba(139,92,246,0.12) 0%, transparent 70%)',
        borderRadius : '50%', pointerEvents: 'none',
      }} />

      {/* Card */}
      <div style={{
        width          : '100%',
        maxWidth       : setupComplete === false ? '460px' : '420px',
        background     : 'rgba(15,23,42,0.8)',
        backdropFilter : 'blur(20px)',
        border         : '1px solid rgba(99,102,241,0.3)',
        borderRadius   : '16px',
        padding        : '2.5rem',
        boxShadow      : '0 25px 50px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05)',
        position       : 'relative',
        zIndex         : 1,
        transition     : 'max-width 0.3s ease',
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
          <span style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f1f5f9', letterSpacing: '0.5px' }}>
            VisionX
          </span>
        </div>

        {/* Loading state */}
        {setupComplete === null && (
          <div style={{ textAlign: 'center', padding: '2rem 0', color: '#64748b' }}>
            <div style={{
              width: '24px', height: '24px',
              border: '2px solid #334155',
              borderTop: '2px solid #6366f1',
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite',
              margin: '0 auto 1rem',
            }} />
            <p style={{ margin: 0 }}>Loading...</p>
            <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
          </div>
        )}

        {/* First-time setup */}
        {setupComplete === false && (
          <>
            {setupError && (
              <div style={{
                background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
                borderRadius: '8px', padding: '0.75rem', marginBottom: '1rem',
                color: '#fca5a5', fontSize: '0.85rem', textAlign: 'center'
              }}>
                Warning: {setupError}
              </div>
            )}
            <SetupForm onSuccess={handleSuccess} />
          </>
        )}

        {/* Normal login */}
        {setupComplete === true && (
          <LoginForm onSuccess={handleSuccess} />
        )}

      </div>
    </div>
  )
}
