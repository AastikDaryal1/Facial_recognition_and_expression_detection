/**
 * src/pages/SignupInvitePage.jsx
 * ───────────────────────────────
 * Page where invited users create their account.
 * Now includes full_name and contact fields.
 */

import { useState, useEffect } from 'react'
import { useNavigate, useSearchParams, Link } from 'react-router-dom'
import { Sparkles, Mail, Lock, Eye, EyeOff, UserCheck, ShieldCheck, User, Phone } from 'lucide-react'
import { signupInvite, decodeToken } from '../api'
import { useAuth } from '../AuthContext'

const ROLE_LABELS = {
  super_admin : { label: 'Super Admin', color: '#f59e0b' },
  org_admin   : { label: 'Org Admin',   color: '#6366f1' },
  member      : { label: 'Member',      color: '#10b981' },
}

export default function SignupInvitePage() {
  const navigate       = useNavigate()
  const [searchParams] = useSearchParams()
  const { user }       = useAuth()

  const inviteToken = searchParams.get('token') || ''

  const [fullName,    setFullName]    = useState('')
  const [contact,     setContact]     = useState('')
  const [email,       setEmail]       = useState('')
  const [role,        setRole]        = useState('')
  const [password,    setPassword]    = useState('')
  const [confirm,     setConfirm]     = useState('')
  const [showPass,    setShowPass]    = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [loading,     setLoading]     = useState(false)
  const [error,       setError]       = useState('')
  const [tokenError,  setTokenError]  = useState('')

  if (user) {
    const home = { super_admin: '/admin', org_admin: '/org-dashboard', member: '/' }
    navigate(home[user.role] || '/', { replace: true })
    return null
  }

  useEffect(() => {
    if (!inviteToken) { setTokenError('No invite token found. Please use the link from your invitation email.'); return }
    const payload = decodeToken(inviteToken)
    if (!payload) { setTokenError('Invalid invite link. Please ask for a new invitation.'); return }
    const now = Math.floor(Date.now() / 1000)
    if (payload.exp && payload.exp < now) { setTokenError('This invite link has expired. Please ask for a new invitation.'); return }
    if (payload.invite_email) setEmail(payload.invite_email)
    if (payload.invite_role)  setRole(payload.invite_role)
  }, [inviteToken])

  const roleInfo = ROLE_LABELS[role] || null

  const handleSignup = async (e) => {
    e.preventDefault()
    if (!fullName.trim()) { setError('Please enter your full name.'); return }
    if (!password)        { setError('Please enter a password.'); return }
    if (password.length < 8) { setError('Password must be at least 8 characters.'); return }
    if (password !== confirm) { setError('Passwords do not match.'); return }
    setLoading(true); setError('')
    try {
      const result = await signupInvite(fullName.trim(), email, password, contact, inviteToken)
      const home = { super_admin: '/admin', org_admin: '/org-dashboard', member: '/' }
      navigate(home[result.role] || '/', { replace: true })
    } catch (err) { setError(err.message || 'Signup failed. Please try again.') }
    setLoading(false)
  }

  const inp = (extra = {}) => ({ width: '100%', background: 'rgba(30,41,59,0.8)', border: '1px solid rgba(71,85,105,0.5)', borderRadius: '8px', padding: '0.75rem 0.75rem 0.75rem 2.5rem', color: '#f1f5f9', fontSize: '0.95rem', outline: 'none', boxSizing: 'border-box', ...extra })
  const lbl = { display: 'block', color: '#94a3b8', fontSize: '0.85rem', marginBottom: '0.5rem', fontWeight: 500 }
  const ico = { position: 'absolute', left: '14px', top: '50%', transform: 'translateY(-50%)', color: '#475569' }

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem' }}>
      <div style={{ position: 'fixed', top: '20%', left: '10%', width: '300px', height: '300px', background: 'radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%)', borderRadius: '50%', pointerEvents: 'none' }} />
      <div style={{ width: '100%', maxWidth: '440px', background: 'rgba(15,23,42,0.8)', backdropFilter: 'blur(20px)', border: '1px solid rgba(99,102,241,0.3)', borderRadius: '16px', padding: '2.5rem', boxShadow: '0 25px 50px rgba(0,0,0,0.5)', position: 'relative', zIndex: 1 }}>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '2rem', justifyContent: 'center' }}>
          <Sparkles size={22} color="#6366f1" />
          <span style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f1f5f9' }}>VisionX</span>
        </div>

        {tokenError ? (
          <div style={{ textAlign: 'center' }}>
            <div style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '8px', padding: '1.5rem', color: '#fca5a5', marginBottom: '1.5rem' }}>
              <ShieldCheck size={32} style={{ marginBottom: '0.75rem', opacity: 0.7 }} />
              <p style={{ margin: 0, fontSize: '0.9rem' }}>{tokenError}</p>
            </div>
            <Link to="/login" style={{ color: '#6366f1', fontSize: '0.9rem' }}>Back to Login</Link>
          </div>
        ) : (
          <>
            <h1 style={{ fontSize: '1.6rem', fontWeight: 700, color: '#f1f5f9', marginBottom: '0.4rem', textAlign: 'center', marginTop: 0 }}>Create your account</h1>
            <p style={{ color: '#64748b', textAlign: 'center', marginBottom: '1.5rem', fontSize: '0.9rem' }}>You've been invited to join VisionX</p>

            {roleInfo && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', background: `rgba(${roleInfo.color === '#6366f1' ? '99,102,241' : roleInfo.color === '#10b981' ? '16,185,129' : '245,158,11'}, 0.1)`, border: `1px solid ${roleInfo.color}40`, borderRadius: '8px', padding: '0.6rem 1rem', marginBottom: '1.5rem' }}>
                <UserCheck size={16} color={roleInfo.color} />
                <span style={{ color: roleInfo.color, fontSize: '0.85rem', fontWeight: 600 }}>You're being added as: {roleInfo.label}</span>
              </div>
            )}

            <form onSubmit={handleSignup}>

              {/* Full Name */}
              <div style={{ marginBottom: '1.2rem' }}>
                <label style={lbl}>Full Name</label>
                <div style={{ position: 'relative' }}>
                  <User size={16} style={ico} />
                  <input type="text" value={fullName} onChange={e => setFullName(e.target.value)} placeholder="Your full name" style={inp()} onFocus={e => e.target.style.borderColor = 'rgba(99,102,241,0.6)'} onBlur={e => e.target.style.borderColor = 'rgba(71,85,105,0.5)'} />
                </div>
              </div>

              {/* Contact (optional) */}
              <div style={{ marginBottom: '1.2rem' }}>
                <label style={lbl}>Phone Number <span style={{ color: '#475569', fontWeight: 400 }}>(optional)</span></label>
                <div style={{ position: 'relative' }}>
                  <Phone size={16} style={ico} />
                  <input type="tel" value={contact} onChange={e => setContact(e.target.value)} placeholder="+91 98765 43210" style={inp()} onFocus={e => e.target.style.borderColor = 'rgba(99,102,241,0.6)'} onBlur={e => e.target.style.borderColor = 'rgba(71,85,105,0.5)'} />
                </div>
              </div>

              {/* Email — read only */}
              <div style={{ marginBottom: '1.2rem' }}>
                <label style={lbl}>Email address</label>
                <div style={{ position: 'relative' }}>
                  <Mail size={16} style={{ ...ico, color: '#334155' }} />
                  <input type="email" value={email} readOnly style={inp({ background: 'rgba(15,23,42,0.6)', color: '#64748b', cursor: 'not-allowed', border: '1px solid rgba(71,85,105,0.3)' })} />
                </div>
                <p style={{ color: '#475569', fontSize: '0.78rem', marginTop: '0.3rem' }}>Email is set by your invite and cannot be changed.</p>
              </div>

              {/* Password */}
              <div style={{ marginBottom: '1.2rem' }}>
                <label style={lbl}>Create password</label>
                <div style={{ position: 'relative' }}>
                  <Lock size={16} style={ico} />
                  <input type={showPass ? 'text' : 'password'} value={password} onChange={e => setPassword(e.target.value)} placeholder="Minimum 8 characters" style={inp({ padding: '0.75rem 2.5rem 0.75rem 2.5rem' })} onFocus={e => e.target.style.borderColor = 'rgba(99,102,241,0.6)'} onBlur={e => e.target.style.borderColor = 'rgba(71,85,105,0.5)'} />
                  <button type="button" onClick={() => setShowPass(!showPass)} style={{ position: 'absolute', right: '14px', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', color: '#475569', cursor: 'pointer', padding: 0, display: 'flex' }}>{showPass ? <EyeOff size={16} /> : <Eye size={16} />}</button>
                </div>
              </div>

              {/* Confirm */}
              <div style={{ marginBottom: '1.5rem' }}>
                <label style={lbl}>Confirm password</label>
                <div style={{ position: 'relative' }}>
                  <Lock size={16} style={ico} />
                  <input type={showConfirm ? 'text' : 'password'} value={confirm} onChange={e => setConfirm(e.target.value)} placeholder="Re-enter your password" style={inp({ padding: '0.75rem 2.5rem 0.75rem 2.5rem', border: `1px solid ${confirm && confirm !== password ? 'rgba(239,68,68,0.5)' : 'rgba(71,85,105,0.5)'}` })} onFocus={e => e.target.style.borderColor = 'rgba(99,102,241,0.6)'} onBlur={e => e.target.style.borderColor = confirm && confirm !== password ? 'rgba(239,68,68,0.5)' : 'rgba(71,85,105,0.5)'} />
                  <button type="button" onClick={() => setShowConfirm(!showConfirm)} style={{ position: 'absolute', right: '14px', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', color: '#475569', cursor: 'pointer', padding: 0, display: 'flex' }}>{showConfirm ? <EyeOff size={16} /> : <Eye size={16} />}</button>
                </div>
                {confirm && confirm !== password && <p style={{ color: '#f87171', fontSize: '0.78rem', marginTop: '0.3rem' }}>Passwords do not match.</p>}
              </div>

              {error && <div style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '8px', padding: '0.75rem 1rem', marginBottom: '1.2rem', color: '#fca5a5', fontSize: '0.875rem' }}>{error}</div>}

              <button type="submit" disabled={loading} style={{ width: '100%', background: loading ? 'rgba(99,102,241,0.5)' : 'linear-gradient(135deg, #6366f1, #8b5cf6)', border: 'none', borderRadius: '8px', padding: '0.85rem', color: '#fff', fontSize: '0.95rem', fontWeight: 600, cursor: loading ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', boxShadow: '0 4px 15px rgba(99,102,241,0.3)' }}>
                {loading ? (<><div style={{ width: '16px', height: '16px', border: '2px solid rgba(255,255,255,0.3)', borderTop: '2px solid #fff', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />Creating account...</>) : (<><UserCheck size={16} />Create Account</>)}
              </button>
              <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
            </form>

            <div style={{ borderTop: '1px solid rgba(71,85,105,0.3)', marginTop: '2rem', paddingTop: '1.5rem', textAlign: 'center' }}>
              <p style={{ color: '#475569', fontSize: '0.85rem', margin: 0 }}>
                Already have an account?{' '}<Link to="/login" style={{ color: '#6366f1', textDecoration: 'none', fontWeight: 500 }}>Sign in</Link>
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
