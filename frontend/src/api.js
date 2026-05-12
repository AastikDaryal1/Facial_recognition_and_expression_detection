/**
 * src/api.js
 * ──────────────────────────────────────────────────────────────────────────
 * Central API client.
 *
 * - Stores access token in memory (safer than localStorage)
 * - Stores refresh token in localStorage (survives page reload)
 * - Every call automatically attaches Authorization: Bearer <token>
 * - On 401, silently refreshes the token and retries once
 * - On refresh failure, redirects to /login
 *
 * Usage:
 *   import { login, fetchUsers, predictImage } from './api'
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

// ── Token storage ─────────────────────────────────────────────────────────

let _accessToken = null

export const setAccessToken  = (t) => { _accessToken = t }
export const getAccessToken  = ()  => _accessToken
export const clearAccessToken = () => { _accessToken = null }

export const setRefreshToken  = (t) => localStorage.setItem('refresh_token', t)
export const getRefreshToken  = ()  => localStorage.getItem('refresh_token')
export const clearRefreshToken = () => localStorage.removeItem('refresh_token')


// ── Core fetch wrapper ────────────────────────────────────────────────────

async function apiFetch(path, options = {}, retry = true) {
  const headers = {
    ...(options.headers || {}),
  }

  // Don't set Content-Type for FormData (browser sets it with boundary)
  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = headers['Content-Type'] || 'application/json'
  }

  if (_accessToken) {
    headers['Authorization'] = `Bearer ${_accessToken}`
  }

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers })

  // Auto-refresh on 401
  if (res.status === 401 && retry) {
    const refreshed = await tryRefresh()
    if (refreshed) return apiFetch(path, options, false)  // retry once
    // Refresh failed — redirect to login
    clearAccessToken()
    clearRefreshToken()
    window.location.href = '/login'
    return
  }

  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch { /* ignore */ }
    throw new Error(detail)
  }

  // 204 No Content
  if (res.status === 204) return null

  return res.json()
}

async function tryRefresh() {
  const rt = getRefreshToken()
  if (!rt) return false
  try {
    const res = await fetch(`${API_BASE}/auth/refresh?refresh_token=${encodeURIComponent(rt)}`, {
      method: 'POST',
    })
    if (!res.ok) return false
    const data = await res.json()
    setAccessToken(data.access_token)
    setRefreshToken(data.refresh_token)
    return true
  } catch {
    return false
  }
}


// ── Auth ──────────────────────────────────────────────────────────────────

export async function login(email, password) {
  const data = await apiFetch('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  }, false)
  setAccessToken(data.access_token)
  setRefreshToken(data.refresh_token)
  // Decode role from JWT payload (middle part)
  const payload = JSON.parse(atob(data.access_token.split('.')[1]))
  return { email, role: payload.role, org_id: payload.org_id }
}

export async function logout() {
  try {
    await apiFetch('/auth/logout', { method: 'POST' })
  } catch { /* ignore errors on logout */ }
  clearAccessToken()
  clearRefreshToken()
}

export async function signupWithInvite(email, password, inviteToken) {
  return apiFetch('/auth/signup-invite', {
    method: 'POST',
    body: JSON.stringify({ email, password, invite_token: inviteToken }),
  }, false)
}

export async function inviteUser(email, role) {
  return apiFetch('/auth/invite', {
    method: 'POST',
    body: JSON.stringify({ email, role }),
  })
}

// Restore session from stored refresh token (call on app load)
export async function restoreSession() {
  const rt = getRefreshToken()
  if (!rt) return null
  const ok = await tryRefresh()
  if (!ok) { clearRefreshToken(); return null }
  const payload = JSON.parse(atob(_accessToken.split('.')[1]))
  return { email: payload.sub, role: payload.role, org_id: payload.org_id }
}


// ── Prediction ────────────────────────────────────────────────────────────

export async function predictImage(file) {
  const formData = new FormData()
  formData.append('file', file)
  return apiFetch('/predict/image', { method: 'POST', body: formData })
}

export async function predictBase64(image_b64, filename = 'frame.jpg') {
  return apiFetch('/predict/base64', {
    method: 'POST',
    body: JSON.stringify({ image_b64, filename }),
  })
}


// ── Model ─────────────────────────────────────────────────────────────────

export async function fetchModelInfo() {
  return apiFetch('/model/info')
}

export async function fetchMetrics() {
  return apiFetch('/metrics')
}


// ── Users ─────────────────────────────────────────────────────────────────

export async function fetchUsers() {
  return apiFetch('/users')
}

export async function fetchUser(id) {
  return apiFetch(`/users/${id}`)
}

export async function changeUserRole(id, role) {
  return apiFetch(`/users/${id}/role`, {
    method: 'PATCH',
    body: JSON.stringify({ role }),
  })
}

export async function deactivateUser(id) {
  return apiFetch(`/users/${id}/deactivate`, { method: 'PATCH' })
}

export async function activateUser(id) {
  return apiFetch(`/users/${id}/activate`, { method: 'PATCH' })
}

export async function deleteUser(id) {
  return apiFetch(`/users/${id}`, { method: 'DELETE' })
}


// ── Organisations ─────────────────────────────────────────────────────────

export async function fetchOrganisations() {
  return apiFetch('/organisations')
}

export async function fetchOrganisation(id) {
  return apiFetch(`/organisations/${id}`)
}

export async function createOrganisation(name) {
  return apiFetch('/organisations', {
    method: 'POST',
    body: JSON.stringify({ name }),
  })
}


// ── Persons ───────────────────────────────────────────────────────────────

export async function fetchPersons() {
  return apiFetch('/persons')
}

export async function fetchPerson(id) {
  return apiFetch(`/persons/${id}`)
}

export async function createPerson({ full_name, employee_id, department }) {
  return apiFetch('/persons', {
    method: 'POST',
    body: JSON.stringify({ full_name, employee_id, department }),
  })
}

export async function updatePerson(id, updates) {
  return apiFetch(`/persons/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(updates),
  })
}

export async function deletePerson(id) {
  return apiFetch(`/persons/${id}`, { method: 'DELETE' })
}


// ── Sessions ──────────────────────────────────────────────────────────────

export async function fetchSessions(page = 1) {
  return apiFetch(`/sessions?page=${page}`)
}

export async function fetchSession(id) {
  return apiFetch(`/sessions/${id}`)
}

export async function deleteSession(id) {
  return apiFetch(`/sessions/${id}`, { method: 'DELETE' })
}


// ── Audit logs ────────────────────────────────────────────────────────────

export async function fetchAuditLogs(page = 1) {
  return apiFetch(`/audit?page=${page}`)
}
