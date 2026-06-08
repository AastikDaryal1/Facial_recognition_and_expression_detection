/**
 * src/api.js
 * ──────────
 * Centralized API utility.
 * All backend calls go through here — no fetch() calls scattered in components.
 * Handles JWT token storage, attachment, and auto-refresh on 401.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || `${window.location.protocol}//${window.location.hostname}:8000`

// ─────────────────────────────────────────────────────────────────────────────
// Token storage (in memory — more secure than localStorage for access tokens)
// Refresh token goes in localStorage so it survives page refresh
// ─────────────────────────────────────────────────────────────────────────────

let _accessToken = null

export function getAccessToken() {
  return _accessToken
}

export function setAccessToken(token) {
  _accessToken = token
}

export function getRefreshToken() {
  return localStorage.getItem('refresh_token')
}

export function setRefreshToken(token) {
  if (token) {
    localStorage.setItem('refresh_token', token)
  } else {
    localStorage.removeItem('refresh_token')
  }
}

export function clearTokens() {
  _accessToken = null
  localStorage.removeItem('refresh_token')
  localStorage.removeItem('user_role')
  localStorage.removeItem('user_email')
}

// ─────────────────────────────────────────────────────────────────────────────
// User info (stored in localStorage so it survives page refresh)
// ─────────────────────────────────────────────────────────────────────────────

export function getUserRole() {
  return localStorage.getItem('user_role')
}

export function setUserInfo(role, email) {
  localStorage.setItem('user_role', role)
  localStorage.setItem('user_email', email)
}

export function getUserEmail() {
  return localStorage.getItem('user_email')
}

// ─────────────────────────────────────────────────────────────────────────────
// JWT decode (reads role/email from token without a library)
// ─────────────────────────────────────────────────────────────────────────────

export function decodeToken(token) {
  try {
    const base64Url = token.split('.')[1]
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/')
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    )
    return JSON.parse(jsonPayload)
  } catch {
    return null
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core fetch wrapper — attaches Bearer token, handles 401 auto-refresh
// ─────────────────────────────────────────────────────────────────────────────

async function apiFetch(path, options = {}, retry = true) {
  const headers = {
    ...(options.headers || {}),
  }

  // Attach access token if available
  if (_accessToken) {
    headers['Authorization'] = `Bearer ${_accessToken}`
  }

  // Don't set Content-Type for FormData (browser sets it with boundary)
  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
  }

  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  })

  // Auto-refresh on 401
  if (res.status === 401 && retry) {
    const refreshed = await tryRefreshToken()
    if (refreshed) {
      return apiFetch(path, options, false) // retry once
    } else {
      clearTokens()
      window.location.href = '/login'
      throw new Error('Session expired. Please log in again.')
    }
  }

  return res
}

// ─────────────────────────────────────────────────────────────────────────────
// Token refresh
// ─────────────────────────────────────────────────────────────────────────────

async function tryRefreshToken() {
  const refreshToken = getRefreshToken()
  if (!refreshToken) return false

  try {
    const res = await fetch(`${API_BASE}/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken }),
    })

    if (!res.ok) return false

    const data = await res.json()
    setAccessToken(data.access_token)
    if (data.refresh_token) setRefreshToken(data.refresh_token)
    return true
  } catch {
    return false
  }
}

/**
 * Attempts to restore a session from stored tokens.
 * Called on app load in AuthContext.
 */
export async function restoreSession() {
  const refreshToken = getRefreshToken()
  const storedRole  = getUserRole()
  const storedEmail = getUserEmail()

  if (!refreshToken || !storedRole || !storedEmail) {
    return null
  }

  const refreshed = await tryRefreshToken()
  if (refreshed) {
    const payload = decodeToken(getAccessToken())
    const role  = payload?.role  || storedRole
    const email = payload?.email || storedEmail
    setUserInfo(role, email)
    return { email, role }
  } else {
    clearTokens()
    return null
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Auth endpoints
// ─────────────────────────────────────────────────────────────────────────────

export async function login(email, password) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })

  const data = await res.json()

  if (!res.ok) {
    throw new Error(data.detail || 'Login failed.')
  }

  // Store tokens
  setAccessToken(data.access_token)
  setRefreshToken(data.refresh_token)

  // Decode role from token
  const payload = decodeToken(data.access_token)
  if (payload) {
    setUserInfo(payload.role, email)
  }

  return { role: payload?.role, email }
}

export async function logout() {
  try {
    await apiFetch('/auth/logout', { method: 'POST' })
  } catch {
    // Even if logout fails on server, clear local tokens
  }
  clearTokens()
}

export async function signupInvite(fullName, email, password, contact, inviteToken) {
  const res = await fetch(`${API_BASE}/auth/signup-invite`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      full_name    : fullName,
      email,
      password,
      contact      : contact || null,
      invite_token : inviteToken,
    }),
  })

  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Signup failed.')

  setAccessToken(data.access_token)
  setRefreshToken(data.refresh_token)

  const payload = decodeToken(data.access_token)
  if (payload) setUserInfo(payload.role, email)

  return { role: payload?.role, email }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model / System endpoints
// ─────────────────────────────────────────────────────────────────────────────

export async function fetchModelInfo() {
  const res = await apiFetch('/model/info')
  if (!res.ok) throw new Error('Failed to fetch model info.')
  return res.json()
}

export async function fetchMetrics() {
  const res = await apiFetch('/metrics')
  if (!res.ok) throw new Error('Failed to fetch metrics.')
  return res.json()
}

// ─────────────────────────────────────────────────────────────────────────────
// Prediction endpoints
// ─────────────────────────────────────────────────────────────────────────────

export async function predictImage(file) {
  const formData = new FormData()
  formData.append('file', file)

  const res = await apiFetch('/predict/image', {
    method: 'POST',
    body: formData,
  })

  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Prediction failed.')
  return data
}

export async function predictBase64(image_b64, filename = 'live.jpg', detection_method = 'Live Feed') {
  const res = await apiFetch('/predict/base64', {
    method: 'POST',
    body: JSON.stringify({ image_b64, filename, detection_method }),
  })

  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Prediction failed.')
  return data
}

// ─────────────────────────────────────────────────────────────────────────────
// Users endpoints
// ─────────────────────────────────────────────────────────────────────────────

export async function fetchUsers() {
  const res = await apiFetch('/users')
  if (!res.ok) throw new Error('Failed to fetch users.')
  return res.json()
}

export async function deactivateUser(userId) {
  const res = await apiFetch(`/users/${userId}/deactivate`, { method: 'PATCH' })
  if (!res.ok) throw new Error('Failed to deactivate user.')
  return res.json()
}

export async function activateUser(userId) {
  const res = await apiFetch(`/users/${userId}/activate`, { method: 'PATCH' })
  if (!res.ok) throw new Error('Failed to activate user.')
  return res.json()
}

export async function changeUserRole(userId, role) {
  const res = await apiFetch(`/users/${userId}/role`, {
    method: 'PATCH',
    body: JSON.stringify({ role }),
  })
  if (!res.ok) throw new Error('Failed to change role.')
  return res.json()
}

export async function deleteUser(userId) {
  const res = await apiFetch(`/users/${userId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to delete user.')
}

// ─────────────────────────────────────────────────────────────────────────────
// Invite endpoints
// ─────────────────────────────────────────────────────────────────────────────

export async function inviteUser(email, role, orgId) {
  const res = await apiFetch('/auth/invite', {
    method: 'POST',
    body: JSON.stringify({ email, role, org_id: orgId || null }),
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Invite failed.')
  return data
}

// ─────────────────────────────────────────────────────────────────────────────
// Organisations endpoints
// ─────────────────────────────────────────────────────────────────────────────

export async function fetchOrganisations() {
  const res = await apiFetch('/organisations')
  if (!res.ok) throw new Error('Failed to fetch organisations.')
  return res.json()
}

export async function createOrganisation(name) {
  const res = await apiFetch('/organisations', {
    method: 'POST',
    body: JSON.stringify({ name }),
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Failed to create organisation.')
  return data
}

export async function deleteOrganisation(orgId) {
  const res = await apiFetch(`/organisations/${orgId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to delete organisation.')
}

// ─────────────────────────────────────────────────────────────────────────────
// Persons endpoints
// ─────────────────────────────────────────────────────────────────────────────

export async function fetchPersons() {
  const res = await apiFetch('/persons')
  if (!res.ok) throw new Error('Failed to fetch persons.')
  return res.json()
}

export async function createPerson(fullName, employeeId, department) {
  const res = await apiFetch('/persons', {
    method: 'POST',
    body: JSON.stringify({
      full_name: fullName,
      employee_id: employeeId,
      department,
    }),
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Failed to create person.')
  return data
}

export async function deletePerson(personId) {
  const res = await apiFetch(`/persons/${personId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to delete person.')
}

/**
 * Upload face photos for a person.
 * @param {string} personId
 * @param {File[]} files
 */
export async function uploadPersonPhotos(personId, files) {
  const formData = new FormData()
  for (const file of files) {
    formData.append('files', file)
  }
  // apiFetch sets Content-Type for JSON — for FormData we must NOT set it manually
  const token = getAccessToken()
  const res = await fetch(`${API_BASE}/persons/${personId}/photos`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Failed to upload photos.')
  return data
}

/**
 * Trigger retraining pipeline for a person and mark them as enrolled.
 * @param {string} personId
 */
export async function retrainPerson(personId) {
  const res = await apiFetch(`/persons/${personId}/retrain`, { method: 'POST' })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Retrain failed.')
  return data
}

/**
 * Trigger global model retraining for all enrolled persons.
 */
export async function retrainAll() {
  const res = await apiFetch('/persons/retrain_all', { method: 'POST' })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Global retrain failed.')
  return data
}

// ─────────────────────────────────────────────────────────────────────────────
// Sessions endpoints
// ─────────────────────────────────────────────────────────────────────────────

export async function fetchSessions() {
  const res = await apiFetch('/sessions')
  if (!res.ok) throw new Error('Failed to fetch sessions.')
  return res.json()
}

export async function deleteSession(sessionId) {
  const res = await apiFetch(`/sessions/${sessionId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to delete session.')
}

export async function updateSessionNote(sessionId, note) {
  const res = await apiFetch(`/sessions/${sessionId}/note`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ note }),
  })
  if (!res.ok) throw new Error('Failed to update session note.')
  return res.json()
}

// ─────────────────────────────────────────────────────────────────────────────
// Audit logs endpoints
// ─────────────────────────────────────────────────────────────────────────────

export async function fetchAuditLogs() {
  const res = await apiFetch('/audit')
  if (!res.ok) throw new Error('Failed to fetch audit logs.')
  return res.json()
}

// ─────────────────────────────────────────────────────────────────────────────
// System / Cloud Sync
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Manually trigger a GCS cloud sync + model retrain (super_admin only).
 */
export async function triggerSync() {
  const res = await apiFetch('/system/sync', { method: 'POST' })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Sync failed.')
  return data
}

export async function checkSetup() {
  const res = await fetch(`${API_BASE}/auth/check-setup`)
  const data = await res.json()
  return data.setup_complete
}

export async function signupSuperAdmin(fullName, email, password, orgName) {
  const res = await fetch(`${API_BASE}/auth/signup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ full_name: fullName, email, password, org_name: orgName }),
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Setup failed.')

  setAccessToken(data.access_token)
  setRefreshToken(data.refresh_token)

  const payload = decodeToken(data.access_token)
  if (payload) setUserInfo(payload.role, email)

  return { role: payload?.role, email }
}