// Thin API client. In dev, Vite proxies /api to http://localhost:8000.
const BASE = import.meta.env.VITE_API_URL || ''

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed (${res.status})`)
  }
  return res.json()
}

export const predictURL = (url) =>
  request('/api/predict', { method: 'POST', body: JSON.stringify({ url }) })

export const explainURL = (url) =>
  request('/api/explain', { method: 'POST', body: JSON.stringify({ url }) })

export const getMetrics = () => request('/api/metrics')
export const getFeatureImportance = () => request('/api/feature-importance')
