export const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8001'

export function getToken() {
  return sessionStorage.getItem('token')
}

export function setToken(token) {
  sessionStorage.setItem('token', token)
}

export function clearToken() {
  sessionStorage.removeItem('token')
}
