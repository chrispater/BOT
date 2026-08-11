import React, { useState } from 'react'
import axios from 'axios'

export default function AuthScreen({ setToken, setError, error }) {
  const [isLogin, setIsLogin] = useState(true)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const endpoint = isLogin ? '/auth/login' : '/auth/register'
      const res = await axios.post(`/api${endpoint}`, { username, password })
      localStorage.setItem('token', res.data.access_token)
      setToken(res.data.access_token)
    } catch (e) {
      setError(e.response?.data?.detail || 'Authentication failed')
    }
    setLoading(false)
  }

  return (
    <div className="app">
      <header className="header">
        <h1>CryptoBot Pro</h1>
        <p className="subtitle">Autonomous ML-Powered Trading</p>
      </header>
      <div className="content">
        {error && <div className="error-message">{error}</div>}
        <div className="card">
          <h2 className="card-title">{isLogin ? 'Sign In' : 'Create Account'}</h2>
          <form onSubmit={handleSubmit}>
            <div className="input-group">
              <label>Username</label>
              <input type="text" value={username} onChange={e => setUsername(e.target.value)} placeholder="Enter username" required />
            </div>
            <div className="input-group">
              <label>Password</label>
              <input type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="Enter password" required />
            </div>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? 'Please wait...' : (isLogin ? 'Login' : 'Create Account')}
            </button>
          </form>
          <p style={{ textAlign: 'center', marginTop: 18, color: 'var(--text-secondary)', fontSize: 14 }}>
            {isLogin ? "Don't have an account? " : "Already have an account? "}
            <span style={{ color: 'var(--accent)', cursor: 'pointer', fontWeight: 600 }} onClick={() => setIsLogin(!isLogin)}>
              {isLogin ? 'Sign Up' : 'Login'}
            </span>
          </p>
        </div>
      </div>
    </div>
  )
}
