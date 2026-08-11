import React, { useState, useEffect } from 'react'
import { createApi } from './api'
import AuthScreen from './pages/Auth'
import DashboardPage from './pages/Dashboard'
import TradesPage from './pages/Trades'
import StrategyPage from './pages/Strategy'
import OptimizePage from './pages/Optimize'
import SettingsPage from './pages/Settings'
import AdminPage from './pages/Admin'
import InsightsPage from './pages/Insights'
import { HomeIcon, ChartIcon, BookIcon, SearchIcon, GearIcon, ShieldIcon, PulseIcon } from './components/Icons'
import { regimeLabel, regimeColor } from './utils/format'

function App() {
  const [token, setToken] = useState(localStorage.getItem('token'))
  const [currentPage, setCurrentPage] = useState('dashboard')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [botStatus, setBotStatus] = useState(null)
  const [strategies, setStrategies] = useState(null)
  const [isAdmin, setIsAdmin] = useState(false)

  const api = createApi(token)

  useEffect(() => {
    if (token) {
      fetchBotStatus()
      fetchStrategies()
      checkAdmin()
      const interval = setInterval(fetchBotStatus, 5000)
      return () => clearInterval(interval)
    }
  }, [token])

  useEffect(() => {
    if (error || success) {
      const timer = setTimeout(() => { setError(''); setSuccess('') }, 3000)
      return () => clearTimeout(timer)
    }
  }, [error, success])

  const fetchBotStatus = async () => {
    try {
      const res = await api.get('/bot/status')
      setBotStatus(res.data)
    } catch (e) {
      if (e.response?.status === 401) logout()
    }
  }

  const fetchStrategies = async () => {
    try {
      const res = await api.get('/strategies')
      setStrategies(res.data)
    } catch (e) {}
  }

  const checkAdmin = async () => {
    try {
      await api.get('/admin/users')
      setIsAdmin(true)
    } catch (e) {
      setIsAdmin(false)
    }
  }

  const logout = () => {
    localStorage.removeItem('token')
    setToken(null)
    setBotStatus(null)
    setIsAdmin(false)
  }

  const openInsights = () => setCurrentPage('insights')

  if (!token) {
    return <AuthScreen setToken={setToken} setError={setError} error={error} />
  }

  const headerCoins = botStatus?.selected_coins?.map(c => c.split('/')[0]).join(', ') || 'No coins set'
  const headerLeverage = botStatus?.leverage || '—'

  return (
    <div className="app">
      <header className="header">
        <h1>CryptoBot Pro</h1>
        <p className="subtitle">{headerCoins} · {headerLeverage}x Leverage</p>
        {botStatus && (
          <div className="header-badges">
            {botStatus.market_regime && (
              <span className="regime-badge" style={{ color: regimeColor(botStatus.market_regime), borderColor: regimeColor(botStatus.market_regime), background: `color-mix(in srgb, ${regimeColor(botStatus.market_regime)} 12%, transparent)`, border: `1px solid ${regimeColor(botStatus.market_regime)}` }}>
                {regimeLabel(botStatus.market_regime).toUpperCase()}
              </span>
            )}
            {botStatus.model_type && <span className="model-badge">{botStatus.model_type}</span>}
            {botStatus.signal_engine_active && <span className="signal-badge">SignalEngine</span>}
            {botStatus.mie_gate_enabled && (
              <span
                className="model-badge"
                title={botStatus.mie_any_validated ? 'A validated model is influencing entries' : 'Enabled, still learning — no validated model yet'}
                style={{
                  background: botStatus.mie_any_validated ? 'rgba(0,212,170,0.15)' : 'rgba(245,166,35,0.15)',
                  color: botStatus.mie_any_validated ? '#00d4aa' : '#f5a623',
                  border: `1px solid ${botStatus.mie_any_validated ? '#00d4aa' : '#f5a623'}`,
                }}
              >MIE {botStatus.mie_any_validated ? 'ARMED' : 'LEARNING'}</span>
            )}
            {botStatus.auto_optimize_enabled && (
              <span
                className="model-badge"
                title={
                  botStatus.auto_opt_in_progress ? 'Re-tuning parameters now…'
                  : botStatus.auto_opt_pending ? 'New config queued — applies when flat'
                  : `Self-tuning · last run ${botStatus.hours_since_auto_opt ?? '?'}h ago`
                }
                style={{
                  background: botStatus.auto_opt_in_progress ? 'rgba(74,158,255,0.18)' : 'rgba(0,212,170,0.15)',
                  color: botStatus.auto_opt_in_progress ? '#4a9eff' : '#00d4aa',
                  border: `1px solid ${botStatus.auto_opt_in_progress ? '#4a9eff' : '#00d4aa'}`,
                }}
              >
                {botStatus.auto_opt_in_progress ? 'AUTOPILOT · TUNING'
                  : botStatus.auto_opt_pending ? 'AUTOPILOT · QUEUED'
                  : 'AUTOPILOT'}
              </span>
            )}
          </div>
        )}
      </header>

      <div className="content">
        {error && <div className="error-message">{error}</div>}
        {success && <div className="success-message">{success}</div>}

        {currentPage === 'dashboard' && (
          <DashboardPage botStatus={botStatus} api={api} fetchBotStatus={fetchBotStatus} setError={setError} setSuccess={setSuccess} onOpenInsights={openInsights} />
        )}
        {currentPage === 'insights' && <InsightsPage botStatus={botStatus} api={api} />}
        {currentPage === 'trades' && <TradesPage botStatus={botStatus} api={api} />}
        {currentPage === 'strategy' && <StrategyPage strategies={strategies} api={api} />}
        {currentPage === 'optimize' && <OptimizePage api={api} setError={setError} setSuccess={setSuccess} />}
        {currentPage === 'settings' && (
          <SettingsPage api={api} logout={logout} setError={setError} setSuccess={setSuccess} botStatus={botStatus} onOpenInsights={openInsights} />
        )}
        {currentPage === 'admin' && isAdmin && (
          <AdminPage api={api} setError={setError} setSuccess={setSuccess} />
        )}
      </div>

      <nav className="nav">
        <NavItem icon={<HomeIcon />}   label="Dashboard" active={currentPage === 'dashboard'} onClick={() => setCurrentPage('dashboard')} />
        <NavItem icon={<PulseIcon />}  label="Insights"  active={currentPage === 'insights'}  onClick={() => setCurrentPage('insights')} />
        <NavItem icon={<ChartIcon />}  label="Trades"    active={currentPage === 'trades'}    onClick={() => setCurrentPage('trades')} />
        <NavItem icon={<BookIcon />}   label="Strategy"  active={currentPage === 'strategy'}  onClick={() => setCurrentPage('strategy')} />
        <NavItem icon={<SearchIcon />} label="Optimize"  active={currentPage === 'optimize'}  onClick={() => setCurrentPage('optimize')} />
        <NavItem icon={<GearIcon />}   label="Settings"  active={currentPage === 'settings'}  onClick={() => setCurrentPage('settings')} />
        {isAdmin && (
          <NavItem icon={<ShieldIcon />} label="Admin" active={currentPage === 'admin'} onClick={() => setCurrentPage('admin')} />
        )}
      </nav>
    </div>
  )
}

function NavItem({ icon, label, active, onClick }) {
  return (
    <div className={`nav-item ${active ? 'active' : ''}`} onClick={onClick}>
      {icon}
      <span>{label}</span>
    </div>
  )
}

export default App
