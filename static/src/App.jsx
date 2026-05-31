import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const API_URL = '/api'

function App() {
  const [token, setToken] = useState(localStorage.getItem('token'))
  const [currentPage, setCurrentPage] = useState('dashboard')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [botStatus, setBotStatus] = useState(null)
  const [strategies, setStrategies] = useState(null)
  const [isAdmin, setIsAdmin] = useState(false)

  const api = axios.create({
    baseURL: API_URL,
    headers: token ? { Authorization: `Bearer ${token}` } : {}
  })

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
              <span className={`regime-badge ${botStatus.market_regime}`}>
                {botStatus.market_regime === 'bull' ? 'BULL MKT' : botStatus.market_regime === 'bear' ? 'BEAR MKT' : 'SIDEWAYS'}
              </span>
            )}
            {botStatus.model_type && (
              <span className="model-badge">{botStatus.model_type}</span>
            )}
            {botStatus.signal_engine_active && (
              <span className="signal-badge">SignalEngine</span>
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
          <DashboardPage
            botStatus={botStatus}
            api={api}
            fetchBotStatus={fetchBotStatus}
            setError={setError}
            setSuccess={setSuccess}
          />
        )}
        {currentPage === 'trades' && <TradesPage botStatus={botStatus} api={api} />}
        {currentPage === 'strategy' && <StrategyPage strategies={strategies} api={api} />}
        {currentPage === 'optimize' && (
          <OptimizePage api={api} setError={setError} setSuccess={setSuccess} />
        )}
        {currentPage === 'settings' && (
          <SettingsPage
            api={api}
            logout={logout}
            setError={setError}
            setSuccess={setSuccess}
            botStatus={botStatus}
          />
        )}
        {currentPage === 'admin' && isAdmin && (
          <AdminPage api={api} setError={setError} setSuccess={setSuccess} />
        )}
      </div>

      <nav className="nav">
        <NavItem icon={<HomeIcon />}   label="Dashboard" active={currentPage === 'dashboard'} onClick={() => setCurrentPage('dashboard')} />
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

/* ─────────────────────────── Auth ─────────────────────────── */

function AuthScreen({ setToken, setError, error }) {
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
          <p style={{ textAlign: 'center', marginTop: 18, color: '#8b95a5', fontSize: 14 }}>
            {isLogin ? "Don't have an account? " : "Already have an account? "}
            <span style={{ color: '#e94560', cursor: 'pointer', fontWeight: 600 }} onClick={() => setIsLogin(!isLogin)}>
              {isLogin ? 'Sign Up' : 'Login'}
            </span>
          </p>
        </div>
      </div>
    </div>
  )
}

/* ─────────────────────────── Compound Tracker ──────────────────────────── */

function CompoundTracker({ compound, balance, dynLeverage, maxLeverage }) {
  const levUsed = dynLeverage || maxLeverage
  const levPct = maxLeverage ? Math.round((levUsed / maxLeverage) * 100) : 100

  const fmtDays = (d) => {
    if (d == null) return '—'
    if (d < 1) return '<1 day'
    if (d < 30) return `${Math.round(d)}d`
    if (d < 365) return `${(d / 30).toFixed(1)}mo`
    return `${(d / 365).toFixed(1)}yr`
  }

  // Not enough trades yet
  if (!compound || compound.insufficient_data) {
    const have = compound?.sample_size ?? 0
    return (
      <div className="card" style={{ background: 'linear-gradient(135deg, #0d1117 0%, #0f1a0f 100%)', border: '1px solid rgba(0,212,170,0.2)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          <h2 className="card-title" style={{ margin: 0, color: '#00d4aa' }}>Compound Tracker</h2>
          <span style={{ fontSize: 11, color: '#4a5060', textTransform: 'uppercase', letterSpacing: '1px' }}>{have} / 10 trades</span>
        </div>
        <div style={{ textAlign: 'center', padding: '24px 0', color: '#4a5060', fontSize: 13 }}>
          Collecting data… need {10 - have} more trade{10 - have !== 1 ? 's' : ''} before projecting.
        </div>
      </div>
    )
  }

  // Net-losing or negative avg — show honest state, no milestone table
  if (compound.warning) {
    const roi = compound.avg_trade_roi_pct ?? 0
    const pnl = compound.session_pnl ?? 0
    return (
      <div className="card" style={{ background: 'linear-gradient(135deg, #0d1117 0%, #110a0a 100%)', border: '1px solid rgba(233,69,96,0.25)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          <h2 className="card-title" style={{ margin: 0, color: '#e94560' }}>Compound Tracker</h2>
          <span style={{ fontSize: 11, color: '#4a5060', textTransform: 'uppercase', letterSpacing: '1px' }}>{compound.sample_size} trades sampled</span>
        </div>
        <div style={{ textAlign: 'center', padding: '16px 0', borderBottom: '1px solid rgba(255,255,255,0.06)', marginBottom: 14 }}>
          <div style={{ fontSize: 11, color: '#8b95a5', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: 6 }}>Avg ROI per Trade (last 20)</div>
          <div style={{ fontSize: 42, fontWeight: 900, color: '#e94560', lineHeight: 1 }}>
            {roi >= 0 ? '+' : ''}{roi.toFixed(2)}%
          </div>
          <div style={{ fontSize: 12, color: '#4a5060', marginTop: 6 }}>
            {compound?.trades_per_day?.toFixed(1) || '—'} trades/day
          </div>
        </div>
        <div style={{ background: 'rgba(233,69,96,0.08)', border: '1px solid rgba(233,69,96,0.2)', borderRadius: 10, padding: '12px 14px', fontSize: 12, color: '#e94560', lineHeight: 1.6 }}>
          Session PnL: <strong>{pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}</strong> — bot is net-losing this session.
          No $1M projection shown until the session turns profitable. The autopilot is re-tuning to correct this.
        </div>
        <p style={{ fontSize: 11, color: '#2a3040', textAlign: 'center', margin: '10px 0 0' }}>
          Based on realized trade ROI · 100% reinvestment · No guarantee
        </p>
      </div>
    )
  }

  const roi = compound.avg_trade_roi_pct ?? 0
  const roiColor = roi >= 5 ? '#00d4aa' : roi >= 2 ? '#f5a623' : '#e8eaf0'
  const milestones = [
    { label: '$10K',  trades: compound.trades_to_10k,  days: compound.days_to_10k,  target: 10_000 },
    { label: '$100K', trades: compound.trades_to_100k, days: compound.days_to_100k, target: 100_000 },
    { label: '$1M',   trades: compound.trades_to_1m,   days: compound.days_to_1m,   target: 1_000_000 },
  ]

  return (
    <div className="card" style={{ background: 'linear-gradient(135deg, #0d1117 0%, #0f1a0f 100%)', border: '1px solid rgba(0,212,170,0.2)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
        <h2 className="card-title" style={{ margin: 0, color: '#00d4aa' }}>Compound Tracker</h2>
        <span style={{ fontSize: 11, color: '#4a5060', textTransform: 'uppercase', letterSpacing: '1px' }}>
          {compound.sample_size || 0} trades sampled
        </span>
      </div>

      <div style={{ textAlign: 'center', padding: '16px 0 14px', borderBottom: '1px solid rgba(255,255,255,0.06)', marginBottom: 16 }}>
        <div style={{ fontSize: 11, color: '#8b95a5', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: 6 }}>
          Avg ROI per Trade (last 20)
        </div>
        <div style={{ fontSize: 42, fontWeight: 900, color: roiColor, lineHeight: 1 }}>
          {roi >= 0 ? '+' : ''}{roi.toFixed(2)}%
        </div>
        <div style={{ fontSize: 12, color: '#4a5060', marginTop: 6, display: 'flex', justifyContent: 'center', gap: 12 }}>
          <span>{compound.trades_per_day?.toFixed(1) || '—'} trades/day</span>
          {levUsed && <span style={{ color: levPct >= 90 ? '#00d4aa' : levPct >= 70 ? '#f5a623' : '#8b95a5' }}>
            last entry: {levUsed}x lev ({levPct}% of max)
          </span>}
        </div>
      </div>

      <div style={{ marginBottom: 8 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '80px 1fr 1fr', gap: '6px 0', fontSize: 11, color: '#4a5060', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '1px', paddingBottom: 8, borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
          <span>Target</span>
          <span style={{ textAlign: 'center' }}>Trades</span>
          <span style={{ textAlign: 'right' }}>ETA</span>
        </div>
        {milestones.map(({ label, trades, days, target }) => {
          const reached = balance >= target
          return (
            <div key={label} style={{ display: 'grid', gridTemplateColumns: '80px 1fr 1fr', alignItems: 'center', padding: '10px 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
              <span style={{ fontWeight: 800, fontSize: 15, color: reached ? '#00d4aa' : '#e8eaf0' }}>
                {reached ? '✓ ' : ''}{label}
              </span>
              <span style={{ textAlign: 'center', fontWeight: 700, fontSize: 15, color: reached ? '#00d4aa' : trades != null ? '#f5a623' : '#4a5060' }}>
                {reached ? 'Done' : trades != null ? trades.toLocaleString() : '—'}
              </span>
              <span style={{ textAlign: 'right', fontSize: 13, color: reached ? '#00d4aa' : days != null ? '#8b95a5' : '#4a5060' }}>
                {reached ? '' : fmtDays(days)}
              </span>
            </div>
          )
        })}
      </div>

      <p style={{ fontSize: 11, color: '#2a3040', textAlign: 'center', margin: '8px 0 0' }}>
        Based on realized trade ROI · 100% reinvestment · No guarantee
      </p>
    </div>
  )
}

/* ─────────────────────────── Dashboard ─────────────────────────── */

function DashboardPage({ botStatus, api, fetchBotStatus, setError, setSuccess }) {
  const [loading, setLoading] = useState(false)
  const [manualSymbol, setManualSymbol] = useState('')
  const [manualSide, setManualSide] = useState('long')
  const [manualLoading, setManualLoading] = useState(false)

  const selectedCoinsForEntry = botStatus?.selected_coins || []

  const manualEnter = async () => {
    const sym = manualSymbol || selectedCoinsForEntry[0]
    if (!sym) return
    setManualLoading(true)
    try {
      const res = await api.post('/bot/manual-enter', { symbol: sym, side: manualSide })
      setSuccess(`Opened ${manualSide.toUpperCase()} ${sym.split('/')[0]} @ $${res.data.price?.toLocaleString()}`)
      fetchBotStatus()
    } catch (e) {
      setError(e.response?.data?.detail || 'Manual entry failed')
    }
    setManualLoading(false)
  }

  const manualExit = async (symbol) => {
    setManualLoading(true)
    try {
      const res = await api.post('/bot/manual-exit', { symbol })
      const pnl = res.data.pnl
      setSuccess(`Closed ${symbol.split('/')[0]} @ $${res.data.price?.toLocaleString()} — PnL: ${pnl >= 0 ? '+' : ''}$${pnl?.toFixed(2)}`)
      fetchBotStatus()
    } catch (e) {
      setError(e.response?.data?.detail || 'Manual exit failed')
    }
    setManualLoading(false)
  }

  const startBot = async () => {
    setLoading(true)
    try {
      const res = await api.post('/bot/start')
      setSuccess(`Bot started in ${res.data.simulation_mode ? 'simulation' : 'live'} mode`)
      fetchBotStatus()
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to start bot')
    }
    setLoading(false)
  }

  const stopBot = async () => {
    setLoading(true)
    try {
      await api.post('/bot/stop')
      setSuccess('Bot stopped')
      fetchBotStatus()
    } catch (e) {
      setError('Failed to stop bot')
    }
    setLoading(false)
  }

  const coinSignals = botStatus?.coin_signals || {}
  const selectedCoins = botStatus?.selected_coins || []
  const openPositions = botStatus?.positions || (botStatus?.position ? [botStatus.position] : [])

  return (
    <>
      {/* Status + Control */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h2 className="card-title" style={{ margin: 0 }}>Bot Status</h2>
          {botStatus?.running ? (
            <span className={`status-badge ${botStatus.simulation_mode ? 'simulation' : 'running'}`}>
              <span className={`pulse ${botStatus.simulation_mode ? 'yellow' : 'green'}`}></span>
              {botStatus.simulation_mode ? 'Simulation' : 'Live'}
            </span>
          ) : (
            <span className="status-badge stopped">
              <span className="pulse red"></span>
              Stopped
            </span>
          )}
        </div>

        {botStatus?.running ? (
          <button className="btn btn-danger" onClick={stopBot} disabled={loading}>
            {loading ? 'Stopping...' : 'Stop Bot'}
          </button>
        ) : (
          <button className="btn btn-success" onClick={startBot} disabled={loading}>
            {loading ? 'Starting...' : 'Start Bot'}
          </button>
        )}
      </div>

      {/* Performance */}
      <div className="card">
        <h2 className="card-title">Performance</h2>
        <div className="stat-grid">
          <div className="stat-item">
            <div className="stat-value">${(botStatus?.balance || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
            <div className="stat-label">Balance</div>
          </div>
          <div className="stat-item">
            <div className={`stat-value ${(botStatus?.total_pnl || 0) >= 0 ? 'positive' : 'negative'}`}>
              {(botStatus?.total_pnl || 0) >= 0 ? '+' : ''}${(botStatus?.total_pnl || 0).toFixed(2)}
            </div>
            <div className="stat-label">Total PnL</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">{botStatus?.total_trades || 0}</div>
            <div className="stat-label">Total Trades</div>
          </div>
          <div className="stat-item">
            <div className={`stat-value ${(botStatus?.win_rate || 0) >= 50 ? 'positive' : 'negative'}`}>
              {(botStatus?.win_rate || 0).toFixed(1)}%
            </div>
            <div className="stat-label">Win Rate</div>
          </div>
        </div>

        {/* Kelly fraction + streak row */}
        {botStatus?.kelly_fraction != null && (
          <div style={{ marginTop: 12, padding: '10px 12px', background: '#0f1318', borderRadius: 10, border: '1px solid rgba(255,255,255,0.07)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 13, color: '#8b95a5' }}>Kelly Fraction</span>
            <span style={{ fontSize: 14, fontWeight: 700, color: '#f5a623' }}>
              {(botStatus.kelly_fraction * 100).toFixed(1)}%
            </span>
          </div>
        )}
        {(() => {
          const closed = (botStatus?.recent_trades || []).filter(t => t.type === 'close' && t.pnl !== undefined)
          if (closed.length < 2) return null
          const isWin = closed[closed.length - 1].pnl >= 0
          let streak = 1
          for (let i = closed.length - 2; i >= 0; i--) {
            if ((closed[i].pnl >= 0) === isWin) streak++
            else break
          }
          const color = isWin ? '#00d4aa' : '#ff4444'
          return (
            <div style={{ marginTop: 8, padding: '10px 12px', background: '#0f1318', borderRadius: 10, border: `1px solid ${isWin ? 'rgba(0,212,170,0.2)' : 'rgba(255,68,68,0.2)'}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 13, color: '#8b95a5' }}>{isWin ? 'Win Streak' : 'Loss Streak'}</span>
              <span style={{ fontSize: 14, fontWeight: 700, color }}>{streak}x {isWin ? 'WIN' : 'LOSS'}</span>
            </div>
          )
        })()}
      </div>

      {/* Compound Tracker */}
      {botStatus?.compound && (
        <CompoundTracker compound={botStatus.compound} balance={botStatus.balance} dynLeverage={botStatus.last_dynamic_leverage} maxLeverage={botStatus.leverage} />
      )}

      {/* Coin Signals */}
      {botStatus?.running && selectedCoins.length > 0 && (
        <div className="card">
          <h2 className="card-title">Live Signals</h2>
          {selectedCoins.map(coinSymbol => {
            const coin = coinSymbol.split('/')[0]
            const signal = coinSignals[coin]
            const conf = signal?.confidence || 0
            const confColor = conf >= 0.70 ? '#00d4aa' : conf >= 0.65 ? '#f5a623' : '#e94560'

            return (
              <div key={coin} style={{
                padding: '12px',
                marginBottom: 8,
                background: '#0f1318',
                borderRadius: 10,
                border: '1px solid rgba(255,255,255,0.07)'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 700, color: '#e94560', fontSize: 15 }}>{coin}</span>
                  {signal ? (
                    <span style={{
                      padding: '4px 12px',
                      borderRadius: 20,
                      fontSize: 12,
                      fontWeight: 700,
                      background: signal.signal === 1 ? 'rgba(0,212,170,0.15)' : signal.signal === -1 ? 'rgba(255,68,68,0.15)' : 'rgba(255,255,255,0.06)',
                      color: signal.signal === 1 ? '#00d4aa' : signal.signal === -1 ? '#ff4444' : '#8b95a5',
                      border: `1px solid ${signal.signal === 1 ? 'rgba(0,212,170,0.3)' : signal.signal === -1 ? 'rgba(255,68,68,0.3)' : 'transparent'}`
                    }}>
                      {signal.signal === 1 ? 'LONG' : signal.signal === -1 ? 'SHORT' : 'HOLD'}
                    </span>
                  ) : (
                    <span style={{ color: '#4a5060', fontSize: 12 }}>Waiting...</span>
                  )}
                </div>
                {signal && (
                  <>
                    <div style={{ display: 'flex', gap: 14, fontSize: 12, color: '#8b95a5', marginTop: 6 }}>
                      <span>Conf: <strong style={{ color: confColor }}>{(conf * 100).toFixed(1)}%</strong></span>
                      <span>Price: <strong style={{ color: '#e8eaf0' }}>${signal.price?.toLocaleString()}</strong></span>
                      {signal.rsi && <span>RSI: <strong style={{ color: '#e8eaf0' }}>{signal.rsi?.toFixed(1)}</strong></span>}
                    </div>
                    <div className="confidence-bar-bg">
                      <div className="confidence-bar-fill" style={{ width: `${(conf * 100).toFixed(0)}%`, background: confColor }} />
                    </div>
                  </>
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* Open Positions */}
      {openPositions.length > 0 && (
        <div className="card">
          <h2 className="card-title">Open Positions ({openPositions.length})</h2>
          {openPositions.map((pos, idx) => {
            const coin = pos.symbol ? pos.symbol.split('/')[0] : 'Unknown'
            const isLong = pos.side === 'long'
            return (
              <div key={idx} style={{
                padding: '12px',
                marginBottom: idx < openPositions.length - 1 ? 8 : 0,
                background: '#0f1318',
                borderRadius: 10,
                border: `1px solid ${isLong ? 'rgba(0,212,170,0.25)' : 'rgba(255,68,68,0.25)'}`
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontWeight: 700, color: '#e94560', fontSize: 15 }}>{coin}</span>
                  <span style={{
                    padding: '3px 12px',
                    borderRadius: 20,
                    fontSize: 12,
                    fontWeight: 700,
                    background: isLong ? 'rgba(0,212,170,0.15)' : 'rgba(255,68,68,0.15)',
                    color: isLong ? '#00d4aa' : '#ff4444',
                    border: `1px solid ${isLong ? 'rgba(0,212,170,0.3)' : 'rgba(255,68,68,0.3)'}`,
                    textTransform: 'uppercase'
                  }}>
                    {pos.side}
                  </span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 12, color: '#8b95a5' }}>
                  <span>Entry: <strong style={{ color: '#e8eaf0' }}>${pos.entry_price?.toLocaleString()}</strong></span>
                  <span>Size: <strong style={{ color: '#e8eaf0' }}>{pos.size?.toFixed(4)}</strong></span>
                  <span>Margin: <strong style={{ color: '#e8eaf0' }}>${pos.margin?.toFixed(2)}</strong></span>
                  <span>Leverage: <strong style={{ color: '#f5a623' }}>{pos.leverage || botStatus?.leverage}x</strong></span>
                  {isLong && pos.high_water_mark && (
                    <span>Peak: <strong style={{ color: '#00d4aa' }}>${pos.high_water_mark?.toLocaleString()}</strong></span>
                  )}
                  {!isLong && pos.low_water_mark && (
                    <span>Low: <strong style={{ color: '#ff4444' }}>${pos.low_water_mark?.toLocaleString()}</strong></span>
                  )}
                </div>
                {pos.conf_tier && (
                  <div style={{ marginTop: 6, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{
                      padding: '2px 8px', borderRadius: 10, fontSize: 10, fontWeight: 700,
                      background: pos.conf_tier === 'NO-BRAINER' ? 'rgba(0,212,170,0.2)' : pos.conf_tier === 'STRONG' ? 'rgba(245,166,35,0.2)' : 'rgba(255,255,255,0.08)',
                      color: pos.conf_tier === 'NO-BRAINER' ? '#00d4aa' : pos.conf_tier === 'STRONG' ? '#f5a623' : '#8b95a5',
                    }}>{pos.conf_tier}</span>
                    <button
                      onClick={() => manualExit(pos.symbol)}
                      disabled={manualLoading}
                      style={{
                        padding: '4px 14px', borderRadius: 8, border: '1px solid rgba(233,69,96,0.5)',
                        background: 'rgba(233,69,96,0.12)', color: '#e94560', fontSize: 11,
                        fontWeight: 700, cursor: 'pointer', letterSpacing: 1,
                      }}
                    >EXIT</button>
                  </div>
                )}
                {!pos.conf_tier && (
                  <div style={{ marginTop: 8, display: 'flex', justifyContent: 'flex-end' }}>
                    <button
                      onClick={() => manualExit(pos.symbol)}
                      disabled={manualLoading}
                      style={{
                        padding: '4px 14px', borderRadius: 8, border: '1px solid rgba(233,69,96,0.5)',
                        background: 'rgba(233,69,96,0.12)', color: '#e94560', fontSize: 11,
                        fontWeight: 700, cursor: 'pointer', letterSpacing: 1,
                      }}
                    >EXIT</button>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* Manual Trade */}
      <div className="card">
        <h2 className="card-title">Manual Trade</h2>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', opacity: botStatus?.running ? 1 : 0.5 }}>
          <select
            value={manualSymbol || selectedCoinsForEntry[0] || ''}
            onChange={e => setManualSymbol(e.target.value)}
            disabled={!botStatus?.running}
            style={{
              flex: 1, minWidth: 140, padding: '8px 10px', borderRadius: 8,
              background: '#0f1318', border: '1px solid #2a3040', color: '#e8eaf0', fontSize: 13,
            }}
          >
            {selectedCoinsForEntry.length === 0 && <option value="">No coins selected</option>}
            {selectedCoinsForEntry.map(c => (
              <option key={c} value={c}>{c.split('/')[0]}</option>
            ))}
          </select>
          <div style={{ display: 'flex', borderRadius: 8, overflow: 'hidden', border: '1px solid #2a3040' }}>
            {['long', 'short'].map(s => (
              <button
                key={s}
                onClick={() => setManualSide(s)}
                disabled={!botStatus?.running}
                style={{
                  padding: '8px 18px', border: 'none', fontSize: 12, fontWeight: 700,
                  cursor: botStatus?.running ? 'pointer' : 'not-allowed', letterSpacing: 1, textTransform: 'uppercase',
                  background: manualSide === s
                    ? (s === 'long' ? 'rgba(0,212,170,0.25)' : 'rgba(233,69,96,0.25)')
                    : '#0f1318',
                  color: manualSide === s
                    ? (s === 'long' ? '#00d4aa' : '#e94560')
                    : '#4a5060',
                }}
              >{s}</button>
            ))}
          </div>
          <button
            onClick={manualEnter}
            disabled={manualLoading || !botStatus?.running || selectedCoinsForEntry.length === 0}
            style={{
              padding: '8px 22px', borderRadius: 8, border: 'none', fontSize: 13,
              fontWeight: 700, cursor: botStatus?.running ? 'pointer' : 'not-allowed', letterSpacing: 1,
              background: manualSide === 'long' ? 'rgba(0,212,170,0.2)' : 'rgba(233,69,96,0.2)',
              color: manualSide === 'long' ? '#00d4aa' : '#e94560',
            }}
          >{manualLoading ? '...' : 'ENTER'}</button>
        </div>
        <p style={{ fontSize: 11, color: '#4a5060', marginTop: 8 }}>
          {botStatus?.running
            ? 'Bot resumes auto-trading after the position closes.'
            : 'Start the bot to enable manual entry. The bot then manages the position and resumes looking for the next trade.'}
        </p>
      </div>
    </>
  )
}

/* ─────────────────────────── Trades ─────────────────────────── */

function TradesPage({ botStatus, api }) {
  const [dbTrades, setDbTrades] = useState([])
  useEffect(() => {
    api.get('/trades').then(r => setDbTrades(r.data?.trades || [])).catch(() => {})
  }, [])
  const trades = botStatus?.recent_trades?.length ? botStatus.recent_trades : dbTrades

  return (
    <div className="card">
      <h2 className="card-title">Recent Trades</h2>
      {trades.length === 0 ? (
        <p style={{ color: '#8b95a5', textAlign: 'center', padding: 24 }}>
          No trades yet. Start the bot to begin trading.
        </p>
      ) : (
        <div className="trade-list">
          {trades.slice().reverse().map((trade, i) => (
            <div key={i} className="trade-item">
              <div>
                <div className={`trade-side ${trade.side}`}>
                  {trade.type?.toUpperCase()} {trade.side?.toUpperCase()}
                </div>
                <div style={{ fontSize: 12, color: '#8b95a5', marginTop: 2 }}>
                  {trade.symbol ? trade.symbol.split('/')[0] : ''} · ${trade.price?.toLocaleString()}
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                {trade.pnl !== undefined && (
                  <div className={`trade-pnl ${trade.pnl >= 0 ? 'positive' : 'negative'}`}>
                    {trade.pnl >= 0 ? '+' : ''}${trade.pnl?.toFixed(2)}
                  </div>
                )}
                <div style={{ fontSize: 12, color: '#8b95a5', marginTop: 2 }}>
                  {trade.reason
                    ? trade.reason
                    : trade.conf_tier
                      ? `${trade.conf_tier} · ${trade.leverage || ''}x`
                      : trade.confidence
                        ? (trade.confidence * 100).toFixed(0) + '% conf'
                        : ''}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ─────────────────────────── Strategy / Backtest ─────────────────────────── */

function GrowthProjection({ periodDays, totalReturn, monthlyRoi }) {
  const mr = monthlyRoi != null ? monthlyRoi : (totalReturn / Math.max(periodDays, 1)) * 30
  const projections = [
    { label: '3mo',  months: 3 },
    { label: '6mo',  months: 6 },
    { label: '1yr',  months: 12 },
    { label: '2yr',  months: 24 },
    { label: '3yr',  months: 36 },
  ]
  const milestones = [
    { label: '$10K',   target: 10000 },
    { label: '$100K',  target: 100000 },
    { label: '$1M',    target: 1000000 },
  ]
  const timeToMilestone = (target) => {
    if (mr <= 0) return null
    const months = Math.log(target / 1000) / Math.log(1 + mr / 100)
    const m = Math.ceil(months)
    return m > 0 && m < 600 ? m : null
  }
  const fmtMonths = (m) => m >= 24 ? `${(m / 12).toFixed(1)} yrs` : `${m} mo`

  return (
    <div className="card">
      <h2 className="card-title">Compound Growth Projection</h2>
      <p style={{ fontSize: 12, color: '#8b95a5', marginBottom: 14 }}>
        At <strong style={{ color: '#f5a623' }}>{mr.toFixed(2)}% / month</strong> compounded · Starting capital: $1,000
      </p>

      <div className="growth-table" style={{ marginBottom: 16 }}>
        {projections.map(({ label, months }) => {
          const value = 1000 * Math.pow(1 + mr / 100, months)
          return (
            <div key={months} className="growth-row">
              <span style={{ color: '#8b95a5', fontSize: 13 }}>{label}</span>
              <span style={{ color: '#f5a623', fontWeight: 700, fontSize: 15 }}>
                ${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </span>
            </div>
          )
        })}
      </div>

      <div style={{ paddingTop: 14, borderTop: '1px solid rgba(255,255,255,0.07)' }}>
        <div style={{ fontSize: 11, color: '#8b95a5', marginBottom: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px' }}>
          Time to Milestone (from $1K)
        </div>
        {milestones.map(({ label, target }) => {
          const months = timeToMilestone(target)
          return (
            <div key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
              <span style={{ fontWeight: 700, color: '#e8eaf0', fontSize: 15 }}>{label}</span>
              {months ? (
                <span style={{ color: '#00d4aa', fontWeight: 700, fontSize: 14 }}>{fmtMonths(months)}</span>
              ) : (
                <span style={{ color: '#4a5060', fontSize: 13 }}>N/A at this rate</span>
              )}
            </div>
          )
        })}
      </div>

      <p style={{ fontSize: 11, color: '#4a5060', marginTop: 12, textAlign: 'center' }}>
        Compound interest math, not a guarantee. Past backtest ≠ future results.
      </p>
    </div>
  )
}

function MarketDirectionScanner({ api }) {
  const [scan, setScan] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const runScan = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await api.get('/market/direction')
      setScan(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to scan market direction')
    }
    setLoading(false)
  }

  const dirColor = (label) => {
    if (!label) return '#8b95a5'
    if (label.includes('BULL')) return '#00d4aa'
    if (label.includes('BEAR')) return '#e94560'
    return '#f5a623'
  }
  const regimeColor = { bull: '#00d4aa', bear: '#e94560', sideways: '#f5a623' }

  return (
    <div className="card">
      <h2 className="card-title">Market Direction Scanner</h2>
      <p style={{ color: '#8b95a5', fontSize: 14, marginBottom: 16, lineHeight: 1.5 }}>
        Multi-timeframe (15m · 1h · 4h) trend, momentum and strength read for each selected token.
        Use it to see where each token is heading before you optimize, backtest, and go live.
      </p>
      <button className="btn btn-primary" onClick={runScan} disabled={loading}>
        {loading ? 'Scanning...' : 'Scan Market Direction'}
      </button>

      {error && <div className="error-message" style={{ marginTop: 16 }}>{error}</div>}

      {scan && (
        <div style={{ marginTop: 16 }}>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'center', padding: '10px 14px', background: '#0f1318', borderRadius: 10, border: '1px solid rgba(255,255,255,0.07)', marginBottom: 16 }}>
            <span style={{ fontSize: 13, color: '#8b95a5' }}>
              BTC regime:{' '}
              <strong style={{ color: regimeColor[scan.market_regime] || '#8b95a5', textTransform: 'uppercase' }}>
                {scan.market_regime}
              </strong>
            </span>
            <span style={{ fontSize: 13, color: '#00d4aa' }}>▲ {scan.summary?.bullish} bullish</span>
            <span style={{ fontSize: 13, color: '#e94560' }}>▼ {scan.summary?.bearish} bearish</span>
            <span style={{ fontSize: 13, color: '#f5a623' }}>● {scan.summary?.neutral} neutral</span>
            <span style={{ fontSize: 12, color: '#4a5060', marginLeft: 'auto' }}>ADX gate: {scan.adx_threshold}</span>
          </div>

          {scan.tokens?.map((t) => (
            <div key={t.symbol} style={{ padding: '12px 14px', background: '#0f1318', borderRadius: 10, border: '1px solid rgba(255,255,255,0.07)', marginBottom: 10 }}>
              {t.error ? (
                <div style={{ fontSize: 14 }}>
                  <strong style={{ color: '#e8eaf0' }}>{t.symbol.split('/')[0]}</strong>
                  <span style={{ color: '#4a5060', marginLeft: 10 }}>no data</span>
                </div>
              ) : (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 10, marginBottom: 10 }}>
                    <strong style={{ color: '#e8eaf0', fontSize: 16 }}>{t.symbol.split('/')[0]}</strong>
                    <span style={{ fontSize: 12, fontWeight: 700, color: dirColor(t.label), border: `1px solid ${dirColor(t.label)}`, borderRadius: 6, padding: '2px 8px' }}>
                      {t.label}
                    </span>
                    {t.aligned && (
                      <span style={{ fontSize: 11, color: '#00d4aa' }}>✓ all TFs aligned</span>
                    )}
                    <span style={{ marginLeft: 'auto', fontSize: 12, color: t.tradeable ? '#00d4aa' : '#8b95a5' }}>
                      {t.recommended_bias}
                    </span>
                  </div>

                  <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', fontSize: 12, color: '#8b95a5', marginBottom: 10 }}>
                    <span>Conviction: <strong style={{ color: dirColor(t.label) }}>{(t.conviction * 100).toFixed(0)}%</strong></span>
                    <span>Avg ADX: <strong style={{ color: '#e8eaf0' }}>{t.avg_adx}</strong> ({t.trend_strength})</span>
                  </div>

                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {t.timeframes?.map((tf) => (
                      <div key={tf.timeframe} style={{ flex: '1 1 90px', minWidth: 90, padding: '6px 8px', background: '#161b22', borderRadius: 8, borderLeft: `3px solid ${dirColor(tf.direction.toUpperCase())}` }}>
                        <div style={{ fontSize: 11, color: '#8b95a5', marginBottom: 2 }}>{tf.timeframe}</div>
                        <div style={{ fontSize: 12, fontWeight: 600, color: dirColor(tf.direction.toUpperCase()), textTransform: 'capitalize' }}>{tf.direction}</div>
                        <div style={{ fontSize: 10, color: '#4a5060', marginTop: 2 }}>RSI {tf.rsi} · ADX {tf.adx}</div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          ))}

          <p style={{ fontSize: 11, color: '#4a5060', marginTop: 4, textAlign: 'center' }}>
            Scanned {new Date(scan.scanned_at).toLocaleTimeString()} · directional read only, not a trade signal.
          </p>
        </div>
      )}
    </div>
  )
}

function StrategyPage({ strategies, api }) {
  const [backtestResults, setBacktestResults] = useState(null)
  const [backtestLoading, setBacktestLoading] = useState(false)
  const [backtestError, setBacktestError] = useState('')
  const [expandedCoin, setExpandedCoin] = useState(null)
  const [showAllTrades, setShowAllTrades] = useState(false)

  const runBacktest = async () => {
    setBacktestLoading(true)
    setBacktestError('')
    setExpandedCoin(null)
    setShowAllTrades(false)
    try {
      const res = await api.get('/backtest')
      setBacktestResults(res.data)
    } catch (e) {
      setBacktestError(e.response?.data?.detail || 'Failed to run backtest')
    }
    setBacktestLoading(false)
  }

  if (!strategies) {
    return <div className="loading"><div className="spinner"></div></div>
  }

  return (
    <>
      <MarketDirectionScanner api={api} />

      {/* Backtest Runner */}
      <div className="card">
        <h2 className="card-title">Backtest Strategy</h2>
        <p style={{ color: '#8b95a5', fontSize: 14, marginBottom: 16, lineHeight: 1.5 }}>
          Runs a historical simulation using the full ML pipeline, fees, trailing stops, cooldown, and confidence thresholds — identical to live trading.
        </p>
        <button className="btn btn-primary" onClick={runBacktest} disabled={backtestLoading}>
          {backtestLoading ? 'Running Backtest...' : 'Run Backtest'}
        </button>

        {backtestError && <div className="error-message" style={{ marginTop: 16 }}>{backtestError}</div>}

        {backtestResults && (
          <div style={{ marginTop: 16 }}>
            <div style={{ padding: '10px 14px', background: '#0f1318', borderRadius: 10, border: '1px solid rgba(255,255,255,0.07)', marginBottom: 16 }}>
              <div style={{ fontSize: 12, color: '#8b95a5', marginBottom: 4 }}>
                {backtestResults.leverage}x · {(backtestResults.risk_per_trade * 100).toFixed(1)}% risk ·{' '}
                {(backtestResults.stop_loss_pct * 100).toFixed(1)}% SL · {(backtestResults.take_profit_pct * 100).toFixed(1)}% TP
                {backtestResults.trailing_stop_pct ? ` · ${(backtestResults.trailing_stop_pct * 100).toFixed(1)}% trail` : ''} ·{' '}
                {(backtestResults.min_confidence * 100).toFixed(0)}% conf · {backtestResults.timeframe || '5m'}
              </div>
              <div style={{ fontSize: 12, color: '#e94560', fontWeight: 600 }}>
                {backtestResults.selected_coins?.map(c => c.split('/')[0]).join(', ')}
              </div>
            </div>

            <h3 style={{ fontSize: 13, color: '#8b95a5', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '1px' }}>
              Overall · {backtestResults.period_days} days
            </h3>
            <div className="stat-grid">
              <div className="stat-item">
                <div className={`stat-value ${backtestResults.total_pnl >= 0 ? 'positive' : 'negative'}`}>
                  {backtestResults.total_pnl >= 0 ? '+' : ''}${backtestResults.total_pnl?.toFixed(2)}
                </div>
                <div className="stat-label">Profit/Loss</div>
              </div>
              <div className="stat-item">
                <div className={`stat-value ${backtestResults.total_return >= 0 ? 'positive' : 'negative'}`}>
                  {backtestResults.total_return >= 0 ? '+' : ''}{backtestResults.total_return?.toFixed(2)}%
                </div>
                <div className="stat-label">Return</div>
              </div>
              <div className="stat-item">
                <div className={`stat-value ${backtestResults.win_rate >= 50 ? 'positive' : 'negative'}`}>
                  {backtestResults.win_rate?.toFixed(1)}%
                </div>
                <div className="stat-label">Win Rate</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{backtestResults.total_trades}</div>
                <div className="stat-label">Trades</div>
              </div>
            </div>

            <div style={{ marginTop: 12 }}>
              <div className="indicator-row">
                <span className="indicator-name">Starting Balance</span>
                <span className="indicator-value">${backtestResults.starting_balance?.toLocaleString()}</span>
              </div>
              <div className="indicator-row">
                <span className="indicator-name">Final Balance</span>
                <span className={`indicator-value ${backtestResults.final_balance >= backtestResults.starting_balance ? 'positive' : 'negative'}`}>
                  ${backtestResults.final_balance?.toLocaleString()}
                </span>
              </div>
              <div className="indicator-row">
                <span className="indicator-name">Max Drawdown</span>
                <span className="indicator-value negative">-{backtestResults.max_drawdown?.toFixed(2)}%</span>
              </div>
              <div className="indicator-row">
                <span className="indicator-name">Total Fees (0.06%)</span>
                <span className="indicator-value" style={{ color: '#f5a623' }}>
                  -${backtestResults.total_fees?.toFixed(2) || '0.00'}
                </span>
              </div>
              {backtestResults.monthly_roi > 0 && (
                <div className="indicator-row">
                  <span className="indicator-name">Monthly ROI</span>
                  <span className="indicator-value positive">+{backtestResults.monthly_roi?.toFixed(2)}%</span>
                </div>
              )}
              {backtestResults.calmar_ratio > 0 && (
                <div className="indicator-row">
                  <span className="indicator-name">Calmar Ratio</span>
                  <span className="indicator-value" style={{ color: '#4a9eff' }}>{backtestResults.calmar_ratio?.toFixed(2)}</span>
                </div>
              )}
              {backtestResults.sharpe_ratio > 0 && (
                <div className="indicator-row">
                  <span className="indicator-name">Sharpe Ratio</span>
                  <span className="indicator-value" style={{ color: '#4a9eff' }}>{backtestResults.sharpe_ratio?.toFixed(3)}</span>
                </div>
              )}
              {backtestResults.months_to_1m && (
                <div className="indicator-row">
                  <span className="indicator-name">Time to $1M (from $1K)</span>
                  <span className="indicator-value" style={{ color: '#00d4aa' }}>
                    {backtestResults.months_to_1m >= 24
                      ? `${(backtestResults.months_to_1m / 12).toFixed(1)} yrs`
                      : `${backtestResults.months_to_1m} mo`}
                  </span>
                </div>
              )}
              {backtestResults.total_return > 0 && backtestResults.max_drawdown > 0 && (
                <div className="indicator-row">
                  <span className="indicator-name">Risk-Adj Score</span>
                  <span className="indicator-value" style={{ color: '#4a9eff' }}>
                    {(backtestResults.win_rate * backtestResults.total_return / Math.max(backtestResults.max_drawdown, 1)).toFixed(1)}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Growth Projection — only when profitable */}
      {backtestResults && backtestResults.total_return > 0 && (
        <GrowthProjection
          periodDays={backtestResults.period_days}
          totalReturn={backtestResults.total_return}
          monthlyRoi={backtestResults.monthly_roi}
        />
      )}

      {/* Results by Coin */}
      {backtestResults?.coin_results && (
        <div className="card">
          <h2 className="card-title">Results by Coin</h2>
          {backtestResults.coin_results.map((coinResult, idx) => (
            <div key={idx} style={{
              padding: '12px',
              marginBottom: 8,
              background: '#0f1318',
              borderRadius: 10,
              border: expandedCoin === coinResult.coin ? '1px solid #e94560' : '1px solid rgba(255,255,255,0.07)',
              cursor: 'pointer'
            }} onClick={() => setExpandedCoin(expandedCoin === coinResult.coin ? null : coinResult.coin)}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 700, color: '#e94560', fontSize: 15 }}>{coinResult.coin}</span>
                <span className={coinResult.total_pnl >= 0 ? 'positive' : 'negative'} style={{ fontWeight: 700 }}>
                  {coinResult.total_pnl >= 0 ? '+' : ''}${coinResult.total_pnl?.toFixed(2)}
                </span>
              </div>
              {coinResult.error ? (
                <div style={{ color: '#f5a623', fontSize: 12, marginTop: 4 }}>{coinResult.error}</div>
              ) : (
                <div style={{ display: 'flex', gap: 12, fontSize: 12, color: '#8b95a5', marginTop: 6 }}>
                  <span>Return: <strong className={coinResult.total_return >= 0 ? 'positive' : 'negative'}>{coinResult.total_return >= 0 ? '+' : ''}{coinResult.total_return?.toFixed(1)}%</strong></span>
                  <span>Trades: <strong style={{ color: '#e8eaf0' }}>{coinResult.total_trades}</strong></span>
                  <span>Win: <strong style={{ color: coinResult.win_rate >= 50 ? '#00d4aa' : '#ff4444' }}>{coinResult.win_rate?.toFixed(0)}%</strong></span>
                </div>
              )}

              {expandedCoin === coinResult.coin && coinResult.trades?.length > 0 && (
                <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid rgba(255,255,255,0.07)' }}>
                  <div style={{ fontSize: 12, color: '#4a5060', marginBottom: 8 }}>Trade log ({coinResult.trades.length})</div>
                  {coinResult.trades.map((trade, tIdx) => (
                    <div key={tIdx} style={{
                      display: 'flex', justifyContent: 'space-between',
                      padding: '6px 8px', background: 'rgba(0,0,0,0.2)',
                      borderRadius: 4, marginBottom: 4, fontSize: 11
                    }}>
                      <span style={{ color: trade.side === 'long' ? '#00d4aa' : '#ff4444', textTransform: 'uppercase', fontWeight: 700 }}>{trade.side}</span>
                      <span style={{ color: '#4a5060' }}>Entry: ${trade.entry}</span>
                      <span style={{ color: '#4a5060' }}>Exit: ${trade.exit}</span>
                      <span style={{ fontSize: 10, color: '#4a5060' }}>{trade.reason}</span>
                      <span className={trade.pnl >= 0 ? 'positive' : 'negative'} style={{ fontWeight: 700 }}>
                        {trade.pnl >= 0 ? '+' : ''}${trade.pnl}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* All Trades Log */}
      {backtestResults?.all_trades?.length > 0 && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <h2 className="card-title" style={{ margin: 0 }}>All Trades</h2>
            <button onClick={() => setShowAllTrades(!showAllTrades)} style={{
              padding: '6px 12px', background: 'rgba(233,69,96,0.12)', border: '1px solid rgba(233,69,96,0.3)',
              borderRadius: 8, color: '#e94560', cursor: 'pointer', fontSize: 12, fontWeight: 600
            }}>
              {showAllTrades ? 'Hide' : `Show ${backtestResults.all_trades.length} trades`}
            </button>
          </div>
          {showAllTrades && (
            <div style={{ maxHeight: 300, overflowY: 'auto' }}>
              {backtestResults.all_trades.map((trade, idx) => (
                <div key={idx} style={{
                  display: 'flex', justifyContent: 'space-between',
                  padding: '8px', background: idx % 2 === 0 ? 'rgba(0,0,0,0.15)' : 'transparent',
                  borderRadius: 4, fontSize: 12, alignItems: 'center'
                }}>
                  <span style={{ color: '#e94560', fontWeight: 700, minWidth: 40 }}>{trade.coin}</span>
                  <span style={{ color: trade.side === 'long' ? '#00d4aa' : '#ff4444', textTransform: 'uppercase', minWidth: 50, fontWeight: 600 }}>{trade.side}</span>
                  <span style={{ color: '#4a5060' }}>${trade.entry} → ${trade.exit}</span>
                  <span className={trade.pnl >= 0 ? 'positive' : 'negative'} style={{ fontWeight: 700, minWidth: 70, textAlign: 'right' }}>
                    {trade.pnl >= 0 ? '+' : ''}${trade.pnl} ({trade.pnl_pct}%)
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Strategy Overview */}
      <div className="card">
        <h2 style={{ fontSize: 18, marginBottom: 6, fontWeight: 700 }}>{strategies.name}</h2>
        <p style={{ color: '#8b95a5', fontSize: 14, lineHeight: 1.6 }}>{strategies.description}</p>
      </div>

      {strategies.components?.map((component, i) => (
        <div key={i} className="card strategy-section">
          <div className="strategy-title">
            {component.name}
            <span className="strategy-weight">{component.weight}</span>
          </div>
          <p className="strategy-desc">{component.description}</p>
          {component.details && (
            <ul className="strategy-list">{component.details.map((d, j) => <li key={j}>{d}</li>)}</ul>
          )}
          {component.indicators && (
            <ul className="strategy-list">{component.indicators.map((ind, j) => <li key={j}><strong>{ind.name}:</strong> {ind.desc}</li>)}</ul>
          )}
        </div>
      ))}
    </>
  )
}

/* ─────────────────────────── Optimize ─────────────────────────── */

function OptimizePage({ api, setError, setSuccess }) {
  const [optimizeResults, setOptimizeResults] = useState(null)
  const [optimizeLoading, setOptimizeLoading] = useState(false)
  const [optimizeStatus, setOptimizeStatus] = useState('')
  const [optimizeError, setOptimizeError] = useState('')
  const [selectedConfig, setSelectedConfig] = useState(null)
  const [applyLoading, setApplyLoading] = useState(false)
  const [applyingRunId, setApplyingRunId] = useState(null)
  const [history, setHistory] = useState([])
  const [loadedRunId, setLoadedRunId] = useState(null)
  const pollIntervalRef = useRef(null)

  useEffect(() => {
    api.get('/optimize/history').then(r => setHistory(r.data.runs || [])).catch(() => {})
    return () => { if (pollIntervalRef.current) clearInterval(pollIntervalRef.current) }
  }, [])

  const pollForResults = async () => {
    try {
      const res = await api.get('/optimize/status')
      if (res.data.status === 'completed') {
        setOptimizeResults(res.data.result)
        setLoadedRunId(null)
        setOptimizeLoading(false)
        setOptimizeStatus('')
        clearInterval(pollIntervalRef.current)
        pollIntervalRef.current = null
        api.get('/optimize/history').then(r => setHistory(r.data.runs || [])).catch(() => {})
        return true
      } else if (res.data.status === 'failed') {
        setOptimizeError(res.data.error || 'Optimization failed')
        setOptimizeLoading(false)
        setOptimizeStatus('')
        clearInterval(pollIntervalRef.current)
        pollIntervalRef.current = null
        return true
      } else {
        setOptimizeStatus(res.data.message || 'Running...')
        return false
      }
    } catch (e) { return false }
  }

  const loadHistoricalRun = async (runId) => {
    try {
      const r = await api.get('/optimize/run/' + runId)
      setOptimizeResults(r.data.result)
      setLoadedRunId(runId)
      setSelectedConfig(null)
    } catch (e) {
      setError('Failed to load historical run')
    }
  }

  const applyRunToBot = async (runId) => {
    setApplyingRunId(runId)
    try {
      const r = await api.post('/bot/apply-optimizer/' + runId)
      const cfg = r.data.config
      const liveMsg = r.data.applied_to_live_bot ? ' (applied to live bot)' : ' (saved to settings)'
      setSuccess(`Optimizer config applied${liveMsg}: ${cfg.timeframe} · ${cfg.leverage}x lev · ${(cfg.risk_per_trade * 100).toFixed(1)}% risk`)
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to apply optimizer run')
    }
    setApplyingRunId(null)
  }

  const runOptimization = async () => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
    setOptimizeLoading(true)
    setOptimizeError('')
    setOptimizeResults(null)
    setSelectedConfig(null)
    setOptimizeStatus('Starting optimization...')
    try {
      await api.post('/optimize/start')
      pollIntervalRef.current = setInterval(async () => { await pollForResults() }, 5000)
    } catch (e) {
      setOptimizeError(e.response?.data?.detail || 'Failed to start optimization')
      setOptimizeLoading(false)
    }
  }

  const applyConfig = async (config) => {
    setApplyLoading(true)
    try {
      await api.put('/settings', {
        leverage: config.leverage,
        risk_per_trade: config.risk_per_trade,
        stop_loss_pct: config.stop_loss_pct,
        take_profit_pct: config.take_profit_pct,
        trade_cooldown: config.trade_cooldown,
        min_confidence: config.min_confidence,
        timeframe: config.timeframe,
        ...(config.trailing_stop_pct != null && { trailing_stop_pct: config.trailing_stop_pct }),
      })
      setSuccess(`Applied: ${config.timeframe} · ${config.leverage}x · ${(config.min_confidence * 100).toFixed(0)}% conf`)
      setSelectedConfig(config)
    } catch (e) {
      setError('Failed to apply settings — stop the bot first')
    }
    setApplyLoading(false)
  }

  const resetOptimization = async () => {
    try {
      await api.post('/optimize/reset')
      setOptimizeLoading(false)
      setOptimizeStatus('')
      setOptimizeError('')
      if (pollIntervalRef.current) { clearInterval(pollIntervalRef.current); pollIntervalRef.current = null }
      setSuccess('Reset — you can start a new optimization')
    } catch (e) { setError('Failed to reset') }
  }

  const formatCooldown = (s) => s < 60 ? `${s}s` : `${Math.round(s / 60)}m`

  return (
    <>
      <div className="card">
        <h2 className="card-title">Parameter Optimizer</h2>
        <p style={{ color: '#8b95a5', fontSize: 14, marginBottom: 12, lineHeight: 1.5 }}>
          Automatically discovers the best trading parameters for your coins by testing ~1,350 configurations across all timeframes. Returns top performers ranked by ROI, win rate, drawdown, and trade count.
        </p>
        <p style={{ color: '#e94560', fontSize: 12, marginBottom: 16 }}>
          Warning: This takes 5–15 minutes to complete.
        </p>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-primary" onClick={runOptimization} disabled={optimizeLoading}>
            {optimizeLoading ? (optimizeStatus || 'Optimizing...') : 'Find Best Parameters'}
          </button>
          {optimizeLoading && (
            <button className="btn btn-danger" onClick={resetOptimization} style={{ maxWidth: 90, padding: '14px 12px', fontSize: 13 }}>
              Reset
            </button>
          )}
        </div>
        {optimizeError && <div className="error-message" style={{ marginTop: 16 }}>{optimizeError}</div>}
      </div>

      {optimizeResults && (
        <div className="card">
          <h2 className="card-title">Top Configurations</h2>
          <div style={{ fontSize: 12, color: '#8b95a5', marginBottom: 4 }}>
            Tested {optimizeResults.total_tested} configs · {optimizeResults.valid_configs} profitable · {optimizeResults.days_tested} days
          </div>
          <div style={{ fontSize: 12, color: '#8b95a5', marginBottom: 16 }}>
            Coins: {optimizeResults.selected_coins?.map(c => c.split('/')[0]).join(', ')}
          </div>

          {optimizeResults.top_configs?.length === 0 ? (
            <p style={{ color: '#e94560' }}>No profitable configurations found. Try different coins or timeframes.</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {optimizeResults.top_configs?.map((config, i) => {
                const isSelected = selectedConfig === config
                const monthlyRet = (config.total_return / Math.max(optimizeResults.days_tested, 1)) * 30
                const yearValue  = 1000 * Math.pow(1 + monthlyRet / 100, 12)
                const annualPct  = (yearValue / 1000 - 1) * 100

                return (
                  <div key={i} style={{
                    padding: 14,
                    background: isSelected ? 'rgba(0,212,170,0.08)' : '#0f1318',
                    borderRadius: 12,
                    border: isSelected ? '1px solid rgba(0,212,170,0.4)' : '1px solid rgba(255,255,255,0.07)'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                      <span style={{ fontSize: 14, fontWeight: 700, color: '#e94560' }}>
                        #{i + 1} — {config.timeframe}
                      </span>
                      <span style={{ fontSize: 15, fontWeight: 700, color: config.total_return >= 0 ? '#00d4aa' : '#ff4444' }}>
                        {config.total_return >= 0 ? '+' : ''}{config.total_return.toFixed(1)}% ROI
                      </span>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 12, color: '#8b95a5', marginBottom: 10 }}>
                      <span>Leverage: <strong style={{ color: '#e8eaf0' }}>{config.leverage}x</strong></span>
                      <span>Risk: <strong style={{ color: '#e8eaf0' }}>{(config.risk_per_trade * 100).toFixed(0)}%</strong></span>
                      <span>SL: <strong style={{ color: '#ff4444' }}>{(config.stop_loss_pct * 100).toFixed(2)}%</strong></span>
                      <span>TP: <strong style={{ color: '#00d4aa' }}>{(config.take_profit_pct * 100).toFixed(2)}%</strong></span>
                      <span>Trail: <strong style={{ color: '#e8eaf0' }}>{config.trailing_stop_pct != null ? `${(config.trailing_stop_pct * 100).toFixed(1)}%` : '—'}</strong></span>
                      <span>Conf: <strong style={{ color: '#e8eaf0' }}>{(config.min_confidence * 100).toFixed(0)}%</strong></span>
                      <span>ADX: <strong style={{ color: '#e8eaf0' }}>{config.adx_threshold ?? '—'}</strong></span>
                      <span>Cooldown: <strong style={{ color: '#e8eaf0' }}>{formatCooldown(config.trade_cooldown)}</strong></span>
                      <span>Score: <strong style={{ color: '#4a9eff' }}>{config.score?.toFixed(3)}</strong></span>
                    </div>

                    {/* Compound projection per config */}
                    {monthlyRet > 0 && (
                      <div style={{ padding: '8px 10px', background: 'rgba(245,166,35,0.07)', borderRadius: 8, border: '1px solid rgba(245,166,35,0.18)', marginBottom: 10 }}>
                        <span style={{ fontSize: 11, color: '#8b95a5' }}>12-mo compound: </span>
                        <span style={{ fontSize: 13, fontWeight: 700, color: '#f5a623' }}>
                          ${yearValue.toFixed(0)}&nbsp;
                        </span>
                        <span style={{ fontSize: 11, color: '#00d4aa' }}>
                          (+{annualPct.toFixed(0)}% on $1k)
                        </span>
                      </div>
                    )}

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 12 }}>
                      <span style={{ color: '#00d4aa' }}>
                        {config.win_rate.toFixed(1)}% win · {config.total_trades} trades · ${config.total_pnl.toFixed(0)} PnL
                      </span>
                      <button
                        className="btn btn-secondary"
                        style={{ padding: '6px 14px', fontSize: 12, width: 'auto' }}
                        onClick={() => applyConfig(config)}
                        disabled={applyLoading}
                      >
                        {isSelected ? 'Applied' : 'Apply'}
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      <div className="card">
        <h2 className="card-title">Previous Runs</h2>
        {history.length === 0 ? (
          <p style={{ color: '#8b95a5', fontSize: 14 }}>No previous runs yet.</p>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: '#8b95a5', fontWeight: 600 }}>Date</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: '#8b95a5', fontWeight: 600 }}>Coins</th>
                  <th style={{ textAlign: 'right', padding: '6px 8px', color: '#8b95a5', fontWeight: 600 }}>Best ROI</th>
                  <th style={{ textAlign: 'right', padding: '6px 8px', color: '#8b95a5', fontWeight: 600 }}>Win Rate</th>
                  <th style={{ textAlign: 'right', padding: '6px 8px', color: '#8b95a5', fontWeight: 600 }}>Configs</th>
                  <th style={{ padding: '6px 8px' }}></th>
                </tr>
              </thead>
              <tbody>
                {history.map(run => {
                  const isLoaded = loadedRunId === run.id
                  const isApplying = applyingRunId === run.id
                  return (
                    <tr key={run.id} style={{
                      borderBottom: '1px solid rgba(255,255,255,0.05)',
                      background: isLoaded ? 'rgba(0,212,170,0.06)' : 'transparent',
                    }}>
                      <td style={{ padding: '8px 8px', color: '#e8eaf0' }}>
                        {run.completed_at ? new Date(run.completed_at).toLocaleDateString() : '—'}
                      </td>
                      <td style={{ padding: '8px 8px', color: '#e8eaf0' }}>
                        {(run.coins || []).map(c => c.split('/')[0]).join(', ') || '—'}
                      </td>
                      <td style={{ padding: '8px 8px', textAlign: 'right', color: run.best_roi > 0 ? '#00d4aa' : '#ff4444', fontWeight: 600 }}>
                        {run.best_roi != null ? `${run.best_roi >= 0 ? '+' : ''}${Number(run.best_roi).toFixed(1)}%` : '—'}
                      </td>
                      <td style={{ padding: '8px 8px', textAlign: 'right', color: '#e8eaf0' }}>
                        {run.best_win_rate != null ? `${Number(run.best_win_rate).toFixed(1)}%` : '—'}
                      </td>
                      <td style={{ padding: '8px 8px', textAlign: 'right', color: '#8b95a5' }}>
                        {run.valid_configs}/{run.total_tested}
                      </td>
                      <td style={{ padding: '6px 4px', textAlign: 'right' }}>
                        <div style={{ display: 'flex', gap: 4, justifyContent: 'flex-end' }}>
                          <button
                            className="btn btn-secondary"
                            style={{ padding: '4px 10px', fontSize: 11, width: 'auto', opacity: isLoaded ? 0.5 : 1 }}
                            onClick={() => loadHistoricalRun(run.id)}
                            disabled={isLoaded}
                          >
                            {isLoaded ? '✓' : 'Load'}
                          </button>
                          <button
                            className="btn btn-success"
                            style={{ padding: '4px 10px', fontSize: 11, width: 'auto', background: '#00d4aa', color: '#0a0d12', fontWeight: 700, opacity: isApplying ? 0.5 : 1 }}
                            onClick={() => applyRunToBot(run.id)}
                            disabled={isApplying}
                            title="Hot-apply this run's best config to the running bot"
                          >
                            {isApplying ? '...' : '⚡ Apply'}
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card">
        <h2 className="card-title">How It Works</h2>
        <ul className="strategy-list">
          <li><strong>Smart Search:</strong> Tests 100 random parameter combos per timeframe across all major timeframes</li>
          <li><strong>Trailing Stop:</strong> Also optimises trailing stop distance (0.5%–2%) for locked profits</li>
          <li><strong>Composite Score:</strong> 50% ROI + 30% Win Rate + 20% Trade Count, with drawdown penalty</li>
          <li><strong>Drawdown Guard:</strong> Penalises configs where max drawdown exceeds 15%</li>
          <li><strong>Min Trades:</strong> Rejects configs with fewer than 15 trades (avoids overfitting)</li>
          <li><strong>Compound Projection:</strong> Shows annualised compounding at each config's monthly return</li>
        </ul>
      </div>
    </>
  )
}

/* ─────────────────────────── Settings ─────────────────────────── */

const AVAILABLE_COINS = [
  'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
  'DOGE/USDT:USDT', 'BNB/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT',
  'LINK/USDT:USDT', 'MATIC/USDT:USDT', 'DOT/USDT:USDT', 'UNI/USDT:USDT',
  'SHIB/USDT:USDT', 'LTC/USDT:USDT', 'ATOM/USDT:USDT', 'XLM/USDT:USDT'
]

function SettingsPage({ api, logout, setError, setSuccess, botStatus }) {
  const [apiKey, setApiKey] = useState('')
  const [apiSecret, setApiSecret] = useState('')
  const [apiPassword, setApiPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [hasCredentials, setHasCredentials] = useState(false)

  const [startingBalance, setStartingBalance] = useState(10000)
  const [leverage, setLeverage] = useState(10)
  const [selectedCoins, setSelectedCoins] = useState(['BTC/USDT:USDT'])
  const [riskPerTrade, setRiskPerTrade] = useState(2)
  const [stopLossPct, setStopLossPct] = useState(15)
  const [takeProfitPct, setTakeProfitPct] = useState(30)
  const [tradeCooldown, setTradeCooldown] = useState(5)
  const [minConfidence, setMinConfidence] = useState(65)
  const [timeframe, setTimeframe] = useState('5m')
  const [simulationMode, setSimulationMode] = useState(true)

  const [trailingStopPct, setTrailingStopPct] = useState(10)
  const [maxDrawdownPct, setMaxDrawdownPct] = useState(20)
  const [retrainEvery, setRetrainEvery] = useState(50)
  const [profitRiskMultiplier, setProfitRiskMultiplier] = useState(1.5)
  const [adxThreshold, setAdxThreshold] = useState(18)

  const [settingsLoading, setSettingsLoading] = useState(false)

  const TIMEFRAME_OPTIONS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']
  const isBotRunning = botStatus?.running

  useEffect(() => {
    checkCredentials()
    loadSettings()
  }, [])

  const checkCredentials = async () => {
    try {
      const res = await api.get('/credentials/status')
      setHasCredentials(res.data.has_credentials)
    } catch (e) {}
  }

  const loadSettings = async () => {
    try {
      const res = await api.get('/settings')
      const d = res.data
      setStartingBalance(d.starting_balance || 10000)
      setLeverage(d.leverage || 10)
      setSelectedCoins(d.selected_coins || ['BTC/USDT:USDT'])
      setRiskPerTrade((d.risk_per_trade || 0.02) * 100)
      setStopLossPct((d.stop_loss_pct || 0.15) * 100)
      setTakeProfitPct((d.take_profit_pct || 0.30) * 100)
      setTradeCooldown((d.trade_cooldown || 300) / 60)
      setMinConfidence((d.min_confidence || 0.65) * 100)
      setTimeframe(d.timeframe || '5m')
      setSimulationMode(d.simulation_mode !== false)
      setTrailingStopPct((d.trailing_stop_pct || 0.10) * 100)
      setMaxDrawdownPct((d.max_drawdown_pct || 0.20) * 100)
      setRetrainEvery(d.retrain_every || 50)
      setProfitRiskMultiplier(d.profit_risk_multiplier || 1.5)
      setAdxThreshold(d.adx_threshold || 18)
    } catch (e) {}
  }

  const saveCredentials = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      await api.post('/credentials', { api_key: apiKey, api_secret: apiSecret, api_password: apiPassword })
      setSuccess('API credentials saved')
      setApiKey(''); setApiSecret(''); setApiPassword('')
      checkCredentials()
    } catch (e) {
      setError('Failed to save credentials')
    }
    setLoading(false)
  }

  const saveSettings = async () => {
    if (isBotRunning) { setError('Stop the bot before changing settings'); return }

    const v = {
      risk:       Math.max(0.1, Math.min(100, riskPerTrade)),
      sl:         Math.max(1, Math.min(75, stopLossPct)),
      tp:         Math.max(1, Math.min(200, takeProfitPct)),
      cooldown:   Math.max(1, Math.min(60, Math.round(tradeCooldown))),
      conf:       Math.max(50, Math.min(95, Math.round(minConfidence))),
      trail:      Math.max(1, Math.min(50, trailingStopPct)),
      drawdown:   Math.max(5, Math.min(50, maxDrawdownPct)),
      retrain:    Math.max(10, Math.min(500, Math.round(retrainEvery))),
      multiplier: Math.max(1.0, Math.min(3.0, profitRiskMultiplier)),
      adx:        Math.max(5, Math.min(30, Math.round(adxThreshold))),
    }

    setRiskPerTrade(v.risk); setStopLossPct(v.sl); setTakeProfitPct(v.tp)
    setTradeCooldown(v.cooldown); setMinConfidence(v.conf)
    setTrailingStopPct(v.trail); setMaxDrawdownPct(v.drawdown)
    setRetrainEvery(v.retrain); setProfitRiskMultiplier(v.multiplier)
    setAdxThreshold(v.adx)

    setSettingsLoading(true)
    try {
      await api.put('/settings', {
        starting_balance: startingBalance,
        leverage,
        selected_coins: selectedCoins,
        risk_per_trade: v.risk / 100,
        stop_loss_pct: v.sl / 100,
        take_profit_pct: v.tp / 100,
        trade_cooldown: v.cooldown * 60,
        min_confidence: v.conf / 100,
        timeframe,
        simulation_mode: simulationMode,
        trailing_stop_pct: v.trail / 100,
        max_drawdown_pct: v.drawdown / 100,
        retrain_every: v.retrain,
        profit_risk_multiplier: v.multiplier,
        adx_threshold: v.adx,
      })
      setSuccess('Settings saved')
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to save settings')
    }
    setSettingsLoading(false)
  }

  const toggleCoin = (coin) => {
    if (isBotRunning) return
    if (selectedCoins.includes(coin)) {
      if (selectedCoins.length > 1) setSelectedCoins(selectedCoins.filter(c => c !== coin))
    } else if (selectedCoins.length < 5) {
      setSelectedCoins([...selectedCoins, coin])
    } else {
      setError('Maximum 5 coins allowed')
    }
  }

  const dis = (disabled) => ({ opacity: disabled ? 0.45 : 1 })
  const sel = (disabled) => ({
    opacity: disabled ? 0.45 : 1, width: '100%', padding: '13px 16px',
    borderRadius: 12, border: '1px solid rgba(255,255,255,0.07)',
    background: '#0f1318', color: '#e8eaf0', fontSize: 15, outline: 'none'
  })

  const BotRunningWarning = () => isBotRunning ? (
    <div className="error-message" style={{ marginBottom: 16 }}>Stop the bot to change settings</div>
  ) : null

  return (
    <>
      {/* ── Trading Mode ── */}
      <div className="card">
        <h2 className="card-title">Trading Mode</h2>
        <BotRunningWarning />
        <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
          <button onClick={() => !isBotRunning && setSimulationMode(true)} style={{
            flex: 1, padding: '12px', borderRadius: 10, border: simulationMode ? '2px solid #f5a623' : '1px solid rgba(255,255,255,0.07)',
            background: simulationMode ? 'rgba(245,166,35,0.12)' : 'transparent',
            color: simulationMode ? '#f5a623' : '#8b95a5', cursor: isBotRunning ? 'not-allowed' : 'pointer',
            fontWeight: 600, fontSize: 14, transition: 'all 0.2s'
          }}>
            Simulation
          </button>
          <button onClick={() => !isBotRunning && setSimulationMode(false)} style={{
            flex: 1, padding: '12px', borderRadius: 10, border: !simulationMode ? '2px solid #ff4444' : '1px solid rgba(255,255,255,0.07)',
            background: !simulationMode ? 'rgba(255,68,68,0.12)' : 'transparent',
            color: !simulationMode ? '#ff4444' : '#8b95a5', cursor: isBotRunning ? 'not-allowed' : 'pointer',
            fontWeight: 600, fontSize: 14, transition: 'all 0.2s'
          }}>
            Live Trading
          </button>
        </div>
        <p style={{ fontSize: 12, color: '#4a5060', textAlign: 'center' }}>
          {simulationMode ? 'Paper trading — no real funds at risk' : 'LIVE — real funds will be used. Trade with caution.'}
        </p>
        <button className="btn btn-primary" onClick={saveSettings} disabled={settingsLoading || isBotRunning} style={{ marginTop: 14 }}>
          {settingsLoading ? 'Saving...' : 'Save Mode'}
        </button>
      </div>

      {/* ── Trading Settings ── */}
      <div className="card">
        <h2 className="card-title">Trading Settings</h2>
        <BotRunningWarning />

        <div className="input-group">
          <label>Starting Balance (USDT)</label>
          <input type="number" value={startingBalance} onChange={e => setStartingBalance(Number(e.target.value))} min="100" disabled={isBotRunning} style={dis(isBotRunning)} />
        </div>

        <div className="input-group">
          <label>Leverage</label>
          <select value={leverage} onChange={e => setLeverage(Number(e.target.value))} disabled={isBotRunning} style={sel(isBotRunning)}>
            {[1, 2, 3, 5, 10, 15, 20, 25, 50, 75, 100].map(lev => (
              <option key={lev} value={lev}>{lev}x</option>
            ))}
          </select>
        </div>

        <div className="input-group">
          <label>Timeframe</label>
          <select value={timeframe} onChange={e => setTimeframe(e.target.value)} disabled={isBotRunning} style={sel(isBotRunning)}>
            {TIMEFRAME_OPTIONS.map(tf => <option key={tf} value={tf}>{tf}</option>)}
          </select>
          <p style={{ fontSize: 12, color: '#4a5060', marginTop: 4 }}>Candle timeframe for analysis (uses 1000 candles)</p>
        </div>

        <div className="input-group">
          <label>Trading Coins (max 5)</label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
            {AVAILABLE_COINS.map(coin => {
              const active = selectedCoins.includes(coin)
              return (
                <div key={coin} onClick={() => toggleCoin(coin)} style={{
                  padding: '7px 12px', borderRadius: 8,
                  border: active ? '2px solid #e94560' : '1px solid rgba(255,255,255,0.07)',
                  background: active ? 'rgba(233,69,96,0.15)' : '#0f1318',
                  color: active ? '#e94560' : '#8b95a5',
                  cursor: isBotRunning ? 'not-allowed' : 'pointer',
                  opacity: isBotRunning ? 0.45 : 1,
                  fontSize: 13, fontWeight: active ? 700 : 400, transition: 'all 0.15s'
                }}>
                  {coin.split('/')[0]}
                </div>
              )
            })}
          </div>
          <p style={{ fontSize: 12, color: '#4a5060', marginTop: 8 }}>
            Selected: {selectedCoins.map(c => c.split('/')[0]).join(', ')}
          </p>
        </div>

        <button className="btn btn-primary" onClick={saveSettings} disabled={settingsLoading || isBotRunning}>
          {settingsLoading ? 'Saving...' : 'Save Trading Settings'}
        </button>
      </div>

      {/* ── Risk Management ── */}
      <div className="card">
        <h2 className="card-title">Risk Management</h2>
        <BotRunningWarning />

        <div className="input-group">
          <label>Risk Per Trade (%)</label>
          <input type="number" value={riskPerTrade} onChange={e => setRiskPerTrade(Number(e.target.value))} min="0.1" max="100" step="0.1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: '#4a5060', marginTop: 4 }}>Percentage of balance per trade (0.1%–100%)</p>
        </div>

        <div className="input-group">
          <label>Stop Loss (% of margin)</label>
          <input type="number" value={stopLossPct} onChange={e => setStopLossPct(Number(e.target.value))} min="1" max="75" step="1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: '#4a5060', marginTop: 4 }}>
            Exit when margin lost ≥ this %. At {leverage}x leverage → {(stopLossPct / leverage).toFixed(2)}% price move.
          </p>
        </div>

        <div className="input-group">
          <label>Take Profit (% of margin)</label>
          <input type="number" value={takeProfitPct} onChange={e => setTakeProfitPct(Number(e.target.value))} min="1" max="200" step="1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: '#4a5060', marginTop: 4 }}>
            Exit when margin gained ≥ this %. At {leverage}x leverage → {(takeProfitPct / leverage).toFixed(2)}% price move.
          </p>
        </div>

        <div className="input-group">
          <label>Trade Cooldown (minutes)</label>
          <select value={tradeCooldown} onChange={e => setTradeCooldown(Number(e.target.value))} disabled={isBotRunning} style={sel(isBotRunning)}>
            {[1, 2, 3, 5, 10, 15, 20, 30, 60].map(min => <option key={min} value={min}>{min} min</option>)}
          </select>
          <p style={{ fontSize: 12, color: '#4a5060', marginTop: 4 }}>Wait time between trades per coin</p>
        </div>

        <div className="input-group">
          <label>Minimum Confidence (%)</label>
          <input type="number" value={minConfidence} onChange={e => setMinConfidence(Number(e.target.value))} min="50" max="95" step="1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: '#4a5060', marginTop: 4 }}>Only trade when ML signal confidence exceeds this (50%–95%)</p>
        </div>

        <div className="input-group">
          <label>ADX Entry Threshold</label>
          <input type="number" value={adxThreshold} onChange={e => setAdxThreshold(Number(e.target.value))} min="5" max="30" step="1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: '#4a5060', marginTop: 4 }}>
            Minimum ADX trend strength required to open a trade (5–30). Lower = more trades, higher = stronger trends only. Run Optimizer to find the best value per token.
          </p>
        </div>

        <button className="btn btn-primary" onClick={saveSettings} disabled={settingsLoading || isBotRunning}>
          {settingsLoading ? 'Saving...' : 'Save Risk Settings'}
        </button>
      </div>

      {/* ── Compounding & Protection ── */}
      <div className="card">
        <h2 className="card-title">Compounding & Protection</h2>
        <p style={{ fontSize: 13, color: '#8b95a5', marginBottom: 16, lineHeight: 1.5 }}>
          Controls how profits compound and how the bot protects capital on the path to growth.
        </p>
        <BotRunningWarning />

        <div className="input-group">
          <label>Trailing Stop (% of margin)</label>
          <input type="number" value={trailingStopPct} onChange={e => setTrailingStopPct(Number(e.target.value))} min="1" max="50" step="1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: '#4a5060', marginTop: 4 }}>
            Locks in gains — exits if price retraces this % of margin from its peak while in profit. At {leverage}x leverage → {(trailingStopPct / leverage).toFixed(2)}% price move. (1%–50%)
          </p>
        </div>

        <div className="input-group">
          <label>Max Drawdown Circuit Breaker (%)</label>
          <input type="number" value={maxDrawdownPct} onChange={e => setMaxDrawdownPct(Number(e.target.value))} min="5" max="50" step="1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: '#4a5060', marginTop: 4 }}>
            Stops opening new positions if account drawdown exceeds this. Prevents runaway losses. (5%–50%)
          </p>
        </div>

        <div className="input-group">
          <label>Profit Risk Multiplier</label>
          <input type="number" value={profitRiskMultiplier} onChange={e => setProfitRiskMultiplier(Number(e.target.value))} min="1.0" max="3.0" step="0.1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: '#4a5060', marginTop: 4 }}>
            Applies extra risk to profits above starting balance — "house money" mode. 1.5 = 50% more aggressive on profits. (1.0–3.0)
          </p>
        </div>

        <div className="input-group">
          <label>Model Retrain Interval (cycles)</label>
          <input type="number" value={retrainEvery} onChange={e => setRetrainEvery(Number(e.target.value))} min="10" max="500" step="10" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: '#4a5060', marginTop: 4 }}>
            Retrain the ML model every N cycles to adapt to current market conditions. (10–500)
          </p>
        </div>

        <button className="btn btn-primary" onClick={saveSettings} disabled={settingsLoading || isBotRunning}>
          {settingsLoading ? 'Saving...' : 'Save Compounding Settings'}
        </button>
      </div>

      {/* ── API Credentials ── */}
      <div className="card">
        <h2 className="card-title">Blofin API Credentials</h2>
        {hasCredentials && (
          <div className="success-message" style={{ marginBottom: 16 }}>API credentials configured</div>
        )}
        <form onSubmit={saveCredentials}>
          <div className="input-group">
            <label>API Key</label>
            <input type="text" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="Enter your Blofin API key" />
          </div>
          <div className="input-group">
            <label>API Secret</label>
            <input type="password" value={apiSecret} onChange={e => setApiSecret(e.target.value)} placeholder="Enter your API secret" />
          </div>
          <div className="input-group">
            <label>API Passphrase</label>
            <input type="password" value={apiPassword} onChange={e => setApiPassword(e.target.value)} placeholder="Enter your API passphrase" />
          </div>
          <button type="submit" className="btn btn-primary" disabled={loading || (!apiKey && !apiSecret)}>
            {loading ? 'Saving...' : 'Save Credentials'}
          </button>
        </form>
      </div>

      {/* ── Account ── */}
      <div className="card">
        <h2 className="card-title">Account</h2>
        <button className="btn btn-secondary" onClick={logout}>Logout</button>
      </div>

      <div className="card">
        <h2 className="card-title">Install App</h2>
        <p style={{ color: '#8b95a5', fontSize: 14 }}>
          On Android: open browser menu and tap "Add to Home Screen" to install as an app.
        </p>
      </div>
    </>
  )
}

/* ─────────────────────────── Admin ─────────────────────────── */

function AdminPage({ api, setError, setSuccess }) {
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)

  const loadUsers = async () => {
    setLoading(true)
    try {
      const res = await api.get('/admin/users')
      setUsers(res.data.users || [])
    } catch (e) {
      setError('Failed to load users')
    }
    setLoading(false)
  }

  useEffect(() => { loadUsers() }, [])

  const approveUser = async (userId, makeApproved) => {
    try {
      await api.post(`/admin/users/${userId}/approve`, { action: makeApproved ? 'approve' : 'reject' })
      setSuccess(makeApproved ? 'User approved' : 'User access revoked')
      loadUsers()
    } catch (e) {
      setError('Failed to update user')
    }
  }

  const toggleAdmin = async (userId, makeAdmin) => {
    try {
      await api.post(`/admin/users/${userId}/permissions`, { is_admin: makeAdmin })
      setSuccess(makeAdmin ? 'Admin privileges granted' : 'Admin privileges removed')
      loadUsers()
    } catch (e) {
      setError('Failed to update permissions')
    }
  }

  if (loading) return <div className="loading"><div className="spinner"></div></div>

  const pending  = users.filter(u => u.account_status !== 'approved')
  const approved = users.filter(u => u.account_status === 'approved')

  return (
    <>
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h2 className="card-title" style={{ margin: 0 }}>User Management</h2>
          <button onClick={loadUsers} style={{
            padding: '6px 12px', background: 'rgba(233,69,96,0.12)',
            border: '1px solid rgba(233,69,96,0.3)', borderRadius: 8,
            color: '#e94560', cursor: 'pointer', fontSize: 12, fontWeight: 600
          }}>
            Refresh
          </button>
        </div>

        {users.length === 0 && (
          <p style={{ color: '#8b95a5', textAlign: 'center', padding: 20 }}>No users found</p>
        )}

        {pending.length > 0 && (
          <>
            <div style={{ fontSize: 12, color: '#f5a623', fontWeight: 600, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Pending Approval ({pending.length})
            </div>
            {pending.map(user => <UserItem key={user.id} user={user} onApprove={approveUser} onToggleAdmin={toggleAdmin} />)}
            <div style={{ height: 12 }} />
          </>
        )}

        {approved.length > 0 && (
          <>
            <div style={{ fontSize: 12, color: '#00d4aa', fontWeight: 600, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Approved ({approved.length})
            </div>
            {approved.map(user => <UserItem key={user.id} user={user} onApprove={approveUser} onToggleAdmin={toggleAdmin} />)}
          </>
        )}
      </div>

      <div className="card">
        <h2 className="card-title">Admin Guide</h2>
        <ul className="strategy-list">
          <li>New registrations require approval before accessing the bot</li>
          <li>Approved users can start the bot and manage settings</li>
          <li>Admin users can approve/revoke other users and grant admin rights</li>
          <li>Revoking access immediately prevents the user from logging in</li>
        </ul>
      </div>
    </>
  )
}

function UserItem({ user, onApprove, onToggleAdmin }) {
  const isApproved = user.account_status === 'approved'
  const statusLabel = user.account_status === 'approved' ? 'Approved'
    : user.account_status === 'rejected' ? 'Rejected' : 'Pending'

  return (
    <div className="user-item">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 5 }}>{user.username}</div>
          <div>
            <span className={`user-tag ${isApproved ? 'approved' : 'pending'}`}>{statusLabel}</span>
            {user.is_admin && <span className="user-tag is-admin">Admin</span>}
          </div>
        </div>
        <span style={{ fontSize: 12, color: '#4a5060' }}>#{user.id}</span>
      </div>
      <div className="user-actions">
        <button
          className={isApproved ? 'revoke' : 'approve'}
          onClick={() => onApprove(user.id, !isApproved)}
        >
          {isApproved ? 'Revoke Access' : 'Approve'}
        </button>
        <button
          className={user.is_admin ? 'revoke' : 'admin-on'}
          onClick={() => onToggleAdmin(user.id, !user.is_admin)}
        >
          {user.is_admin ? 'Remove Admin' : 'Make Admin'}
        </button>
      </div>
    </div>
  )
}

/* ─────────────────────────── Nav ─────────────────────────── */

function NavItem({ icon, label, active, onClick }) {
  return (
    <div className={`nav-item ${active ? 'active' : ''}`} onClick={onClick}>
      {icon}
      <span>{label}</span>
    </div>
  )
}

/* ─────────────────────────── Icons ─────────────────────────── */

function HomeIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
      <polyline points="9 22 9 12 15 12 15 22"></polyline>
    </svg>
  )
}

function ChartIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="20" x2="18" y2="10"></line>
      <line x1="12" y1="20" x2="12" y2="4"></line>
      <line x1="6" y1="20" x2="6" y2="14"></line>
    </svg>
  )
}

function BookIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
    </svg>
  )
}

function GearIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3"></circle>
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
    </svg>
  )
}

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8"></circle>
      <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
    </svg>
  )
}

function ShieldIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
    </svg>
  )
}

export default App
