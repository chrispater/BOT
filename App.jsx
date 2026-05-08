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

  const api = axios.create({
    baseURL: API_URL,
    headers: token ? { Authorization: `Bearer ${token}` } : {}
  })

  useEffect(() => {
    if (token) {
      fetchBotStatus()
      fetchStrategies()
      const interval = setInterval(fetchBotStatus, 5000)
      return () => clearInterval(interval)
    }
  }, [token])

  useEffect(() => {
    if (error || success) {
      const timer = setTimeout(() => {
        setError('')
        setSuccess('')
      }, 3000)
      return () => clearTimeout(timer)
    }
  }, [error, success])

  const fetchBotStatus = async () => {
    try {
      const res = await api.get('/bot/status')
      setBotStatus(res.data)
    } catch (e) {
      if (e.response?.status === 401) {
        logout()
      }
    }
  }

  const fetchStrategies = async () => {
    try {
      const res = await api.get('/strategies')
      setStrategies(res.data)
    } catch (e) {}
  }

  const logout = () => {
    localStorage.removeItem('token')
    setToken(null)
    setBotStatus(null)
  }

  if (!token) {
    return <AuthScreen setToken={setToken} setError={setError} error={error} />
  }

  const headerCoins = botStatus?.selected_coins?.map(c => c.split('/')[0]).join(', ') || 'Configure coins in Settings'
  const headerLeverage = botStatus?.leverage || '10'

  return (
    <div className="app">
      <header className="header">
        <h1>Crypto Trading Bot</h1>
        <p className="subtitle">{headerCoins} | {headerLeverage}x Leverage</p>
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
        {currentPage === 'trades' && <TradesPage botStatus={botStatus} />}
        {currentPage === 'strategy' && <StrategyPage strategies={strategies} api={api} />}
        {currentPage === 'optimize' && (
          <OptimizePage
            api={api}
            setError={setError}
            setSuccess={setSuccess}
          />
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
      </div>

      <nav className="nav">
        <NavItem
          icon={<HomeIcon />}
          label="Dashboard"
          active={currentPage === 'dashboard'}
          onClick={() => setCurrentPage('dashboard')}
        />
        <NavItem
          icon={<ChartIcon />}
          label="Trades"
          active={currentPage === 'trades'}
          onClick={() => setCurrentPage('trades')}
        />
        <NavItem
          icon={<BookIcon />}
          label="Strategy"
          active={currentPage === 'strategy'}
          onClick={() => setCurrentPage('strategy')}
        />
        <NavItem
          icon={<SearchIcon />}
          label="Optimize"
          active={currentPage === 'optimize'}
          onClick={() => setCurrentPage('optimize')}
        />
        <NavItem
          icon={<GearIcon />}
          label="Settings"
          active={currentPage === 'settings'}
          onClick={() => setCurrentPage('settings')}
        />
      </nav>
    </div>
  )
}

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
        <h1>Crypto Trading Bot</h1>
        <p className="subtitle">Autonomous ML-Powered Trading</p>
      </header>

      <div className="content">
        {error && <div className="error-message">{error}</div>}

        <div className="card">
          <h2 className="card-title">{isLogin ? 'Login' : 'Create Account'}</h2>

          <form onSubmit={handleSubmit}>
            <div className="input-group">
              <label>Username</label>
              <input
                type="text"
                value={username}
                onChange={e => setUsername(e.target.value)}
                placeholder="Enter username"
                required
              />
            </div>

            <div className="input-group">
              <label>Password</label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="Enter password"
                required
              />
            </div>

            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? 'Please wait...' : (isLogin ? 'Login' : 'Create Account')}
            </button>
          </form>

          <p style={{ textAlign: 'center', marginTop: 16, color: '#a0a0a0' }}>
            {isLogin ? "Don't have an account? " : "Already have an account? "}
            <span
              style={{ color: '#e94560', cursor: 'pointer' }}
              onClick={() => setIsLogin(!isLogin)}
            >
              {isLogin ? 'Sign Up' : 'Login'}
            </span>
          </p>
        </div>
      </div>
    </div>
  )
}

function DashboardPage({ botStatus, api, fetchBotStatus, setError, setSuccess }) {
  const [loading, setLoading] = useState(false)

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
  // Multi-position support: show all open positions
  const openPositions = botStatus?.positions || (botStatus?.position ? [botStatus.position] : [])

  return (
    <>
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

      {botStatus?.running && selectedCoins.length > 0 && (
        <div className="card">
          <h2 className="card-title">Signals by Coin</h2>
          {selectedCoins.map(coinSymbol => {
            const coin = coinSymbol.split('/')[0]
            const signal = coinSignals[coin]
            return (
              <div key={coin} style={{
                padding: '12px',
                marginBottom: 8,
                background: 'rgba(255,255,255,0.03)',
                borderRadius: 8,
                border: '1px solid #333'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontWeight: 600, color: '#e94560', fontSize: 16 }}>{coin}</span>
                  {signal ? (
                    <span style={{
                      padding: '4px 12px',
                      borderRadius: 4,
                      fontSize: 12,
                      fontWeight: 600,
                      background: signal.signal === 1 ? 'rgba(0,200,83,0.2)' : signal.signal === -1 ? 'rgba(255,82,82,0.2)' : 'rgba(150,150,150,0.2)',
                      color: signal.signal === 1 ? '#00c853' : signal.signal === -1 ? '#ff5252' : '#999'
                    }}>
                      {signal.signal === 1 ? 'LONG' : signal.signal === -1 ? 'SHORT' : 'HOLD'}
                    </span>
                  ) : (
                    <span style={{ color: '#666', fontSize: 12 }}>Waiting...</span>
                  )}
                </div>
                {signal && (
                  <div style={{ display: 'flex', gap: 16, fontSize: 12, color: '#a0a0a0' }}>
                    <span>Confidence: <strong style={{ color: signal.confidence >= 0.65 ? '#00c853' : '#ff9800' }}>{(signal.confidence * 100).toFixed(1)}%</strong></span>
                    <span>Price: ${signal.price?.toLocaleString()}</span>
                    <span>RSI: {signal.rsi?.toFixed(1)}</span>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      <div className="card">
        <h2 className="card-title">Performance</h2>
        <div className="stat-grid">
          <div className="stat-item">
            <div className="stat-value">${botStatus?.balance?.toLocaleString() || '10,000'}</div>
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
            <div className={`stat-value ${(botStatus?.win_rate || 0) >= 50 ? 'positive' : ''}`}>
              {(botStatus?.win_rate || 0).toFixed(1)}%
            </div>
            <div className="stat-label">Win Rate</div>
          </div>
        </div>
      </div>

      {/* Multi-position display — one card per open position */}
      {openPositions.length > 0 && (
        <div className="card">
          <h2 className="card-title">
            Open Positions ({openPositions.length})
          </h2>
          {openPositions.map((pos, idx) => {
            const coin = pos.symbol ? pos.symbol.split('/')[0] : 'Unknown'
            return (
              <div key={idx} style={{
                padding: '12px',
                marginBottom: idx < openPositions.length - 1 ? 8 : 0,
                background: 'rgba(255,255,255,0.03)',
                borderRadius: 8,
                border: `1px solid ${pos.side === 'long' ? 'rgba(0,200,83,0.3)' : 'rgba(255,82,82,0.3)'}`
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontWeight: 700, color: '#e94560', fontSize: 15 }}>{coin}</span>
                  <span style={{
                    padding: '3px 10px',
                    borderRadius: 4,
                    fontSize: 12,
                    fontWeight: 700,
                    background: pos.side === 'long' ? 'rgba(0,200,83,0.2)' : 'rgba(255,82,82,0.2)',
                    color: pos.side === 'long' ? '#00c853' : '#ff5252',
                    textTransform: 'uppercase'
                  }}>
                    {pos.side}
                  </span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 12, color: '#a0a0a0' }}>
                  <span>Entry: <strong style={{ color: '#fff' }}>${pos.entry_price?.toLocaleString()}</strong></span>
                  <span>Size: <strong style={{ color: '#fff' }}>{pos.size?.toFixed(4)}</strong></span>
                  <span>Margin: <strong style={{ color: '#fff' }}>${pos.margin?.toFixed(2)}</strong></span>
                  {pos.high_water_mark && pos.side === 'long' && (
                    <span>High: <strong style={{ color: '#00c853' }}>${pos.high_water_mark?.toLocaleString()}</strong></span>
                  )}
                  {pos.low_water_mark && pos.side === 'short' && (
                    <span>Low: <strong style={{ color: '#ff5252' }}>${pos.low_water_mark?.toLocaleString()}</strong></span>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </>
  )
}

function TradesPage({ botStatus }) {
  const trades = botStatus?.recent_trades || []

  return (
    <div className="card">
      <h2 className="card-title">Recent Trades</h2>
      {trades.length === 0 ? (
        <p style={{ color: '#a0a0a0', textAlign: 'center', padding: 20 }}>
          No trades yet. Start the bot to begin trading.
        </p>
      ) : (
        <div className="trade-list">
          {trades.slice().reverse().map((trade, i) => (
            <div key={i} className="trade-item">
              <div>
                <div className={`trade-side ${trade.side}`}>
                  {trade.type.toUpperCase()} {trade.side?.toUpperCase()}
                </div>
                <div style={{ fontSize: 12, color: '#a0a0a0' }}>
                  {trade.symbol ? trade.symbol.split('/')[0] : ''} ${trade.price?.toLocaleString()}
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                {trade.pnl !== undefined && (
                  <div className={`trade-pnl ${trade.pnl >= 0 ? 'positive' : 'negative'}`}>
                    {trade.pnl >= 0 ? '+' : ''}${trade.pnl?.toFixed(2)}
                  </div>
                )}
                <div style={{ fontSize: 12, color: '#a0a0a0' }}>
                  {trade.reason || (trade.confidence ? (trade.confidence * 100).toFixed(0) + '% conf' : '')}
                </div>
              </div>
            </div>
          ))}
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
    return (
      <div className="loading">
        <div className="spinner"></div>
      </div>
    )
  }

  return (
    <>
      <div className="card">
        <h2 className="card-title">Backtest Strategy</h2>
        <p style={{ color: '#a0a0a0', fontSize: 14, marginBottom: 16 }}>
          Run a historical backtest using the same ML pipeline and trade logic as live trading. Includes fees, cooldown, trailing stops, and confidence thresholds.
        </p>
        <button
          className="btn btn-primary"
          onClick={runBacktest}
          disabled={backtestLoading}
        >
          {backtestLoading ? 'Running Backtest...' : 'Run Backtest'}
        </button>

        {backtestError && (
          <div className="error-message" style={{ marginTop: 16 }}>{backtestError}</div>
        )}

        {backtestResults && (
          <div style={{ marginTop: 16 }}>
            <div style={{ marginBottom: 12, padding: '10px 12px', background: 'rgba(233, 69, 96, 0.1)', borderRadius: 6, border: '1px solid rgba(233, 69, 96, 0.3)' }}>
              <div style={{ fontSize: 12, color: '#a0a0a0', marginBottom: 4 }}>
                {backtestResults.leverage}x lev | {(backtestResults.risk_per_trade * 100).toFixed(1)}% risk |{' '}
                {(backtestResults.stop_loss_pct * 100).toFixed(1)}% SL | {(backtestResults.take_profit_pct * 100).toFixed(1)}% TP |{' '}
                {backtestResults.trailing_stop_pct ? `${(backtestResults.trailing_stop_pct * 100).toFixed(1)}% trail | ` : ''}
                {(backtestResults.min_confidence * 100).toFixed(0)}% conf | {backtestResults.timeframe || '5m'}
              </div>
              <div style={{ fontSize: 12, color: '#e94560', fontWeight: 600 }}>
                Coins tested: {backtestResults.selected_coins?.map(c => c.split('/')[0]).join(', ')}
              </div>
            </div>

            <h3 style={{ fontSize: 14, color: '#fff', marginBottom: 12, marginTop: 16 }}>Overall Results ({backtestResults.period_days} days)</h3>
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
                <div className={`stat-value ${backtestResults.win_rate >= 50 ? 'positive' : ''}`}>
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
                <span className="indicator-value" style={{ color: '#ff9800' }}>-${backtestResults.total_fees?.toFixed(2) || '0.00'}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {backtestResults?.coin_results && (
        <div className="card">
          <h2 className="card-title">Results by Coin</h2>
          {backtestResults.coin_results.map((coinResult, idx) => (
            <div key={idx} style={{
              padding: '12px',
              marginBottom: 8,
              background: 'rgba(255,255,255,0.03)',
              borderRadius: 8,
              border: expandedCoin === coinResult.coin ? '1px solid #e94560' : '1px solid #333',
              cursor: 'pointer'
            }} onClick={() => setExpandedCoin(expandedCoin === coinResult.coin ? null : coinResult.coin)}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 600, color: '#e94560', fontSize: 16 }}>{coinResult.coin}</span>
                <span className={coinResult.total_pnl >= 0 ? 'positive' : 'negative'} style={{ fontWeight: 600 }}>
                  {coinResult.total_pnl >= 0 ? '+' : ''}${coinResult.total_pnl?.toFixed(2)}
                </span>
              </div>
              {coinResult.error ? (
                <div style={{ color: '#ff9800', fontSize: 12, marginTop: 4 }}>{coinResult.error}</div>
              ) : (
                <div style={{ display: 'flex', gap: 12, fontSize: 12, color: '#a0a0a0', marginTop: 6 }}>
                  <span>Return: <strong className={coinResult.total_return >= 0 ? 'positive' : 'negative'}>{coinResult.total_return >= 0 ? '+' : ''}{coinResult.total_return?.toFixed(1)}%</strong></span>
                  <span>Trades: {coinResult.total_trades}</span>
                  <span>Win: {coinResult.win_rate?.toFixed(0)}%</span>
                </div>
              )}

              {expandedCoin === coinResult.coin && coinResult.trades && coinResult.trades.length > 0 && (
                <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid #333' }}>
                  <div style={{ fontSize: 12, color: '#a0a0a0', marginBottom: 8 }}>Trade Log ({coinResult.trades.length} trades)</div>
                  {coinResult.trades.map((trade, tIdx) => (
                    <div key={tIdx} style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      padding: '6px 8px',
                      background: 'rgba(0,0,0,0.2)',
                      borderRadius: 4,
                      marginBottom: 4,
                      fontSize: 11
                    }}>
                      <span style={{ color: trade.side === 'long' ? '#00c853' : '#ff5252', textTransform: 'uppercase', fontWeight: 600 }}>
                        {trade.side}
                      </span>
                      <span style={{ color: '#666' }}>Entry: ${trade.entry}</span>
                      <span style={{ color: '#666' }}>Exit: ${trade.exit}</span>
                      <span style={{ fontSize: 10, color: '#555' }}>{trade.reason}</span>
                      <span className={trade.pnl >= 0 ? 'positive' : 'negative'} style={{ fontWeight: 600 }}>
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

      {backtestResults?.all_trades && backtestResults.all_trades.length > 0 && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <h2 className="card-title" style={{ margin: 0 }}>All Trades Log</h2>
            <button
              onClick={() => setShowAllTrades(!showAllTrades)}
              style={{
                padding: '6px 12px',
                background: 'rgba(233,69,96,0.2)',
                border: '1px solid #e94560',
                borderRadius: 4,
                color: '#e94560',
                cursor: 'pointer',
                fontSize: 12
              }}
            >
              {showAllTrades ? 'Hide' : `Show ${backtestResults.all_trades.length} trades`}
            </button>
          </div>

          {showAllTrades && (
            <div style={{ maxHeight: 300, overflowY: 'auto' }}>
              {backtestResults.all_trades.map((trade, idx) => (
                <div key={idx} style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  padding: '8px',
                  background: idx % 2 === 0 ? 'rgba(0,0,0,0.2)' : 'transparent',
                  borderRadius: 4,
                  fontSize: 12,
                  alignItems: 'center'
                }}>
                  <span style={{ color: '#e94560', fontWeight: 600, minWidth: 40 }}>{trade.coin}</span>
                  <span style={{ color: trade.side === 'long' ? '#00c853' : '#ff5252', textTransform: 'uppercase', minWidth: 50 }}>
                    {trade.side}
                  </span>
                  <span style={{ color: '#666' }}>${trade.entry} → ${trade.exit}</span>
                  <span className={trade.pnl >= 0 ? 'positive' : 'negative'} style={{ fontWeight: 600, minWidth: 70, textAlign: 'right' }}>
                    {trade.pnl >= 0 ? '+' : ''}${trade.pnl} ({trade.pnl_pct}%)
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="card">
        <h2 style={{ fontSize: 20, marginBottom: 8 }}>{strategies.name}</h2>
        <p style={{ color: '#a0a0a0', fontSize: 14 }}>{strategies.description}</p>
      </div>

      {strategies.components?.map((component, i) => (
        <div key={i} className="card strategy-section">
          <div className="strategy-title">
            {component.name}
            <span className="strategy-weight">{component.weight}</span>
          </div>
          <p className="strategy-desc">{component.description}</p>

          {component.details && (
            <ul className="strategy-list">
              {component.details.map((detail, j) => (
                <li key={j}>{detail}</li>
              ))}
            </ul>
          )}

          {component.indicators && (
            <ul className="strategy-list">
              {component.indicators.map((ind, j) => (
                <li key={j}><strong>{ind.name}:</strong> {ind.desc}</li>
              ))}
            </ul>
          )}
        </div>
      ))}
    </>
  )
}

const AVAILABLE_COINS = [
  'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
  'DOGE/USDT:USDT', 'BNB/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT',
  'LINK/USDT:USDT', 'MATIC/USDT:USDT', 'DOT/USDT:USDT', 'UNI/USDT:USDT',
  'SHIB/USDT:USDT', 'LTC/USDT:USDT', 'ATOM/USDT:USDT'
]

function SettingsPage({ api, logout, setError, setSuccess, botStatus }) {
  const [apiKey, setApiKey] = useState('')
  const [apiSecret, setApiSecret] = useState('')
  const [apiPassword, setApiPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [hasCredentials, setHasCredentials] = useState(false)

  // Core trading settings
  const [startingBalance, setStartingBalance] = useState(10000)
  const [leverage, setLeverage] = useState(10)
  const [selectedCoins, setSelectedCoins] = useState(['BTC/USDT:USDT'])
  const [riskPerTrade, setRiskPerTrade] = useState(2)
  const [stopLossPct, setStopLossPct] = useState(1.5)
  const [takeProfitPct, setTakeProfitPct] = useState(3)
  const [tradeCooldown, setTradeCooldown] = useState(5)
  const [minConfidence, setMinConfidence] = useState(65)
  const [timeframe, setTimeframe] = useState('5m')

  // Compounding & protection settings
  const [trailingStopPct, setTrailingStopPct] = useState(1.0)
  const [maxDrawdownPct, setMaxDrawdownPct] = useState(20)
  const [retrainEvery, setRetrainEvery] = useState(50)
  const [profitRiskMultiplier, setProfitRiskMultiplier] = useState(1.5)

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
      setStopLossPct((d.stop_loss_pct || 0.015) * 100)
      setTakeProfitPct((d.take_profit_pct || 0.03) * 100)
      setTradeCooldown((d.trade_cooldown || 300) / 60)
      setMinConfidence((d.min_confidence || 0.65) * 100)
      setTimeframe(d.timeframe || '5m')
      setTrailingStopPct((d.trailing_stop_pct || 0.01) * 100)
      setMaxDrawdownPct((d.max_drawdown_pct || 0.20) * 100)
      setRetrainEvery(d.retrain_every || 50)
      setProfitRiskMultiplier(d.profit_risk_multiplier || 1.5)
    } catch (e) {}
  }

  const saveCredentials = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      await api.post('/credentials', {
        api_key: apiKey,
        api_secret: apiSecret,
        api_password: apiPassword
      })
      setSuccess('API credentials saved securely')
      setApiKey('')
      setApiSecret('')
      setApiPassword('')
      checkCredentials()
    } catch (e) {
      setError('Failed to save credentials')
    }
    setLoading(false)
  }

  const saveSettings = async () => {
    if (isBotRunning) {
      setError('Stop the bot before changing settings')
      return
    }

    const validatedRisk = Math.max(0.1, Math.min(10, riskPerTrade))
    const validatedSL = Math.max(0.1, Math.min(10, stopLossPct))
    const validatedTP = Math.max(0.1, Math.min(20, takeProfitPct))
    const validatedCooldown = Math.max(1, Math.min(60, Math.round(tradeCooldown)))
    const validatedConfidence = Math.max(50, Math.min(95, Math.round(minConfidence)))
    const validatedTrail = Math.max(0.1, Math.min(5, trailingStopPct))
    const validatedDrawdown = Math.max(5, Math.min(50, maxDrawdownPct))
    const validatedRetrain = Math.max(10, Math.min(500, Math.round(retrainEvery)))
    const validatedMultiplier = Math.max(1.0, Math.min(3.0, profitRiskMultiplier))

    setRiskPerTrade(validatedRisk)
    setStopLossPct(validatedSL)
    setTakeProfitPct(validatedTP)
    setTradeCooldown(validatedCooldown)
    setMinConfidence(validatedConfidence)
    setTrailingStopPct(validatedTrail)
    setMaxDrawdownPct(validatedDrawdown)
    setRetrainEvery(validatedRetrain)
    setProfitRiskMultiplier(validatedMultiplier)

    setSettingsLoading(true)
    try {
      await api.put('/settings', {
        starting_balance: startingBalance,
        leverage: leverage,
        selected_coins: selectedCoins,
        risk_per_trade: validatedRisk / 100,
        stop_loss_pct: validatedSL / 100,
        take_profit_pct: validatedTP / 100,
        trade_cooldown: validatedCooldown * 60,
        min_confidence: validatedConfidence / 100,
        timeframe: timeframe,
        trailing_stop_pct: validatedTrail / 100,
        max_drawdown_pct: validatedDrawdown / 100,
        retrain_every: validatedRetrain,
        profit_risk_multiplier: validatedMultiplier,
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
      if (selectedCoins.length > 1) {
        setSelectedCoins(selectedCoins.filter(c => c !== coin))
      }
    } else if (selectedCoins.length < 5) {
      setSelectedCoins([...selectedCoins, coin])
    } else {
      setError('Maximum 5 coins allowed')
    }
  }

  const getCoinDisplayName = (coin) => coin.split('/')[0]

  const inputStyle = (disabled) => ({
    opacity: disabled ? 0.5 : 1
  })
  const selectStyle = (disabled) => ({
    opacity: disabled ? 0.5 : 1,
    width: '100%',
    padding: '12px',
    borderRadius: 8,
    border: '1px solid #333',
    background: '#1a1a2e',
    color: '#fff'
  })

  return (
    <>
      {/* ── Trading Settings ── */}
      <div className="card">
        <h2 className="card-title">Trading Settings</h2>

        {isBotRunning && (
          <div className="error-message" style={{ marginBottom: 16 }}>
            Stop the bot to change trading settings
          </div>
        )}

        <div className="input-group">
          <label>Starting Balance (USDT)</label>
          <input
            type="number"
            value={startingBalance}
            onChange={e => setStartingBalance(Number(e.target.value))}
            min="100"
            disabled={isBotRunning}
            style={inputStyle(isBotRunning)}
          />
        </div>

        <div className="input-group">
          <label>Leverage (1x - 100x)</label>
          <select
            value={leverage}
            onChange={e => setLeverage(Number(e.target.value))}
            disabled={isBotRunning}
            style={selectStyle(isBotRunning)}
          >
            {[1, 2, 3, 5, 10, 15, 20, 25, 50, 75, 100].map(lev => (
              <option key={lev} value={lev}>{lev}x</option>
            ))}
          </select>
        </div>

        <div className="input-group">
          <label>Timeframe</label>
          <select
            value={timeframe}
            onChange={e => setTimeframe(e.target.value)}
            disabled={isBotRunning}
            style={selectStyle(isBotRunning)}
          >
            {TIMEFRAME_OPTIONS.map(tf => (
              <option key={tf} value={tf}>{tf}</option>
            ))}
          </select>
          <p style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
            Candle timeframe for analysis (uses 1000 candles)
          </p>
        </div>

        <div className="input-group">
          <label>Trading Coins (max 5)</label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
            {AVAILABLE_COINS.map(coin => (
              <div
                key={coin}
                onClick={() => toggleCoin(coin)}
                style={{
                  padding: '8px 12px',
                  borderRadius: 6,
                  border: selectedCoins.includes(coin) ? '2px solid #e94560' : '1px solid #333',
                  background: selectedCoins.includes(coin) ? 'rgba(233, 69, 96, 0.2)' : '#1a1a2e',
                  color: selectedCoins.includes(coin) ? '#e94560' : '#a0a0a0',
                  cursor: isBotRunning ? 'not-allowed' : 'pointer',
                  opacity: isBotRunning ? 0.5 : 1,
                  fontSize: 14,
                  fontWeight: selectedCoins.includes(coin) ? 600 : 400
                }}
              >
                {getCoinDisplayName(coin)}
              </div>
            ))}
          </div>
          <p style={{ fontSize: 12, color: '#666', marginTop: 8 }}>
            Selected: {selectedCoins.map(getCoinDisplayName).join(', ')}
          </p>
        </div>

        <button
          className="btn btn-primary"
          onClick={saveSettings}
          disabled={settingsLoading || isBotRunning}
        >
          {settingsLoading ? 'Saving...' : 'Save Trading Settings'}
        </button>
      </div>

      {/* ── Risk Management ── */}
      <div className="card">
        <h2 className="card-title">Risk Management</h2>

        {isBotRunning && (
          <div className="error-message" style={{ marginBottom: 16 }}>
            Stop the bot to change risk settings
          </div>
        )}

        <div className="input-group">
          <label>Risk Per Trade (%)</label>
          <input
            type="number"
            value={riskPerTrade}
            onChange={e => setRiskPerTrade(Number(e.target.value))}
            min="0.1" max="10" step="0.1"
            disabled={isBotRunning}
            style={inputStyle(isBotRunning)}
          />
          <p style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
            Percentage of balance risked per trade (0.1% - 10%)
          </p>
        </div>

        <div className="input-group">
          <label>Stop Loss (%)</label>
          <input
            type="number"
            value={stopLossPct}
            onChange={e => setStopLossPct(Number(e.target.value))}
            min="0.1" max="10" step="0.1"
            disabled={isBotRunning}
            style={inputStyle(isBotRunning)}
          />
          <p style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
            Exit position if loss exceeds this percentage
          </p>
        </div>

        <div className="input-group">
          <label>Take Profit (%)</label>
          <input
            type="number"
            value={takeProfitPct}
            onChange={e => setTakeProfitPct(Number(e.target.value))}
            min="0.1" max="20" step="0.1"
            disabled={isBotRunning}
            style={inputStyle(isBotRunning)}
          />
          <p style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
            Exit position when profit reaches this percentage
          </p>
        </div>

        <div className="input-group">
          <label>Trade Cooldown (minutes)</label>
          <select
            value={tradeCooldown}
            onChange={e => setTradeCooldown(Number(e.target.value))}
            disabled={isBotRunning}
            style={selectStyle(isBotRunning)}
          >
            {[1, 2, 3, 5, 10, 15, 20, 30, 60].map(min => (
              <option key={min} value={min}>{min} min</option>
            ))}
          </select>
          <p style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
            Wait time between trades per coin
          </p>
        </div>

        <div className="input-group">
          <label>Minimum Confidence (%)</label>
          <input
            type="number"
            value={minConfidence}
            onChange={e => setMinConfidence(Number(e.target.value))}
            min="50" max="95" step="1"
            disabled={isBotRunning}
            style={inputStyle(isBotRunning)}
          />
          <p style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
            Only trade when signal confidence exceeds this (50% - 95%)
          </p>
        </div>

        <button
          className="btn btn-primary"
          onClick={saveSettings}
          disabled={settingsLoading || isBotRunning}
        >
          {settingsLoading ? 'Saving...' : 'Save Risk Settings'}
        </button>
      </div>

      {/* ── Compounding & Protection ── */}
      <div className="card">
        <h2 className="card-title">Compounding & Protection</h2>
        <p style={{ fontSize: 13, color: '#a0a0a0', marginBottom: 16 }}>
          These settings control how profits compound and how the bot protects capital.
        </p>

        {isBotRunning && (
          <div className="error-message" style={{ marginBottom: 16 }}>
            Stop the bot to change these settings
          </div>
        )}

        <div className="input-group">
          <label>Trailing Stop (%)</label>
          <input
            type="number"
            value={trailingStopPct}
            onChange={e => setTrailingStopPct(Number(e.target.value))}
            min="0.1" max="5" step="0.1"
            disabled={isBotRunning}
            style={inputStyle(isBotRunning)}
          />
          <p style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
            Once in profit, exit if price drops this % from its peak. Locks in gains on trending moves. (0.1% – 5%)
          </p>
        </div>

        <div className="input-group">
          <label>Max Drawdown Circuit Breaker (%)</label>
          <input
            type="number"
            value={maxDrawdownPct}
            onChange={e => setMaxDrawdownPct(Number(e.target.value))}
            min="5" max="50" step="1"
            disabled={isBotRunning}
            style={inputStyle(isBotRunning)}
          />
          <p style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
            Stop opening new positions if account drawdown exceeds this. Prevents runaway losses. (5% – 50%)
          </p>
        </div>

        <div className="input-group">
          <label>Profit Risk Multiplier</label>
          <input
            type="number"
            value={profitRiskMultiplier}
            onChange={e => setProfitRiskMultiplier(Number(e.target.value))}
            min="1.0" max="3.0" step="0.1"
            disabled={isBotRunning}
            style={inputStyle(isBotRunning)}
          />
          <p style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
            Apply this risk multiplier to profits above starting balance. 1.5 = 50% more aggressive on house money. (1.0 – 3.0)
          </p>
        </div>

        <div className="input-group">
          <label>Model Retrain Interval (cycles)</label>
          <input
            type="number"
            value={retrainEvery}
            onChange={e => setRetrainEvery(Number(e.target.value))}
            min="10" max="500" step="10"
            disabled={isBotRunning}
            style={inputStyle(isBotRunning)}
          />
          <p style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
            Retrain the ML model every N cycles to adapt to current market conditions. Lower = more adaptive, slower. (10 – 500)
          </p>
        </div>

        <button
          className="btn btn-primary"
          onClick={saveSettings}
          disabled={settingsLoading || isBotRunning}
        >
          {settingsLoading ? 'Saving...' : 'Save Compounding Settings'}
        </button>
      </div>

      {/* ── API Credentials ── */}
      <div className="card">
        <h2 className="card-title">Blofin API Credentials</h2>

        {hasCredentials && (
          <div className="success-message" style={{ marginBottom: 16 }}>
            API credentials configured
          </div>
        )}

        <form onSubmit={saveCredentials}>
          <div className="input-group">
            <label>API Key</label>
            <input
              type="text"
              value={apiKey}
              onChange={e => setApiKey(e.target.value)}
              placeholder="Enter your Blofin API key"
            />
          </div>

          <div className="input-group">
            <label>API Secret</label>
            <input
              type="password"
              value={apiSecret}
              onChange={e => setApiSecret(e.target.value)}
              placeholder="Enter your API secret"
            />
          </div>

          <div className="input-group">
            <label>API Password (Passphrase)</label>
            <input
              type="password"
              value={apiPassword}
              onChange={e => setApiPassword(e.target.value)}
              placeholder="Enter your API passphrase"
            />
          </div>

          <button type="submit" className="btn btn-primary" disabled={loading || (!apiKey && !apiSecret)}>
            {loading ? 'Saving...' : 'Save Credentials'}
          </button>
        </form>
      </div>

      {/* ── Account ── */}
      <div className="card">
        <h2 className="card-title">Account</h2>
        <button className="btn btn-secondary" onClick={logout}>
          Logout
        </button>
      </div>

      <div className="card">
        <h2 className="card-title">Install App</h2>
        <p style={{ color: '#a0a0a0', fontSize: 14, marginBottom: 16 }}>
          On Android: Open browser menu and tap "Add to Home Screen" to install this app on your device.
        </p>
      </div>
    </>
  )
}

function OptimizePage({ api, setError, setSuccess }) {
  const [optimizeResults, setOptimizeResults] = useState(null)
  const [optimizeLoading, setOptimizeLoading] = useState(false)
  const [optimizeStatus, setOptimizeStatus] = useState('')
  const [optimizeError, setOptimizeError] = useState('')
  const [selectedConfig, setSelectedConfig] = useState(null)
  const [applyLoading, setApplyLoading] = useState(false)
  const pollIntervalRef = useRef(null)

  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
    }
  }, [])

  const pollForResults = async () => {
    try {
      const res = await api.get('/optimize/status')
      if (res.data.status === 'completed') {
        setOptimizeResults(res.data.result)
        setOptimizeLoading(false)
        setOptimizeStatus('')
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current)
          pollIntervalRef.current = null
        }
        return true
      } else if (res.data.status === 'failed') {
        setOptimizeError(res.data.error || 'Optimization failed')
        setOptimizeLoading(false)
        setOptimizeStatus('')
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current)
          pollIntervalRef.current = null
        }
        return true
      } else {
        setOptimizeStatus(res.data.message || 'Running...')
        return false
      }
    } catch (e) {
      return false
    }
  }

  const runOptimization = async () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
    }

    setOptimizeLoading(true)
    setOptimizeError('')
    setOptimizeResults(null)
    setSelectedConfig(null)
    setOptimizeStatus('Starting optimization...')

    try {
      await api.post('/optimize/start')

      pollIntervalRef.current = setInterval(async () => {
        await pollForResults()
      }, 5000)

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
        // Apply optimized trailing stop if present, otherwise keep existing
        ...(config.trailing_stop_pct != null && { trailing_stop_pct: config.trailing_stop_pct }),
      })
      setSuccess(`Applied: ${config.timeframe} | ${config.leverage}x | ${(config.min_confidence * 100).toFixed(0)}% conf`)
      setSelectedConfig(config)
    } catch (e) {
      setError('Failed to apply settings - stop bot first')
    }
    setApplyLoading(false)
  }

  const formatCooldown = (seconds) => {
    if (seconds < 60) return `${seconds}s`
    return `${Math.round(seconds / 60)}m`
  }

  const resetOptimization = async () => {
    try {
      await api.post('/optimize/reset')
      setOptimizeLoading(false)
      setOptimizeStatus('')
      setOptimizeError('')
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
        pollIntervalRef.current = null
      }
      setSuccess('Optimization reset - you can start a new one')
    } catch (e) {
      setError('Failed to reset optimization')
    }
  }

  return (
    <>
      <div className="card">
        <h2 className="card-title">Parameter Optimizer</h2>
        <p style={{ color: '#a0a0a0', fontSize: 14, marginBottom: 16 }}>
          Automatically find the best trading parameters for your selected coins. Tests ~1,350 configurations across all timeframes including trailing stop distances. Returns top performers ranked by ROI, win rate, and trade count.
        </p>
        <p style={{ color: '#e94560', fontSize: 12, marginBottom: 16 }}>
          Warning: This can take 5-15 minutes to complete.
        </p>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            className="btn btn-primary"
            onClick={runOptimization}
            disabled={optimizeLoading}
          >
            {optimizeLoading ? (optimizeStatus || 'Optimizing...') : 'Find Best Parameters'}
          </button>
          {optimizeLoading && (
            <button
              className="btn"
              onClick={resetOptimization}
              style={{ background: '#e94560' }}
            >
              Reset
            </button>
          )}
        </div>

        {optimizeError && (
          <div className="error-message" style={{ marginTop: 16 }}>{optimizeError}</div>
        )}
      </div>

      {optimizeResults && (
        <div className="card">
          <h2 className="card-title">Top Configurations</h2>
          <div style={{ fontSize: 12, color: '#a0a0a0', marginBottom: 12 }}>
            Tested {optimizeResults.total_tested} configurations | {optimizeResults.valid_configs} profitable | {optimizeResults.days_tested} days backtest
          </div>
          <div style={{ fontSize: 12, color: '#a0a0a0', marginBottom: 16 }}>
            Coins: {optimizeResults.selected_coins?.map(c => c.split('/')[0]).join(', ')}
          </div>

          {optimizeResults.top_configs?.length === 0 ? (
            <p style={{ color: '#e94560' }}>No profitable configurations found. Try different coins.</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {optimizeResults.top_configs?.map((config, i) => (
                <div
                  key={i}
                  style={{
                    padding: 12,
                    background: selectedConfig === config ? 'rgba(38, 166, 154, 0.2)' : 'rgba(233, 69, 96, 0.1)',
                    borderRadius: 8,
                    border: selectedConfig === config ? '1px solid #26a69a' : '1px solid rgba(233, 69, 96, 0.3)'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <span style={{ fontSize: 14, fontWeight: 600, color: '#e94560' }}>
                      #{i + 1} — {config.timeframe}
                    </span>
                    <span style={{
                      fontSize: 14,
                      fontWeight: 700,
                      color: config.total_return >= 0 ? '#26a69a' : '#e94560'
                    }}>
                      {config.total_return >= 0 ? '+' : ''}{config.total_return.toFixed(1)}% ROI
                    </span>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 12, color: '#a0a0a0', marginBottom: 8 }}>
                    <span>Leverage: {config.leverage}x</span>
                    <span>Risk: {(config.risk_per_trade * 100).toFixed(0)}%</span>
                    <span>SL: {(config.stop_loss_pct * 100).toFixed(2)}%</span>
                    <span>TP: {(config.take_profit_pct * 100).toFixed(2)}%</span>
                    <span>Trail: {config.trailing_stop_pct != null ? `${(config.trailing_stop_pct * 100).toFixed(1)}%` : '—'}</span>
                    <span>Conf: {(config.min_confidence * 100).toFixed(0)}%</span>
                    <span>Cooldown: {formatCooldown(config.trade_cooldown)}</span>
                    <span>Score: {config.score?.toFixed(3)}</span>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 12 }}>
                    <span style={{ color: '#26a69a' }}>
                      {config.win_rate.toFixed(1)}% win | {config.total_trades} trades | ${config.total_pnl.toFixed(0)} PnL
                    </span>
                    <button
                      className="btn btn-secondary"
                      style={{ padding: '4px 12px', fontSize: 11 }}
                      onClick={() => applyConfig(config)}
                      disabled={applyLoading}
                    >
                      Apply
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="card">
        <h2 className="card-title">How It Works</h2>
        <ul className="strategy-list">
          <li><strong>Smart Search:</strong> Tests 100 random parameter combinations per timeframe</li>
          <li><strong>Trailing Stop Search:</strong> Also optimises trailing stop distance (0.5%–2%)</li>
          <li><strong>Composite Score:</strong> Ranks by 50% ROI + 30% Win Rate + 20% Trade Count</li>
          <li><strong>Minimum Trades:</strong> Rejects configs with fewer than 15 trades (avoids overfitting)</li>
          <li><strong>Same ML Pipeline:</strong> Uses identical logic as live trading for accuracy</li>
        </ul>
      </div>
    </>
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

function HomeIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
      <polyline points="9 22 9 12 15 12 15 22"></polyline>
    </svg>
  )
}

function ChartIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="18" y1="20" x2="18" y2="10"></line>
      <line x1="12" y1="20" x2="12" y2="4"></line>
      <line x1="6" y1="20" x2="6" y2="14"></line>
    </svg>
  )
}

function BookIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
    </svg>
  )
}

function GearIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="3"></circle>
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
    </svg>
  )
}

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="11" cy="11" r="8"></circle>
      <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
    </svg>
  )
}

export default App
