import React, { useEffect, useState } from 'react'
import { Card, InfoDot, ToggleSwitch } from '../components/Primitives'

const AVAILABLE_COINS = [
  'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
  'DOGE/USDT:USDT', 'BNB/USDT:USDT', 'ADA/USDT:USDT', 'AVAX/USDT:USDT',
  'LINK/USDT:USDT', 'MATIC/USDT:USDT', 'DOT/USDT:USDT', 'UNI/USDT:USDT',
  'SHIB/USDT:USDT', 'LTC/USDT:USDT', 'ATOM/USDT:USDT', 'XLM/USDT:USDT',
]

export default function SettingsPage({ api, logout, setError, setSuccess, botStatus, onOpenInsights }) {
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
  const [dailyLossLimit, setDailyLossLimit] = useState(8)
  const [maxPositions, setMaxPositions] = useState(3)
  const [mieGateEnabled, setMieGateEnabled] = useState(false)

  const [settingsLoading, setSettingsLoading] = useState(false)

  const TIMEFRAME_OPTIONS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h']
  const isBotRunning = botStatus?.running

  useEffect(() => { checkCredentials(); loadSettings() }, [])

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
      setDailyLossLimit(Math.round((d.daily_loss_limit || 0.08) * 100))
      setMaxPositions(d.max_positions || 3)
      setMieGateEnabled(!!d.mie_gate_enabled)
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
      risk: Math.max(0.1, Math.min(100, riskPerTrade)),
      sl: Math.max(1, Math.min(75, stopLossPct)),
      tp: Math.max(1, Math.min(200, takeProfitPct)),
      cooldown: Math.max(1, Math.min(60, Math.round(tradeCooldown))),
      conf: Math.max(50, Math.min(95, Math.round(minConfidence))),
      trail: Math.max(1, Math.min(50, trailingStopPct)),
      drawdown: Math.max(5, Math.min(50, maxDrawdownPct)),
      retrain: Math.max(10, Math.min(500, Math.round(retrainEvery))),
      multiplier: Math.max(1.0, Math.min(3.0, profitRiskMultiplier)),
      adx: Math.max(5, Math.min(30, Math.round(adxThreshold))),
      dayLoss: Math.max(1, Math.min(50, Math.round(dailyLossLimit))),
      maxPos: Math.max(1, Math.min(10, Math.round(maxPositions))),
    }
    setRiskPerTrade(v.risk); setStopLossPct(v.sl); setTakeProfitPct(v.tp)
    setTradeCooldown(v.cooldown); setMinConfidence(v.conf)
    setTrailingStopPct(v.trail); setMaxDrawdownPct(v.drawdown)
    setRetrainEvery(v.retrain); setProfitRiskMultiplier(v.multiplier)
    setAdxThreshold(v.adx)
    setDailyLossLimit(v.dayLoss); setMaxPositions(v.maxPos)

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
        daily_loss_limit: v.dayLoss / 100,
        max_positions: v.maxPos,
        mie_gate_enabled: mieGateEnabled,
      })
      setSuccess('Settings saved')
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to save settings')
    }
    setSettingsLoading(false)
  }

  const saveMieToggle = async (next) => {
    setMieGateEnabled(next)
    if (isBotRunning) { setError('Stop the bot before changing this'); setMieGateEnabled(!next); return }
    try {
      await api.put('/settings', { mie_gate_enabled: next })
      setSuccess(next ? 'Market Intelligence gate enabled' : 'Market Intelligence gate disabled')
    } catch (e) {
      setMieGateEnabled(!next)
      setError('Failed to update')
    }
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
    borderRadius: 12, border: '1px solid var(--border)', background: 'var(--bg-input)',
    color: 'var(--text-primary)', fontSize: 15, outline: 'none',
  })

  const BotRunningWarning = () => isBotRunning ? (
    <div className="error-message" style={{ marginBottom: 16 }}>Stop the bot to change settings</div>
  ) : null

  return (
    <>
      {/* ── Trading Mode ── */}
      <Card title="Trading Mode">
        <BotRunningWarning />
        <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
          <button onClick={() => !isBotRunning && setSimulationMode(true)} style={{
            flex: 1, padding: 12, borderRadius: 10, border: simulationMode ? '2px solid var(--accent-gold)' : '1px solid var(--border)',
            background: simulationMode ? 'rgba(245,166,35,0.12)' : 'transparent',
            color: simulationMode ? 'var(--accent-gold)' : 'var(--text-secondary)', cursor: isBotRunning ? 'not-allowed' : 'pointer',
            fontWeight: 600, fontSize: 14, transition: 'all 0.2s',
          }}>Simulation</button>
          <button onClick={() => !isBotRunning && setSimulationMode(false)} style={{
            flex: 1, padding: 12, borderRadius: 10, border: !simulationMode ? '2px solid var(--accent-red)' : '1px solid var(--border)',
            background: !simulationMode ? 'rgba(255,68,68,0.12)' : 'transparent',
            color: !simulationMode ? 'var(--accent-red)' : 'var(--text-secondary)', cursor: isBotRunning ? 'not-allowed' : 'pointer',
            fontWeight: 600, fontSize: 14, transition: 'all 0.2s',
          }}>Live Trading</button>
        </div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', textAlign: 'center' }}>
          {simulationMode ? 'Paper trading — no real funds at risk' : 'LIVE — real funds will be used. Trade with caution.'}
        </p>
        <button className="btn btn-primary" onClick={saveSettings} disabled={settingsLoading || isBotRunning} style={{ marginTop: 14 }}>
          {settingsLoading ? 'Saving...' : 'Save Mode'}
        </button>
      </Card>

      {/* ── Trading Settings ── */}
      <Card title="Trading Settings">
        <BotRunningWarning />
        <div className="input-group">
          <label>Starting Balance (USDT)</label>
          <input type="number" value={startingBalance} onChange={e => setStartingBalance(Number(e.target.value))} min="100" disabled={isBotRunning} style={dis(isBotRunning)} />
        </div>
        <div className="input-group">
          <label>Leverage</label>
          <select value={leverage} onChange={e => setLeverage(Number(e.target.value))} disabled={isBotRunning} style={sel(isBotRunning)}>
            {[1, 2, 3, 5, 8, 10].map(lev => <option key={lev} value={lev}>{lev}x</option>)}
          </select>
        </div>
        <div className="input-group">
          <label>Timeframe</label>
          <select value={timeframe} onChange={e => setTimeframe(e.target.value)} disabled={isBotRunning} style={sel(isBotRunning)}>
            {TIMEFRAME_OPTIONS.map(tf => <option key={tf} value={tf}>{tf}</option>)}
          </select>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>Candle timeframe for analysis (uses 1000 candles)</p>
        </div>
        <div className="input-group">
          <label>Trading Coins (max 5)</label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
            {AVAILABLE_COINS.map(coin => {
              const active = selectedCoins.includes(coin)
              return (
                <div key={coin} onClick={() => toggleCoin(coin)} style={{
                  padding: '7px 12px', borderRadius: 8,
                  border: active ? '2px solid var(--accent)' : '1px solid var(--border)',
                  background: active ? 'var(--accent-dim)' : 'var(--bg-card-alt)',
                  color: active ? 'var(--accent)' : 'var(--text-secondary)',
                  cursor: isBotRunning ? 'not-allowed' : 'pointer', opacity: isBotRunning ? 0.45 : 1,
                  fontSize: 13, fontWeight: active ? 700 : 400, transition: 'all 0.15s',
                }}>{coin.split('/')[0]}</div>
              )
            })}
          </div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>Selected: {selectedCoins.map(c => c.split('/')[0]).join(', ')}</p>
        </div>
        <button className="btn btn-primary" onClick={saveSettings} disabled={settingsLoading || isBotRunning}>
          {settingsLoading ? 'Saving...' : 'Save Trading Settings'}
        </button>
      </Card>

      {/* ── Risk Management ── */}
      <Card title="Risk Management">
        <BotRunningWarning />
        <div className="input-group">
          <label>Risk Per Trade (%)</label>
          <input type="number" value={riskPerTrade} onChange={e => setRiskPerTrade(Number(e.target.value))} min="0.1" max="100" step="0.1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>Percentage of balance per trade (0.1%–100%)</p>
        </div>
        <div className="input-group">
          <label>Stop Loss (% of margin)</label>
          <input type="number" value={stopLossPct} onChange={e => setStopLossPct(Number(e.target.value))} min="1" max="75" step="1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>Exit when margin lost ≥ this %. At {leverage}x leverage → {(stopLossPct / leverage).toFixed(2)}% price move.</p>
        </div>
        <div className="input-group">
          <label>Take Profit (% of margin)</label>
          <input type="number" value={takeProfitPct} onChange={e => setTakeProfitPct(Number(e.target.value))} min="1" max="200" step="1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>Exit when margin gained ≥ this %. At {leverage}x leverage → {(takeProfitPct / leverage).toFixed(2)}% price move.</p>
        </div>
        <div className="input-group">
          <label>Trade Cooldown (minutes)</label>
          <select value={tradeCooldown} onChange={e => setTradeCooldown(Number(e.target.value))} disabled={isBotRunning} style={sel(isBotRunning)}>
            {[1, 2, 3, 5, 10, 15, 20, 30, 60].map(min => <option key={min} value={min}>{min} min</option>)}
          </select>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>Wait time between trades per coin</p>
        </div>
        <div className="input-group">
          <label>Minimum Confidence (%)</label>
          <input type="number" value={minConfidence} onChange={e => setMinConfidence(Number(e.target.value))} min="50" max="95" step="1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>Only trade when ML signal confidence exceeds this (50%–95%)</p>
        </div>
        <div className="input-group">
          <label>ADX Entry Threshold</label>
          <input type="number" value={adxThreshold} onChange={e => setAdxThreshold(Number(e.target.value))} min="5" max="30" step="1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>Minimum trend strength required to open a trade (5–30). Lower = more trades, higher = stronger trends only. Run Optimizer to find the best value per token.</p>
        </div>
        <div className="input-group">
          <label>Daily Loss Limit (%)</label>
          <input type="number" value={dailyLossLimit} onChange={e => setDailyLossLimit(Number(e.target.value))} min="1" max="50" step="1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>If the bot loses this % of day-start balance in one calendar day (UTC), it stops opening new positions until midnight. (1–50%, default 8%)</p>
        </div>
        <div className="input-group">
          <label>Max Concurrent Positions</label>
          <input type="number" value={maxPositions} onChange={e => setMaxPositions(Number(e.target.value))} min="1" max="10" step="1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>Maximum open positions at once across all coins. (1–10, default 3)</p>
        </div>
        <button className="btn btn-primary" onClick={saveSettings} disabled={settingsLoading || isBotRunning}>
          {settingsLoading ? 'Saving...' : 'Save Risk Settings'}
        </button>
      </Card>

      {/* ── Decision Engine ── */}
      <Card title="Decision Engine">
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16, lineHeight: 1.5 }}>
          Two systems can hold the bot back from a trade the ML signal wants to take, once they've accumulated
          enough evidence that a condition doesn't have a real edge after costs. See the Insights tab for live reasoning.
        </p>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 0', borderBottom: '1px solid var(--border)' }}>
          <div>
            <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>
              Market Intelligence Engine
              <InfoDot>
                Records market conditions continuously and only lets a trade through once a model has been validated
                against data it never trained on. Off by default. It can only skip a trade the ML signal wanted —
                never force one.
              </InfoDot>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
              {botStatus?.mie_available ? 'Available' : 'Unavailable on this deployment'}
              {botStatus?.mie_resolved_observations != null && ` · ${botStatus.mie_resolved_observations.toLocaleString()} observations recorded`}
            </div>
          </div>
          <ToggleSwitch checked={mieGateEnabled} onChange={saveMieToggle} disabled={isBotRunning || !botStatus?.mie_available} />
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 0' }}>
          <div>
            <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>
              Quality Gate
              <InfoDot>
                Scores every setup against its own trading history, sliced by regime, volatility, session and more.
                Arms itself automatically once there are ≥30 closed trades and ≥3 statistically qualified conditions —
                nothing to configure here.
              </InfoDot>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
              {botStatus?.quality_gate_enabled ? 'Armed — actively scoring entries' : 'Accumulating evidence, not yet armed'}
            </div>
          </div>
          {onOpenInsights && (
            <button onClick={onOpenInsights} style={{ background: 'none', border: 'none', color: 'var(--accent)', fontSize: 12, fontWeight: 700, cursor: 'pointer' }}>
              Details →
            </button>
          )}
        </div>
      </Card>

      {/* ── Compounding & Protection ── */}
      <Card title="Compounding & Protection">
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16, lineHeight: 1.5 }}>
          Controls how profits compound and how the bot protects capital on the path to growth.
        </p>
        <BotRunningWarning />
        <div className="input-group">
          <label>Trailing Stop (% of margin)</label>
          <input type="number" value={trailingStopPct} onChange={e => setTrailingStopPct(Number(e.target.value))} min="1" max="50" step="1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>Locks in gains — exits if price retraces this % of margin from its peak while in profit. At {leverage}x leverage → {(trailingStopPct / leverage).toFixed(2)}% price move. (1%–50%)</p>
        </div>
        <div className="input-group">
          <label>Max Drawdown Circuit Breaker (%)</label>
          <input type="number" value={maxDrawdownPct} onChange={e => setMaxDrawdownPct(Number(e.target.value))} min="5" max="50" step="1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>Stops opening new positions if account drawdown exceeds this. (5%–50%)</p>
        </div>
        <div className="input-group">
          <label>Profit Risk Multiplier</label>
          <input type="number" value={profitRiskMultiplier} onChange={e => setProfitRiskMultiplier(Number(e.target.value))} min="1.0" max="3.0" step="0.1" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>Applies extra risk to profits above starting balance — "house money" mode. 1.5 = 50% more aggressive on profits. (1.0–3.0)</p>
        </div>
        <div className="input-group">
          <label>Model Retrain Interval (cycles)</label>
          <input type="number" value={retrainEvery} onChange={e => setRetrainEvery(Number(e.target.value))} min="10" max="500" step="10" disabled={isBotRunning} style={dis(isBotRunning)} />
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>Retrain the ML model every N cycles to adapt to current market conditions. (10–500)</p>
        </div>
        <button className="btn btn-primary" onClick={saveSettings} disabled={settingsLoading || isBotRunning}>
          {settingsLoading ? 'Saving...' : 'Save Compounding Settings'}
        </button>
      </Card>

      {/* ── API Credentials ── */}
      <Card title="Blofin API Credentials">
        {hasCredentials && <div className="success-message" style={{ marginBottom: 16 }}>API credentials configured</div>}
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
      </Card>

      <Card title="Account">
        <button className="btn btn-secondary" onClick={logout}>Logout</button>
      </Card>

      <Card title="Install App">
        <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>
          On Android: open browser menu and tap "Add to Home Screen" to install as an app.
        </p>
      </Card>
    </>
  )
}
