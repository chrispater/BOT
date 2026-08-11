import React, { useState } from 'react'
import { Card, Badge, Meter, InfoDot } from '../components/Primitives'
import {
  fmtUsd, fmtSignedUsd, fmtPct, fmtSignedPct, coinOf, colorForScore, colorForSignal,
  regimeLabel, regimeColor,
} from '../utils/format'

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

  if (!compound || compound.insufficient_data) {
    const have = compound?.sample_size ?? 0
    return (
      <Card tone="green" title={<span style={{ color: 'var(--accent-green)' }}>Compound Tracker</span>}
        right={<span style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>{have} / 10 trades</span>}>
        <p style={{ textAlign: 'center', padding: '24px 0', color: 'var(--text-muted)', fontSize: 13 }}>
          Collecting data… need {10 - have} more trade{10 - have !== 1 ? 's' : ''} before projecting.
        </p>
      </Card>
    )
  }

  if (compound.warning) {
    const roi = compound.avg_trade_roi_pct ?? 0
    const pnl = compound.session_pnl ?? 0
    return (
      <Card tone="red" title={<span style={{ color: 'var(--accent-red)' }}>Compound Tracker</span>}
        right={<span style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>{compound.sample_size} trades sampled</span>}>
        <div style={{ textAlign: 'center', padding: '16px 0', borderBottom: '1px solid var(--border)', marginBottom: 14 }}>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>Avg ROI per Trade (last 20)</div>
          <div style={{ fontSize: 42, fontWeight: 900, color: 'var(--accent-red)', lineHeight: 1 }}>{roi >= 0 ? '+' : ''}{roi.toFixed(2)}%</div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 6 }}>{compound?.trades_per_day?.toFixed(1) || '—'} trades/day</div>
        </div>
        <div style={{ background: 'var(--accent-red-dim)', border: '1px solid rgba(233,69,96,0.2)', borderRadius: 10, padding: '12px 14px', fontSize: 12, color: 'var(--accent-red)', lineHeight: 1.6 }}>
          Session PnL: <strong>{fmtSignedUsd(pnl)}</strong> — bot is net-losing this session.
          No $1M projection shown until the session turns profitable. The autopilot is re-tuning to correct this.
        </div>
        <p style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center', margin: '10px 0 0' }}>
          Based on realized trade ROI · 100% reinvestment · No guarantee
        </p>
      </Card>
    )
  }

  const roi = compound.avg_trade_roi_pct ?? 0
  const roiColor = roi >= 5 ? 'var(--accent-green)' : roi >= 2 ? 'var(--accent-gold)' : 'var(--text-primary)'
  const milestones = [
    { label: '$10K', trades: compound.trades_to_10k, days: compound.days_to_10k, target: 10_000 },
    { label: '$100K', trades: compound.trades_to_100k, days: compound.days_to_100k, target: 100_000 },
    { label: '$1M', trades: compound.trades_to_1m, days: compound.days_to_1m, target: 1_000_000 },
  ]

  return (
    <Card tone="green" title={<span style={{ color: 'var(--accent-green)' }}>Compound Tracker</span>}
      right={<span style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>{compound.sample_size || 0} trades sampled</span>}>
      <div style={{ textAlign: 'center', padding: '16px 0 14px', borderBottom: '1px solid var(--border)', marginBottom: 16 }}>
        <div style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>
          Avg ROI per Trade (last 20)
        </div>
        <div style={{ fontSize: 42, fontWeight: 900, color: roiColor, lineHeight: 1 }}>{roi >= 0 ? '+' : ''}{roi.toFixed(2)}%</div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 6, display: 'flex', justifyContent: 'center', gap: 12 }}>
          <span>{compound.trades_per_day?.toFixed(1) || '—'} trades/day</span>
          {levUsed && <span style={{ color: levPct >= 90 ? 'var(--accent-green)' : levPct >= 70 ? 'var(--accent-gold)' : 'var(--text-secondary)' }}>
            last entry: {levUsed}x lev ({levPct}% of max)
          </span>}
        </div>
      </div>

      <div style={{ marginBottom: 8 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '80px 1fr 1fr', gap: '6px 0', fontSize: 11, color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, paddingBottom: 8, borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
          <span>Target</span><span style={{ textAlign: 'center' }}>Trades</span><span style={{ textAlign: 'right' }}>ETA</span>
        </div>
        {milestones.map(({ label, trades, days, target }) => {
          const reached = balance >= target
          return (
            <div key={label} style={{ display: 'grid', gridTemplateColumns: '80px 1fr 1fr', alignItems: 'center', padding: '10px 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
              <span style={{ fontWeight: 800, fontSize: 15, color: reached ? 'var(--accent-green)' : 'var(--text-primary)' }}>{reached ? '✓ ' : ''}{label}</span>
              <span style={{ textAlign: 'center', fontWeight: 700, fontSize: 15, color: reached ? 'var(--accent-green)' : trades != null ? 'var(--accent-gold)' : 'var(--text-muted)' }}>
                {reached ? 'Done' : trades != null ? trades.toLocaleString() : '—'}
              </span>
              <span style={{ textAlign: 'right', fontSize: 13, color: reached ? 'var(--accent-green)' : days != null ? 'var(--text-secondary)' : 'var(--text-muted)' }}>
                {reached ? '' : fmtDays(days)}
              </span>
            </div>
          )
        })}
      </div>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center', margin: '8px 0 0' }}>
        Based on realized trade ROI · 100% reinvestment · No guarantee
      </p>
    </Card>
  )
}

/* ─────────────────── Decision Engine status strip ─────────────────────── */
// Surfaces, at a glance, whether the two evidence-gating systems (Market
// Intelligence Engine + the quality gate) currently have anything to say, and
// whether either is actively holding the bot back from a trade it would
// otherwise take. Full reasoning per coin lives on the Insights tab — this is
// the "is something watching, and is it awake" summary for the home screen.
function DecisionEngineStrip({ botStatus, onOpenInsights }) {
  if (!botStatus?.running) return null

  const mieOn = !!botStatus.mie_gate_enabled
  const mieReady = !!botStatus.mie_any_validated
  const mieObs = botStatus.mie_resolved_observations || 0
  const qualityOn = !!botStatus.quality_gate_enabled
  const mieDecisions = botStatus.mie_decisions || {}
  const activeVetoes = Object.values(mieDecisions).filter(d => d.action === 'DO_NOTHING' && (d.blockers || []).length > 0).length

  const statusOf = (enabled, ready) => {
    if (!enabled) return { label: 'Off', color: 'var(--text-muted)' }
    if (!ready) return { label: 'Learning', color: 'var(--accent-gold)' }
    return { label: 'Armed', color: 'var(--accent-green)' }
  }
  const mieStat = statusOf(mieOn, mieReady)
  const qStat = statusOf(qualityOn, qualityOn) // quality gate self-reports armed only once ready

  return (
    <Card
      title="Decision Engine"
      right={<button onClick={onOpenInsights} style={{
        background: 'none', border: 'none', color: 'var(--accent)', fontSize: 12, fontWeight: 700, cursor: 'pointer',
      }}>See why →</button>}
    >
      <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 12, lineHeight: 1.5 }}>
        Two independent systems can hold a trade back until there's measured evidence it has an edge.
        "No Trade" is a normal, healthy result — not a malfunction.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        <div style={{ padding: '10px 12px', background: 'var(--bg-card-alt)', borderRadius: 10, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4 }}>Market Intelligence</div>
          <Badge color={mieStat.color}>{mieStat.label}</Badge>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>{mieObs.toLocaleString()} observations recorded</div>
        </div>
        <div style={{ padding: '10px 12px', background: 'var(--bg-card-alt)', borderRadius: 10, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4 }}>Quality Gate</div>
          <Badge color={qStat.color}>{qStat.label}</Badge>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>{qualityOn ? 'Scoring every setup' : 'Accumulating trade history'}</div>
        </div>
      </div>
      {activeVetoes > 0 && (
        <div style={{ marginTop: 10, padding: '8px 12px', background: 'rgba(74,158,255,0.08)', border: '1px solid rgba(74,158,255,0.2)', borderRadius: 10, fontSize: 12, color: 'var(--accent-blue)' }}>
          Currently holding back {activeVetoes} coin{activeVetoes !== 1 ? 's' : ''} with no measured edge right now.
        </div>
      )}
    </Card>
  )
}

/* ─────────────────────── Per-coin "why" line ───────────────────────────── */
// Combines the legacy ML/quality signal with the MIE decision (when present)
// into one sentence a non-technical reader can parse without opening Insights.
function describeCoinState(botStatus, coin, signal) {
  const mie = (botStatus.mie_decisions || {})[coin]
  if (!signal) return { text: 'Waiting for first read…', color: 'var(--text-muted)' }

  if (signal.signal !== 0) {
    const dir = signal.signal === 1 ? 'LONG' : 'SHORT'
    return {
      text: `Would trade ${dir} — ${fmtPct(signal.confidence * 100, 0)} confidence${signal.quality != null ? `, quality ${signal.quality}/100` : ''}`,
      color: colorForSignal(signal.signal),
    }
  }
  if (mie && mie.action === 'DO_NOTHING' && (mie.blockers || []).length > 0) {
    return { text: `No trade — ${mie.blockers[0]}`, color: 'var(--accent-blue)' }
  }
  if (signal.quality_reason && signal.quality != null && signal.quality < 55) {
    return { text: `No trade — ${signal.quality_reason}`, color: 'var(--accent-gold)' }
  }
  return { text: 'No trade — no qualifying signal this candle', color: 'var(--text-secondary)' }
}

/* ─────────────────────────── Dashboard ─────────────────────────── */

export default function DashboardPage({ botStatus, api, fetchBotStatus, setError, setSuccess, onOpenInsights }) {
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
      setSuccess(`Opened ${manualSide.toUpperCase()} ${coinOf(sym)} @ ${fmtUsd(res.data.price)}`)
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
      setSuccess(`Closed ${coinOf(symbol)} @ ${fmtUsd(res.data.price)} — PnL: ${fmtSignedUsd(pnl)}`)
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
      <Card
        title="Bot Status"
        right={botStatus?.running ? (
          <span className={`status-badge ${botStatus.simulation_mode ? 'simulation' : 'running'}`}>
            <span className={`pulse ${botStatus.simulation_mode ? 'yellow' : 'green'}`}></span>
            {botStatus.simulation_mode ? 'Simulation' : 'Live'}
          </span>
        ) : (
          <span className="status-badge stopped"><span className="pulse red"></span>Stopped</span>
        )}
      >
        {botStatus?.running ? (
          <button className="btn btn-danger" onClick={stopBot} disabled={loading}>{loading ? 'Stopping...' : 'Stop Bot'}</button>
        ) : (
          <button className="btn btn-success" onClick={startBot} disabled={loading}>{loading ? 'Starting...' : 'Start Bot'}</button>
        )}
      </Card>

      <DecisionEngineStrip botStatus={botStatus} onOpenInsights={onOpenInsights} />

      {/* Performance */}
      <Card title="Performance">
        <div className="stat-grid">
          <div className="stat-item">
            <div className="stat-value">{fmtUsd(botStatus?.balance, { maximumFractionDigits: 0 })}</div>
            <div className="stat-label">Balance</div>
          </div>
          <div className="stat-item">
            <div className={`stat-value ${(botStatus?.total_pnl || 0) >= 0 ? 'positive' : 'negative'}`}>{fmtSignedUsd(botStatus?.total_pnl)}</div>
            <div className="stat-label">Total PnL</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">{botStatus?.total_trades || 0}</div>
            <div className="stat-label">Total Trades</div>
          </div>
          <div className="stat-item">
            <div className={`stat-value ${(botStatus?.win_rate || 0) >= 50 ? 'positive' : 'negative'}`}>{fmtPct(botStatus?.win_rate)}</div>
            <div className="stat-label">Win Rate</div>
          </div>
        </div>

        {botStatus?.running && (
          <div style={{ marginTop: 8, padding: '10px 12px', background: 'var(--bg-card-alt)', borderRadius: 10, border: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Today</span>
            <span style={{ fontSize: 14, fontWeight: 700, color: (botStatus.day_pnl || 0) >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
              {fmtSignedPct(botStatus.day_pnl_pct)}
            </span>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Limit: −{fmtPct((botStatus.daily_loss_limit || 0.08) * 100, 0)} · Max pos: {botStatus.max_positions || 3}
            </span>
          </div>
        )}

        {botStatus?.kelly_fraction != null && (
          <div style={{ marginTop: 8, padding: '10px 12px', background: 'var(--bg-card-alt)', borderRadius: 10, border: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              Kelly Fraction
              <InfoDot>How much of the balance goes into the next trade, sized down from the theoretical optimum for safety. Rises as the bot's realized win/loss record improves.</InfoDot>
            </span>
            <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--accent-gold)' }}>{fmtPct(botStatus.kelly_fraction * 100)}</span>
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
          const color = isWin ? 'var(--accent-green)' : 'var(--accent-red)'
          return (
            <div style={{ marginTop: 8, padding: '10px 12px', background: 'var(--bg-card-alt)', borderRadius: 10, border: `1px solid ${isWin ? 'rgba(0,212,170,0.2)' : 'rgba(255,68,68,0.2)'}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{isWin ? 'Win Streak' : 'Loss Streak'}</span>
              <span style={{ fontSize: 14, fontWeight: 700, color }}>{streak}x {isWin ? 'WIN' : 'LOSS'}</span>
            </div>
          )
        })()}
      </Card>

      {botStatus?.compound && (
        <CompoundTracker compound={botStatus.compound} balance={botStatus.balance} dynLeverage={botStatus.last_dynamic_leverage} maxLeverage={botStatus.leverage} />
      )}

      {/* Live Signals */}
      {botStatus?.running && selectedCoins.length > 0 && (
        <Card title="Live Signals">
          {selectedCoins.map(coinSymbol => {
            const coin = coinOf(coinSymbol)
            const signal = coinSignals[coin]
            const conf = signal?.confidence || 0
            const confColor = colorForScore(conf * 100, { good: 70, warn: 65 })
            const why = describeCoinState(botStatus, coin, signal)

            return (
              <div key={coin} style={{ padding: 12, marginBottom: 8, background: 'var(--bg-card-alt)', borderRadius: 10, border: '1px solid var(--border)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 700, color: 'var(--accent)', fontSize: 15 }}>{coin}</span>
                  {signal ? (
                    <Badge color={colorForSignal(signal.signal)}>{signal.signal === 1 ? 'LONG' : signal.signal === -1 ? 'SHORT' : 'HOLD'}</Badge>
                  ) : (
                    <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>Waiting...</span>
                  )}
                </div>
                {signal && (
                  <>
                    <div style={{ display: 'flex', gap: 14, fontSize: 12, color: 'var(--text-secondary)', marginTop: 6, flexWrap: 'wrap' }}>
                      <span>Conf: <strong style={{ color: confColor }}>{fmtPct(conf * 100)}</strong></span>
                      <span>Price: <strong style={{ color: 'var(--text-primary)' }}>{fmtUsd(signal.price)}</strong></span>
                      {signal.regime && <span>Regime: <strong style={{ color: regimeColor(signal.regime) }}>{regimeLabel(signal.regime)}</strong></span>}
                    </div>
                    <div className="confidence-bar-bg"><div className="confidence-bar-fill" style={{ width: `${(conf * 100).toFixed(0)}%`, background: confColor }} /></div>
                    <div style={{ marginTop: 8, fontSize: 12, color: why.color, lineHeight: 1.4 }}>{why.text}</div>
                  </>
                )}
              </div>
            )
          })}
        </Card>
      )}

      {/* Open Positions */}
      {openPositions.length > 0 && (
        <Card title={`Open Positions (${openPositions.length})`}>
          {openPositions.map((pos, idx) => {
            const coin = coinOf(pos.symbol)
            const isLong = pos.side === 'long'
            const sideColor = isLong ? 'var(--accent-green)' : 'var(--accent-red)'
            return (
              <div key={idx} style={{ padding: 12, marginBottom: idx < openPositions.length - 1 ? 8 : 0, background: 'var(--bg-card-alt)', borderRadius: 10, border: `1px solid ${isLong ? 'rgba(0,212,170,0.25)' : 'rgba(255,68,68,0.25)'}` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontWeight: 700, color: 'var(--accent)', fontSize: 15 }}>{coin}</span>
                  <Badge color={sideColor}>{pos.side?.toUpperCase()}</Badge>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
                  <span>Entry: <strong style={{ color: 'var(--text-primary)' }}>{fmtUsd(pos.entry_price)}</strong></span>
                  <span>Size: <strong style={{ color: 'var(--text-primary)' }}>{pos.size?.toFixed(4)}</strong></span>
                  <span>Margin: <strong style={{ color: 'var(--text-primary)' }}>{fmtUsd(pos.margin)}</strong></span>
                  <span>Leverage: <strong style={{ color: 'var(--accent-gold)' }}>{pos.leverage || botStatus?.leverage}x</strong></span>
                  {isLong && pos.high_water_mark && <span>Peak: <strong style={{ color: 'var(--accent-green)' }}>{fmtUsd(pos.high_water_mark)}</strong></span>}
                  {!isLong && pos.low_water_mark && <span>Low: <strong style={{ color: 'var(--accent-red)' }}>{fmtUsd(pos.low_water_mark)}</strong></span>}
                </div>
                <div style={{ marginTop: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  {pos.conf_tier ? (
                    <Badge color={pos.conf_tier === 'NO-BRAINER' ? 'var(--accent-green)' : pos.conf_tier === 'STRONG' ? 'var(--accent-gold)' : 'var(--text-secondary)'}>
                      {pos.conf_tier}
                    </Badge>
                  ) : <span />}
                  <button
                    onClick={() => manualExit(pos.symbol)} disabled={manualLoading}
                    style={{ padding: '4px 14px', borderRadius: 8, border: '1px solid rgba(233,69,96,0.5)', background: 'var(--accent-dim)', color: 'var(--accent)', fontSize: 11, fontWeight: 700, cursor: 'pointer', letterSpacing: 1 }}
                  >EXIT</button>
                </div>
              </div>
            )
          })}
        </Card>
      )}

      {/* Manual Trade */}
      <Card title="Manual Trade">
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', opacity: botStatus?.running ? 1 : 0.5 }}>
          <select
            value={manualSymbol || selectedCoinsForEntry[0] || ''}
            onChange={e => setManualSymbol(e.target.value)}
            disabled={!botStatus?.running}
            style={{ flex: 1, minWidth: 140, padding: '8px 10px', borderRadius: 8, background: 'var(--bg-input)', border: '1px solid #2a3040', color: 'var(--text-primary)', fontSize: 13 }}
          >
            {selectedCoinsForEntry.length === 0 && <option value="">No coins selected</option>}
            {selectedCoinsForEntry.map(c => <option key={c} value={c}>{coinOf(c)}</option>)}
          </select>
          <div style={{ display: 'flex', borderRadius: 8, overflow: 'hidden', border: '1px solid #2a3040' }}>
            {['long', 'short'].map(s => (
              <button
                key={s} onClick={() => setManualSide(s)} disabled={!botStatus?.running}
                style={{
                  padding: '8px 18px', border: 'none', fontSize: 12, fontWeight: 700,
                  cursor: botStatus?.running ? 'pointer' : 'not-allowed', letterSpacing: 1, textTransform: 'uppercase',
                  background: manualSide === s ? (s === 'long' ? 'rgba(0,212,170,0.25)' : 'rgba(233,69,96,0.25)') : 'var(--bg-input)',
                  color: manualSide === s ? (s === 'long' ? 'var(--accent-green)' : 'var(--accent)') : 'var(--text-muted)',
                }}
              >{s}</button>
            ))}
          </div>
          <button
            onClick={manualEnter} disabled={manualLoading || !botStatus?.running || selectedCoinsForEntry.length === 0}
            style={{
              padding: '8px 22px', borderRadius: 8, border: 'none', fontSize: 13, fontWeight: 700,
              cursor: botStatus?.running ? 'pointer' : 'not-allowed', letterSpacing: 1,
              background: manualSide === 'long' ? 'rgba(0,212,170,0.2)' : 'rgba(233,69,96,0.2)',
              color: manualSide === 'long' ? 'var(--accent-green)' : 'var(--accent)',
            }}
          >{manualLoading ? '...' : 'ENTER'}</button>
        </div>
        <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
          {botStatus?.running
            ? 'Bot resumes auto-trading after the position closes. Manual entries skip the Decision Engine gates above.'
            : 'Start the bot to enable manual entry. The bot then manages the position and resumes looking for the next trade.'}
        </p>
      </Card>
    </>
  )
}
