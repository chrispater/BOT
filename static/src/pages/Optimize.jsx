import React, { useEffect, useRef, useState } from 'react'
import { Card, EmptyState, InfoDot } from '../components/Primitives'
import MieValidationPanel from '../components/MieValidation'

export default function OptimizePage({ api, setError, setSuccess }) {
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
      <Card title="Timeframe Comparison"
        right={<InfoDot>
          This used to grid-search ~1,350 combinations of leverage/stop-loss/take-profit/confidence for the
          best backtested ROI — exactly the curve-fitting the Market Intelligence Engine exists to move away
          from. It now compares only TIMEFRAME, walk-forward validated, with every risk setting held at
          whatever you've configured in Settings. If you want risk parameters tuned by evidence rather than
          a search, that's what the Market Intelligence Validation panel below is for.
        </InfoDot>}
      >
        <p style={{ color: 'var(--text-secondary)', fontSize: 14, marginBottom: 12, lineHeight: 1.5 }}>
          Backtests your current risk settings across up to 7 timeframes, walk-forward validated so a result
          has to hold up across multiple out-of-sample periods, not just look good once.
        </p>
        <p style={{ color: 'var(--accent)', fontSize: 12, marginBottom: 16 }}>Takes a few minutes to complete.</p>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-primary" onClick={runOptimization} disabled={optimizeLoading}>
            {optimizeLoading ? (optimizeStatus || 'Comparing...') : 'Compare Timeframes'}
          </button>
          {optimizeLoading && (
            <button className="btn btn-danger" onClick={resetOptimization} style={{ maxWidth: 90, padding: '14px 12px', fontSize: 13 }}>Reset</button>
          )}
        </div>
        {optimizeError && <div className="error-message" style={{ marginTop: 16 }}>{optimizeError}</div>}
      </Card>

      <MieValidationPanel api={api} />

      {optimizeResults && (
        <Card title="Results">
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>
            {optimizeResults.valid_configs} timeframe{optimizeResults.valid_configs !== 1 ? 's' : ''} cleared the minimum trade/win-rate bar · {optimizeResults.days_tested} days
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 16 }}>
            Coins: {optimizeResults.selected_coins?.map(c => c.split('/')[0]).join(', ')}
          </div>

          {optimizeResults.top_configs?.length > 0 && (() => {
            const c0 = optimizeResults.top_configs[0]
            return (
              <div style={{ padding: '8px 10px', background: 'var(--bg-card-alt)', borderRadius: 8, border: '1px solid var(--border)', marginBottom: 14, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6 }}>
                Held fixed at your current settings: {c0.leverage}x leverage · {(c0.risk_per_trade * 100).toFixed(1)}% risk ·{' '}
                {(c0.stop_loss_pct * 100).toFixed(2)}% SL · {(c0.take_profit_pct * 100).toFixed(2)}% TP ·{' '}
                {(c0.min_confidence * 100).toFixed(0)}% confidence · {formatCooldown(c0.trade_cooldown)} cooldown ·{' '}
                {c0.adx_threshold ?? '—'} ADX. Change these in Settings, not by searching for a better-looking backtest.
              </div>
            )
          })()}

          {optimizeResults.top_configs?.length === 0 ? (
            <p style={{ color: 'var(--accent)' }}>No timeframe cleared the minimum trade count / win rate bar for these coins.</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {optimizeResults.top_configs?.map((config, i) => {
                const isSelected = selectedConfig === config
                const monthlyRet = (config.total_return / Math.max(optimizeResults.days_tested, 1)) * 30
                const yearValue = 1000 * Math.pow(1 + monthlyRet / 100, 12)
                const annualPct = (yearValue / 1000 - 1) * 100

                return (
                  <div key={i} style={{
                    padding: 14, background: isSelected ? 'rgba(0,212,170,0.08)' : 'var(--bg-card-alt)', borderRadius: 12,
                    border: isSelected ? '1px solid rgba(0,212,170,0.4)' : '1px solid var(--border)',
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                      <span style={{ fontSize: 15, fontWeight: 700, color: 'var(--accent)' }}>{config.timeframe}</span>
                      <span style={{ fontSize: 15, fontWeight: 700, color: config.total_return >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                        {config.total_return >= 0 ? '+' : ''}{config.total_return.toFixed(1)}% ROI
                      </span>
                    </div>

                    {monthlyRet > 0 && (
                      <div style={{ padding: '8px 10px', background: 'rgba(245,166,35,0.07)', borderRadius: 8, border: '1px solid rgba(245,166,35,0.18)', marginBottom: 10 }}>
                        <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>12-mo compound: </span>
                        <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--accent-gold)' }}>${yearValue.toFixed(0)}&nbsp;</span>
                        <span style={{ fontSize: 11, color: 'var(--accent-green)' }}>(+{annualPct.toFixed(0)}% on $1k)</span>
                      </div>
                    )}

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 12 }}>
                      <span style={{ color: 'var(--accent-green)' }}>
                        {config.win_rate.toFixed(1)}% win · {config.total_trades} trades · ${config.total_pnl.toFixed(0)} PnL
                      </span>
                      <button className="btn btn-secondary" style={{ padding: '6px 14px', fontSize: 12, width: 'auto' }} onClick={() => applyConfig(config)} disabled={applyLoading}>
                        {isSelected ? 'Applied' : 'Use this timeframe'}
                      </button>
                    </div>

                    {config.wf_folds > 1 && (
                      <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-secondary)', display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
                        <span title="Fraction of out-of-sample walk-forward windows that were profitable">
                          Walk-forward: <strong style={{ color: config.wf_consistency >= 0.75 ? 'var(--accent-green)' : config.wf_consistency >= 0.5 ? 'var(--accent-gold)' : 'var(--accent-red)' }}>
                            {Math.round((config.wf_consistency || 0) * 100)}% folds green
                          </strong> ({config.wf_folds})
                        </span>
                        {Array.isArray(config.fold_returns) && config.fold_returns.length > 0 && (
                          <span style={{ display: 'flex', gap: 3 }}>
                            {config.fold_returns.map((r, idx) => (
                              <span key={idx} title={`Fold ${idx + 1}: ${r >= 0 ? '+' : ''}${r}%`}
                                style={{ width: 7, height: 14, borderRadius: 2, background: r >= 0 ? 'var(--accent-green)' : 'var(--accent-red)', opacity: 0.85 }} />
                            ))}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </Card>
      )}

      <Card title="Previous Runs">
        {history.length === 0 ? (
          <EmptyState>No previous runs yet.</EmptyState>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: 'var(--text-secondary)', fontWeight: 600 }}>Date</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: 'var(--text-secondary)', fontWeight: 600 }}>Coins</th>
                  <th style={{ textAlign: 'right', padding: '6px 8px', color: 'var(--text-secondary)', fontWeight: 600 }}>Best ROI</th>
                  <th style={{ textAlign: 'right', padding: '6px 8px', color: 'var(--text-secondary)', fontWeight: 600 }}>Win Rate</th>
                  <th style={{ textAlign: 'right', padding: '6px 8px', color: 'var(--text-secondary)', fontWeight: 600 }}>Configs</th>
                  <th style={{ padding: '6px 8px' }}></th>
                </tr>
              </thead>
              <tbody>
                {history.map(run => {
                  const isLoaded = loadedRunId === run.id
                  const isApplying = applyingRunId === run.id
                  return (
                    <tr key={run.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', background: isLoaded ? 'rgba(0,212,170,0.06)' : 'transparent' }}>
                      <td style={{ padding: '8px 8px', color: 'var(--text-primary)' }}>{run.completed_at ? new Date(run.completed_at).toLocaleDateString() : '—'}</td>
                      <td style={{ padding: '8px 8px', color: 'var(--text-primary)' }}>{(run.coins || []).map(c => c.split('/')[0]).join(', ') || '—'}</td>
                      <td style={{ padding: '8px 8px', textAlign: 'right', color: run.best_roi > 0 ? 'var(--accent-green)' : 'var(--accent-red)', fontWeight: 600 }}>
                        {run.best_roi != null ? `${run.best_roi >= 0 ? '+' : ''}${Number(run.best_roi).toFixed(1)}%` : '—'}
                      </td>
                      <td style={{ padding: '8px 8px', textAlign: 'right', color: 'var(--text-primary)' }}>{run.best_win_rate != null ? `${Number(run.best_win_rate).toFixed(1)}%` : '—'}</td>
                      <td style={{ padding: '8px 8px', textAlign: 'right', color: 'var(--text-secondary)' }}>{run.valid_configs}/{run.total_tested}</td>
                      <td style={{ padding: '6px 4px', textAlign: 'right' }}>
                        <div style={{ display: 'flex', gap: 4, justifyContent: 'flex-end' }}>
                          <button className="btn btn-secondary" style={{ padding: '4px 10px', fontSize: 11, width: 'auto', opacity: isLoaded ? 0.5 : 1 }} onClick={() => loadHistoricalRun(run.id)} disabled={isLoaded}>
                            {isLoaded ? '✓' : 'Load'}
                          </button>
                          <button
                            className="btn btn-success" style={{ padding: '4px 10px', fontSize: 11, width: 'auto', background: 'var(--accent-green)', color: '#0a0d12', fontWeight: 700, opacity: isApplying ? 0.5 : 1 }}
                            onClick={() => applyRunToBot(run.id)} disabled={isApplying} title="Hot-apply this run's best config to the running bot"
                          >{isApplying ? '...' : '⚡ Apply'}</button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card title="How It Works">
        <ul className="strategy-list">
          <li><strong>Timeframe only:</strong> Leverage, risk per trade, stop-loss, take-profit, trailing stop, confidence, cooldown and ADX threshold all stay at your current Settings — nothing is grid-searched for the best-looking backtest</li>
          <li><strong>Walk-forward validated:</strong> Each timeframe is trained on an expanding window and tested on the next out-of-sample segment, rolled forward — a result only counts if it holds up across multiple forward periods</li>
          <li><strong>Composite Score:</strong> Weighted blend of return, win rate, Calmar ratio and walk-forward consistency, with a drawdown penalty — used to rank timeframes, not to hunt for a config</li>
          <li><strong>Min Trades:</strong> A timeframe with too few trades to be statistically meaningful is excluded rather than ranked</li>
          <li><strong>To change risk parameters:</strong> Use Settings directly, or wait for the Market Intelligence Engine above to validate a horizon — it earns the right to size and gate trades through evidence, not a search</li>
        </ul>
      </Card>
    </>
  )
}
