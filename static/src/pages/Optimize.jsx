import React, { useEffect, useRef, useState } from 'react'
import { Card, EmptyState } from '../components/Primitives'

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
      <Card title="Parameter Optimizer">
        <p style={{ color: 'var(--text-secondary)', fontSize: 14, marginBottom: 12, lineHeight: 1.5 }}>
          Automatically discovers the best trading parameters for your coins by testing ~1,350 configurations across all timeframes. Returns top performers ranked by ROI, win rate, drawdown, and trade count.
        </p>
        <p style={{ color: 'var(--accent)', fontSize: 12, marginBottom: 16 }}>Warning: This takes 5–15 minutes to complete.</p>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-primary" onClick={runOptimization} disabled={optimizeLoading}>
            {optimizeLoading ? (optimizeStatus || 'Optimizing...') : 'Find Best Parameters'}
          </button>
          {optimizeLoading && (
            <button className="btn btn-danger" onClick={resetOptimization} style={{ maxWidth: 90, padding: '14px 12px', fontSize: 13 }}>Reset</button>
          )}
        </div>
        {optimizeError && <div className="error-message" style={{ marginTop: 16 }}>{optimizeError}</div>}
      </Card>

      {optimizeResults && (
        <Card title="Top Configurations">
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>
            Tested {optimizeResults.total_tested} configs · {optimizeResults.valid_configs} profitable · {optimizeResults.days_tested} days
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 16 }}>
            Coins: {optimizeResults.selected_coins?.map(c => c.split('/')[0]).join(', ')}
          </div>

          {optimizeResults.top_configs?.length === 0 ? (
            <p style={{ color: 'var(--accent)' }}>No profitable configurations found. Try different coins or timeframes.</p>
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
                      <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--accent)' }}>#{i + 1} — {config.timeframe}</span>
                      <span style={{ fontSize: 15, fontWeight: 700, color: config.total_return >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                        {config.total_return >= 0 ? '+' : ''}{config.total_return.toFixed(1)}% ROI
                      </span>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 12, color: 'var(--text-secondary)', marginBottom: 10 }}>
                      <span>Leverage: <strong style={{ color: 'var(--text-primary)' }}>{config.leverage}x</strong></span>
                      <span>Risk: <strong style={{ color: 'var(--text-primary)' }}>{(config.risk_per_trade * 100).toFixed(0)}%</strong></span>
                      <span>SL: <strong style={{ color: 'var(--accent-red)' }}>{(config.stop_loss_pct * 100).toFixed(2)}%</strong></span>
                      <span>TP: <strong style={{ color: 'var(--accent-green)' }}>{(config.take_profit_pct * 100).toFixed(2)}%</strong></span>
                      <span>Trail: <strong style={{ color: 'var(--text-primary)' }}>{config.trailing_stop_pct != null ? `${(config.trailing_stop_pct * 100).toFixed(1)}%` : '—'}</strong></span>
                      <span>Conf: <strong style={{ color: 'var(--text-primary)' }}>{(config.min_confidence * 100).toFixed(0)}%</strong></span>
                      <span>ADX: <strong style={{ color: 'var(--text-primary)' }}>{config.adx_threshold ?? '—'}</strong></span>
                      <span>Cooldown: <strong style={{ color: 'var(--text-primary)' }}>{formatCooldown(config.trade_cooldown)}</strong></span>
                      <span>Score: <strong style={{ color: 'var(--accent-blue)' }}>{config.score?.toFixed(3)}</strong></span>
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
                        {isSelected ? 'Applied' : 'Apply'}
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
          <li><strong>Smart Search:</strong> Tests 100 random parameter combos per timeframe across all major timeframes</li>
          <li><strong>Trailing Stop:</strong> Also optimises trailing stop distance (0.5%–2%) for locked profits</li>
          <li><strong>Composite Score:</strong> 50% ROI + 30% Win Rate + 20% Trade Count, with drawdown penalty</li>
          <li><strong>Drawdown Guard:</strong> Penalises configs where max drawdown exceeds 15%</li>
          <li><strong>Min Trades:</strong> Rejects configs with fewer than 15 trades (avoids overfitting)</li>
          <li><strong>Compound Projection:</strong> Shows annualised compounding at each config's monthly return</li>
        </ul>
      </Card>
    </>
  )
}
