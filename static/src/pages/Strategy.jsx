import React, { useState } from 'react'
import { Card, EmptyState, InfoDot } from '../components/Primitives'
import { fmtUsd, fmtSignedUsd, coinOf, regimeColor, regimeLabel } from '../utils/format'

/* ─────────────────────────── Growth Projection ─────────────────────────── */

function GrowthProjection({ periodDays, totalReturn, monthlyRoi }) {
  const mr = monthlyRoi != null ? monthlyRoi : (totalReturn / Math.max(periodDays, 1)) * 30
  const projections = [
    { label: '3mo', months: 3 }, { label: '6mo', months: 6 }, { label: '1yr', months: 12 },
    { label: '2yr', months: 24 }, { label: '3yr', months: 36 },
  ]
  const milestones = [{ label: '$10K', target: 10000 }, { label: '$100K', target: 100000 }, { label: '$1M', target: 1000000 }]
  const timeToMilestone = (target) => {
    if (mr <= 0) return null
    const months = Math.log(target / 1000) / Math.log(1 + mr / 100)
    const m = Math.ceil(months)
    return m > 0 && m < 600 ? m : null
  }
  const fmtMonths = (m) => m >= 24 ? `${(m / 12).toFixed(1)} yrs` : `${m} mo`

  return (
    <Card title="Compound Growth Projection">
      <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 14 }}>
        At <strong style={{ color: 'var(--accent-gold)' }}>{mr.toFixed(2)}% / month</strong> compounded · Starting capital: $1,000
      </p>
      <div className="growth-table" style={{ marginBottom: 16 }}>
        {projections.map(({ label, months }) => {
          const value = 1000 * Math.pow(1 + mr / 100, months)
          return (
            <div key={months} className="growth-row">
              <span style={{ color: 'var(--text-secondary)', fontSize: 13 }}>{label}</span>
              <span style={{ color: 'var(--accent-gold)', fontWeight: 700, fontSize: 15 }}>
                {value.toLocaleString(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })}
              </span>
            </div>
          )
        })}
      </div>
      <div style={{ paddingTop: 14, borderTop: '1px solid var(--border)' }}>
        <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 1 }}>
          Time to Milestone (from $1K)
        </div>
        {milestones.map(({ label, target }) => {
          const months = timeToMilestone(target)
          return (
            <div key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
              <span style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: 15 }}>{label}</span>
              {months
                ? <span style={{ color: 'var(--accent-green)', fontWeight: 700, fontSize: 14 }}>{fmtMonths(months)}</span>
                : <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>N/A at this rate</span>}
            </div>
          )
        })}
      </div>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 12, textAlign: 'center' }}>
        Compound interest math, not a guarantee. Past backtest ≠ future results.
      </p>
    </Card>
  )
}

/* ─────────────────────── Market Direction Scanner ───────────────────────── */

function MarketDirectionScanner({ api }) {
  const [scan, setScan] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const runScan = async () => {
    setLoading(true); setError('')
    try {
      const res = await api.get('/market/direction')
      setScan(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to scan market direction')
    }
    setLoading(false)
  }

  const dirColor = (label) => {
    if (!label) return 'var(--text-secondary)'
    if (label.includes('BULL')) return 'var(--accent-green)'
    if (label.includes('BEAR')) return 'var(--accent)'
    return 'var(--accent-gold)'
  }

  return (
    <Card title="Market Direction Scanner">
      <p style={{ color: 'var(--text-secondary)', fontSize: 14, marginBottom: 16, lineHeight: 1.5 }}>
        Multi-timeframe (15m · 1h · 4h) trend, momentum and strength read for each selected token.
        Use it to see where each token is heading before you optimize, backtest, and go live.
      </p>
      <button className="btn btn-primary" onClick={runScan} disabled={loading}>
        {loading ? 'Scanning...' : 'Scan Market Direction'}
      </button>

      {error && <div className="error-message" style={{ marginTop: 16 }}>{error}</div>}

      {scan && (
        <div style={{ marginTop: 16 }}>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'center', padding: '10px 14px', background: 'var(--bg-card-alt)', borderRadius: 10, border: '1px solid var(--border)', marginBottom: 16 }}>
            <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              BTC regime: <strong style={{ color: regimeColor(scan.market_regime), textTransform: 'uppercase' }}>{scan.market_regime}</strong>
            </span>
            <span style={{ fontSize: 13, color: 'var(--accent-green)' }}>▲ {scan.summary?.bullish} bullish</span>
            <span style={{ fontSize: 13, color: 'var(--accent)' }}>▼ {scan.summary?.bearish} bearish</span>
            <span style={{ fontSize: 13, color: 'var(--accent-gold)' }}>● {scan.summary?.neutral} neutral</span>
            <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 'auto' }}>ADX gate: {scan.adx_threshold}</span>
          </div>

          {scan.tokens?.map((t) => (
            <div key={t.symbol} style={{ padding: '12px 14px', background: 'var(--bg-card-alt)', borderRadius: 10, border: '1px solid var(--border)', marginBottom: 10 }}>
              {t.error ? (
                <div style={{ fontSize: 14 }}>
                  <strong style={{ color: 'var(--text-primary)' }}>{coinOf(t.symbol)}</strong>
                  <span style={{ color: 'var(--text-muted)', marginLeft: 10 }}>no data</span>
                </div>
              ) : (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 10, marginBottom: 10 }}>
                    <strong style={{ color: 'var(--text-primary)', fontSize: 16 }}>{coinOf(t.symbol)}</strong>
                    <span style={{ fontSize: 12, fontWeight: 700, color: dirColor(t.label), border: `1px solid ${dirColor(t.label)}`, borderRadius: 6, padding: '2px 8px' }}>{t.label}</span>
                    {t.aligned && <span style={{ fontSize: 11, color: 'var(--accent-green)' }}>✓ all TFs aligned</span>}
                    <span style={{ marginLeft: 'auto', fontSize: 12, color: t.tradeable ? 'var(--accent-green)' : 'var(--text-secondary)' }}>{t.recommended_bias}</span>
                  </div>
                  <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', fontSize: 12, color: 'var(--text-secondary)', marginBottom: 10 }}>
                    <span>Conviction: <strong style={{ color: dirColor(t.label) }}>{(t.conviction * 100).toFixed(0)}%</strong></span>
                    <span>Avg ADX: <strong style={{ color: 'var(--text-primary)' }}>{t.avg_adx}</strong> ({t.trend_strength})</span>
                  </div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {t.timeframes?.map((tf) => (
                      <div key={tf.timeframe} style={{ flex: '1 1 90px', minWidth: 90, padding: '6px 8px', background: '#161b22', borderRadius: 8, borderLeft: `3px solid ${dirColor(tf.direction.toUpperCase())}` }}>
                        <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 2 }}>{tf.timeframe}</div>
                        <div style={{ fontSize: 12, fontWeight: 600, color: dirColor(tf.direction.toUpperCase()), textTransform: 'capitalize' }}>{tf.direction}</div>
                        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>RSI {tf.rsi} · ADX {tf.adx}</div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          ))}
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4, textAlign: 'center' }}>
            Scanned {new Date(scan.scanned_at).toLocaleTimeString()} · directional read only, not a trade signal.
          </p>
        </div>
      )}
    </Card>
  )
}

/* ─────────────────────────── Strategy / Backtest ─────────────────────────── */

export default function StrategyPage({ strategies, api }) {
  const [backtestResults, setBacktestResults] = useState(null)
  const [backtestLoading, setBacktestLoading] = useState(false)
  const [backtestError, setBacktestError] = useState('')
  const [expandedCoin, setExpandedCoin] = useState(null)
  const [showAllTrades, setShowAllTrades] = useState(false)

  const runBacktest = async () => {
    setBacktestLoading(true); setBacktestError(''); setExpandedCoin(null); setShowAllTrades(false)
    try {
      const res = await api.get('/backtest')
      setBacktestResults(res.data)
    } catch (e) {
      setBacktestError(e.response?.data?.detail || 'Failed to run backtest')
    }
    setBacktestLoading(false)
  }

  if (!strategies) return <div className="loading"><div className="spinner"></div></div>

  return (
    <>
      <MarketDirectionScanner api={api} />

      <Card title="Backtest — Legacy ML Pipeline"
        right={<InfoDot>
          This simulates your CURRENT settings historically — it isn't a search, and nothing here gets
          curve-fit to the result. It's the ML+setup pipeline's own single-config sanity check, separate
          from the Market Intelligence Engine's purged walk-forward validation (see the Optimize tab for
          that — Market Intelligence Validation panel).
        </InfoDot>}
      >
        <p style={{ color: 'var(--text-secondary)', fontSize: 14, marginBottom: 16, lineHeight: 1.5 }}>
          Runs a historical simulation of the legacy ML+setup pipeline using your CURRENT settings — fees, trailing stops, cooldown, and confidence thresholds — identical to live trading. Reports what those settings would have done; it doesn't search for better ones.
        </p>
        <button className="btn btn-primary" onClick={runBacktest} disabled={backtestLoading}>
          {backtestLoading ? 'Running Backtest...' : 'Run Backtest'}
        </button>

        {backtestError && <div className="error-message" style={{ marginTop: 16 }}>{backtestError}</div>}

        {backtestResults && (
          <div style={{ marginTop: 16 }}>
            <div style={{ padding: '10px 14px', background: 'var(--bg-card-alt)', borderRadius: 10, border: '1px solid var(--border)', marginBottom: 16 }}>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>
                {backtestResults.leverage}x · {(backtestResults.risk_per_trade * 100).toFixed(1)}% risk ·{' '}
                {(backtestResults.stop_loss_pct * 100).toFixed(1)}% SL · {(backtestResults.take_profit_pct * 100).toFixed(1)}% TP
                {backtestResults.trailing_stop_pct ? ` · ${(backtestResults.trailing_stop_pct * 100).toFixed(1)}% trail` : ''} ·{' '}
                {(backtestResults.min_confidence * 100).toFixed(0)}% conf · {backtestResults.timeframe || '5m'}
              </div>
              <div style={{ fontSize: 12, color: 'var(--accent)', fontWeight: 600 }}>
                {backtestResults.selected_coins?.map(coinOf).join(', ')}
              </div>
            </div>

            <h3 style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>
              Overall · {backtestResults.period_days} days
            </h3>
            <div className="stat-grid">
              <div className="stat-item">
                <div className={`stat-value ${backtestResults.total_pnl >= 0 ? 'positive' : 'negative'}`}>{fmtSignedUsd(backtestResults.total_pnl)}</div>
                <div className="stat-label">Profit/Loss</div>
              </div>
              <div className="stat-item">
                <div className={`stat-value ${backtestResults.total_return >= 0 ? 'positive' : 'negative'}`}>
                  {backtestResults.total_return >= 0 ? '+' : ''}{backtestResults.total_return?.toFixed(2)}%
                </div>
                <div className="stat-label">Return</div>
              </div>
              <div className="stat-item">
                <div className={`stat-value ${backtestResults.win_rate >= 50 ? 'positive' : 'negative'}`}>{backtestResults.win_rate?.toFixed(1)}%</div>
                <div className="stat-label">Win Rate</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{backtestResults.total_trades}</div>
                <div className="stat-label">Trades</div>
              </div>
            </div>

            <div style={{ marginTop: 12 }}>
              <div className="indicator-row"><span className="indicator-name">Starting Balance</span><span className="indicator-value">{fmtUsd(backtestResults.starting_balance)}</span></div>
              <div className="indicator-row">
                <span className="indicator-name">Final Balance</span>
                <span className={`indicator-value ${backtestResults.final_balance >= backtestResults.starting_balance ? 'positive' : 'negative'}`}>{fmtUsd(backtestResults.final_balance)}</span>
              </div>
              <div className="indicator-row"><span className="indicator-name">Max Drawdown</span><span className="indicator-value negative">-{backtestResults.max_drawdown?.toFixed(2)}%</span></div>
              <div className="indicator-row">
                <span className="indicator-name">Total Fees (0.06%)</span>
                <span className="indicator-value" style={{ color: 'var(--accent-gold)' }}>-{fmtUsd(backtestResults.total_fees || 0)}</span>
              </div>
              {backtestResults.monthly_roi > 0 && (
                <div className="indicator-row"><span className="indicator-name">Monthly ROI</span><span className="indicator-value positive">+{backtestResults.monthly_roi?.toFixed(2)}%</span></div>
              )}
              {backtestResults.calmar_ratio > 0 && (
                <div className="indicator-row"><span className="indicator-name">Calmar Ratio</span><span className="indicator-value" style={{ color: 'var(--accent-blue)' }}>{backtestResults.calmar_ratio?.toFixed(2)}</span></div>
              )}
              {backtestResults.sharpe_ratio > 0 && (
                <div className="indicator-row"><span className="indicator-name">Sharpe Ratio</span><span className="indicator-value" style={{ color: 'var(--accent-blue)' }}>{backtestResults.sharpe_ratio?.toFixed(3)}</span></div>
              )}
              {backtestResults.months_to_1m && (
                <div className="indicator-row">
                  <span className="indicator-name">Time to $1M (from $1K)</span>
                  <span className="indicator-value" style={{ color: 'var(--accent-green)' }}>
                    {backtestResults.months_to_1m >= 24 ? `${(backtestResults.months_to_1m / 12).toFixed(1)} yrs` : `${backtestResults.months_to_1m} mo`}
                  </span>
                </div>
              )}
              {backtestResults.total_return > 0 && backtestResults.max_drawdown > 0 && (
                <div className="indicator-row">
                  <span className="indicator-name">Risk-Adj Score</span>
                  <span className="indicator-value" style={{ color: 'var(--accent-blue)' }}>
                    {(backtestResults.win_rate * backtestResults.total_return / Math.max(backtestResults.max_drawdown, 1)).toFixed(1)}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </Card>

      {backtestResults && backtestResults.total_return > 0 && (
        <GrowthProjection periodDays={backtestResults.period_days} totalReturn={backtestResults.total_return} monthlyRoi={backtestResults.monthly_roi} />
      )}

      {backtestResults?.coin_results && (
        <Card title="Results by Coin">
          {backtestResults.coin_results.map((coinResult, idx) => (
            <div key={idx} style={{
              padding: 12, marginBottom: 8, background: 'var(--bg-card-alt)', borderRadius: 10,
              border: expandedCoin === coinResult.coin ? '1px solid var(--accent)' : '1px solid var(--border)', cursor: 'pointer',
            }} onClick={() => setExpandedCoin(expandedCoin === coinResult.coin ? null : coinResult.coin)}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 700, color: 'var(--accent)', fontSize: 15 }}>{coinResult.coin}</span>
                <span className={coinResult.total_pnl >= 0 ? 'positive' : 'negative'} style={{ fontWeight: 700 }}>{fmtSignedUsd(coinResult.total_pnl)}</span>
              </div>
              {coinResult.error ? (
                <div style={{ color: 'var(--accent-gold)', fontSize: 12, marginTop: 4 }}>{coinResult.error}</div>
              ) : (
                <div style={{ display: 'flex', gap: 12, fontSize: 12, color: 'var(--text-secondary)', marginTop: 6 }}>
                  <span>Return: <strong className={coinResult.total_return >= 0 ? 'positive' : 'negative'}>{coinResult.total_return >= 0 ? '+' : ''}{coinResult.total_return?.toFixed(1)}%</strong></span>
                  <span>Trades: <strong style={{ color: 'var(--text-primary)' }}>{coinResult.total_trades}</strong></span>
                  <span>Win: <strong style={{ color: coinResult.win_rate >= 50 ? 'var(--accent-green)' : 'var(--accent-red)' }}>{coinResult.win_rate?.toFixed(0)}%</strong></span>
                </div>
              )}
              {expandedCoin === coinResult.coin && coinResult.trades?.length > 0 && (
                <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid var(--border)' }}>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>Trade log ({coinResult.trades.length})</div>
                  {coinResult.trades.map((trade, tIdx) => (
                    <div key={tIdx} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 8px', background: 'rgba(0,0,0,0.2)', borderRadius: 4, marginBottom: 4, fontSize: 11 }}>
                      <span style={{ color: trade.side === 'long' ? 'var(--accent-green)' : 'var(--accent-red)', textTransform: 'uppercase', fontWeight: 700 }}>{trade.side}</span>
                      <span style={{ color: 'var(--text-muted)' }}>Entry: ${trade.entry}</span>
                      <span style={{ color: 'var(--text-muted)' }}>Exit: ${trade.exit}</span>
                      <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{trade.reason}</span>
                      <span className={trade.pnl >= 0 ? 'positive' : 'negative'} style={{ fontWeight: 700 }}>{trade.pnl >= 0 ? '+' : ''}${trade.pnl}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </Card>
      )}

      {backtestResults?.all_trades?.length > 0 && (
        <Card title="All Trades" right={
          <button onClick={() => setShowAllTrades(!showAllTrades)} style={{
            padding: '6px 12px', background: 'var(--accent-dim)', border: '1px solid rgba(233,69,96,0.3)',
            borderRadius: 8, color: 'var(--accent)', cursor: 'pointer', fontSize: 12, fontWeight: 600,
          }}>{showAllTrades ? 'Hide' : `Show ${backtestResults.all_trades.length} trades`}</button>
        }>
          {showAllTrades && (
            <div style={{ maxHeight: 300, overflowY: 'auto' }}>
              {backtestResults.all_trades.map((trade, idx) => (
                <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', padding: 8, background: idx % 2 === 0 ? 'rgba(0,0,0,0.15)' : 'transparent', borderRadius: 4, fontSize: 12, alignItems: 'center' }}>
                  <span style={{ color: 'var(--accent)', fontWeight: 700, minWidth: 40 }}>{trade.coin}</span>
                  <span style={{ color: trade.side === 'long' ? 'var(--accent-green)' : 'var(--accent-red)', textTransform: 'uppercase', minWidth: 50, fontWeight: 600 }}>{trade.side}</span>
                  <span style={{ color: 'var(--text-muted)' }}>${trade.entry} → ${trade.exit}</span>
                  <span className={trade.pnl >= 0 ? 'positive' : 'negative'} style={{ fontWeight: 700, minWidth: 70, textAlign: 'right' }}>{trade.pnl >= 0 ? '+' : ''}${trade.pnl} ({trade.pnl_pct}%)</span>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}

      <Card>
        <h2 style={{ fontSize: 18, marginBottom: 6, fontWeight: 700 }}>{strategies.name}</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: 14, lineHeight: 1.6, marginBottom: 10 }}>{strategies.description}</p>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.5 }}>
          This describes the legacy ML+setup engine's feature set below. It's still what trades when the
          Market Intelligence Engine doesn't have a validated opinion — see Insights for how the two combine.
        </p>
      </Card>

      {strategies.components?.map((component, i) => (
        <div key={i} className="card strategy-section">
          <div className="strategy-title">
            {component.name}
            <span className="strategy-weight">{component.weight}</span>
          </div>
          <p className="strategy-desc">{component.description}</p>
          {component.details && <ul className="strategy-list">{component.details.map((d, j) => <li key={j}>{d}</li>)}</ul>}
          {component.indicators && <ul className="strategy-list">{component.indicators.map((ind, j) => <li key={j}><strong>{ind.name}:</strong> {ind.desc}</li>)}</ul>}
        </div>
      ))}
    </>
  )
}
