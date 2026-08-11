import React, { useEffect, useState } from 'react'
import { Card, Badge, Meter, InfoDot, EmptyState, SegmentedControl } from '../components/Primitives'
import {
  fmtPct, fmtR, coinOf, colorForScore, colorForSignal, regimeLabel, regimeColor,
  humanizeBucketKey, timeAgo,
} from '../utils/format'

const MIE_FIT_TARGET = 500   // matches TradingService._mie_min_rows_to_fit
const QUALITY_TRADE_TARGET = 30
const QUALITY_BUCKET_TARGET = 3

/* ─────────────────────── Right Now: per-coin reasoning ─────────────────── */

function CoinReasonCard({ coin, signal, mie }) {
  const hasSignal = !!signal
  const tradingNow = hasSignal && signal.signal !== 0
  const dir = signal?.signal === 1 ? 'LONG' : signal?.signal === -1 ? 'SHORT' : null

  return (
    <div style={{ padding: 14, marginBottom: 10, background: 'var(--bg-card-alt)', borderRadius: 12, border: '1px solid var(--border)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <span style={{ fontWeight: 700, color: 'var(--accent)', fontSize: 16 }}>{coin}</span>
        {tradingNow ? (
          <Badge color={colorForSignal(signal.signal)} filled>{dir}</Badge>
        ) : (
          <Badge color="var(--text-secondary)">NO TRADE</Badge>
        )}
      </div>

      {!hasSignal && <EmptyState>Waiting for the first read on this coin.</EmptyState>}

      {hasSignal && (
        <>
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', fontSize: 12, color: 'var(--text-secondary)', marginBottom: 10 }}>
            <span>Confidence: <strong style={{ color: colorForScore(signal.confidence * 100, { good: 70, warn: 65 }) }}>{fmtPct(signal.confidence * 100)}</strong></span>
            {signal.regime && <span>Regime: <strong style={{ color: regimeColor(signal.regime) }}>{regimeLabel(signal.regime)}</strong></span>}
            {signal.quality != null && <span>Quality score: <strong style={{ color: colorForScore(signal.quality) }}>{signal.quality}/100</strong></span>}
          </div>

          {/* Quality-gate reasoning (the observation/edge-analytics system) */}
          {signal.quality_reason && (
            <div style={{ padding: '8px 10px', background: 'rgba(245,166,35,0.06)', border: '1px solid rgba(245,166,35,0.15)', borderRadius: 8, fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8 }}>
              <strong style={{ color: 'var(--accent-gold)' }}>Quality Gate:</strong> {signal.quality_reason}
            </div>
          )}

          {/* MIE reasoning */}
          {mie && (
            <div style={{ padding: '8px 10px', background: 'rgba(74,158,255,0.06)', border: '1px solid rgba(74,158,255,0.15)', borderRadius: 8, fontSize: 12, color: 'var(--text-secondary)' }}>
              <strong style={{ color: 'var(--accent-blue)' }}>Market Intelligence:</strong>{' '}
              {mie.action === 'DO_NOTHING'
                ? (mie.blockers?.[0] || mie.reasons?.[0] || 'No validated positive-EV setup right now.')
                : `Agrees — ${mie.action}, ${mie.historical_sample ?? 0} comparable observations, expectancy ${fmtR(mie.expectancy_r)}.`}
              {mie.historical_sample != null && mie.action === 'DO_NOTHING' && (
                <span style={{ display: 'block', marginTop: 4, color: 'var(--text-muted)' }}>
                  {mie.historical_sample} comparable observations · regime {regimeLabel(mie.regime)}
                </span>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

function RightNowTab({ botStatus }) {
  const selectedCoins = botStatus?.selected_coins || []
  const coinSignals = botStatus?.coin_signals || {}
  const mieDecisions = botStatus?.mie_decisions || {}

  if (!botStatus?.running) {
    return <Card title="Right Now"><EmptyState>Start the bot to see live reasoning per coin.</EmptyState></Card>
  }

  return (
    <Card title="Right Now" right={<span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Updates every few seconds</span>}>
      <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 14, lineHeight: 1.5 }}>
        What the bot is deciding for each coin this cycle, and why — in plain language.
      </p>
      {selectedCoins.length === 0 && <EmptyState>No coins selected. Add some in Settings.</EmptyState>}
      {selectedCoins.map(sym => {
        const coin = coinOf(sym)
        return <CoinReasonCard key={coin} coin={coin} signal={coinSignals[coin]} mie={mieDecisions[coin]} />
      })}
    </Card>
  )
}

/* ─────────────────────── Gate status / arming progress ─────────────────── */

function GateProgress({ label, current, target, ready, subtitle }) {
  const pct = target ? Math.min(100, (current / target) * 100) : (ready ? 100 : 0)
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
        <span style={{ fontSize: 13, color: 'var(--text-primary)', fontWeight: 600 }}>{label}</span>
        <span style={{ fontSize: 12, color: ready ? 'var(--accent-green)' : 'var(--text-secondary)' }}>
          {ready ? 'Ready' : target ? `${Math.min(current, target).toLocaleString()} / ${target.toLocaleString()}` : '—'}
        </span>
      </div>
      <Meter value={pct} color={ready ? 'var(--accent-green)' : 'var(--accent-gold)'} />
      {subtitle && <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>{subtitle}</div>}
    </div>
  )
}

function GateStatusCard({ botStatus, edgeReport }) {
  const mieOn = !!botStatus?.mie_gate_enabled
  const mieObs = botStatus?.mie_resolved_observations || 0
  const mieReady = !!botStatus?.mie_any_validated

  const profile = edgeReport?.profile
  const qualifiedBuckets = profile ? Object.values(profile.buckets || {}).filter(b => b.qualified).length : 0
  const tradeN = profile?.n || 0
  const qualityReady = !!edgeReport?.gate_armed

  return (
    <Card title="Evidence Gates"
      right={<InfoDot>Both systems default to letting the bot trade normally. They only step in once they've accumulated enough evidence to say a condition doesn't have an edge — that's when they start blocking entries.</InfoDot>}
    >
      <div style={{ marginBottom: 4 }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 10 }}>
          Market Intelligence Engine {mieOn ? '' : '(off — enable in Settings)'}
        </div>
        <GateProgress
          label="Observations for first model fit"
          current={mieObs} target={mieReady ? null : MIE_FIT_TARGET} ready={mieReady}
          subtitle={mieReady ? 'A validated model exists for at least one time horizon.' : 'Records market state every cycle, whether or not it trades.'}
        />
      </div>
      <div style={{ marginTop: 8 }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 10 }}>
          Quality Gate (edge post-mortem)
        </div>
        <GateProgress
          label="Closed trades analyzed"
          current={tradeN} target={qualityReady ? null : QUALITY_TRADE_TARGET} ready={qualityReady}
        />
        {!qualityReady && (
          <GateProgress
            label="Qualified condition buckets"
            current={qualifiedBuckets} target={QUALITY_BUCKET_TARGET} ready={qualifiedBuckets >= QUALITY_BUCKET_TARGET}
          />
        )}
      </div>
    </Card>
  )
}

/* ─────────────────────── Post-mortem: readable edge report ─────────────── */

function BucketRow({ bucketKey, bucket, scaleMax }) {
  const lb = bucket.expectancy_lb
  const positive = lb >= 0
  const barPct = scaleMax ? Math.min(100, (Math.abs(lb) / scaleMax) * 100) : 0
  return (
    <div style={{ padding: '10px 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8, marginBottom: 4 }}>
        <span style={{ fontSize: 13, color: 'var(--text-primary)' }}>{humanizeBucketKey(bucketKey)}</span>
        <span style={{ fontSize: 13, fontWeight: 700, color: positive ? 'var(--accent-green)' : 'var(--accent-red)', whiteSpace: 'nowrap' }}>
          {fmtR(lb)}{bucket.provisional ? ' *' : ''}
        </span>
      </div>
      <Meter value={barPct} color={positive ? 'var(--accent-green)' : 'var(--accent-red)'} height={4} />
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>
        n={bucket.n} · win rate {fmtPct(bucket.win_rate * 100, 0)} · point estimate {fmtR(bucket.expectancy)}
      </div>
    </div>
  )
}

function PostMortemTab({ edgeReport, loading, error, onRefresh }) {
  if (loading) return <Card title="Post-Mortem"><EmptyState>Loading…</EmptyState></Card>
  if (error) return <Card title="Post-Mortem"><EmptyState>{error}</EmptyState></Card>

  const profile = edgeReport?.profile
  if (!profile?.ok) {
    return (
      <Card title="Post-Mortem">
        <EmptyState>{edgeReport?.summary || 'Not enough closed trades yet — this fills in automatically as the bot trades.'}</EmptyState>
      </Card>
    )
  }

  const buckets = Object.entries(profile.buckets || {}).filter(([, b]) => b.qualified)
  const best = [...buckets].sort((a, b) => b[1].expectancy_lb - a[1].expectancy_lb).slice(0, 6)
  const bleeding = [...buckets].filter(([, b]) => b.expectancy_lb < 0).sort((a, b) => a[1].expectancy_lb - b[1].expectancy_lb).slice(0, 6)
  const scaleMax = Math.max(0.1, ...buckets.map(([, b]) => Math.abs(b.expectancy_lb)))

  return (
    <>
      <Card title="Overall" right={<button onClick={onRefresh} style={{ background: 'none', border: 'none', color: 'var(--accent)', fontSize: 12, fontWeight: 700, cursor: 'pointer' }}>Refresh</button>}>
        <div className="stat-grid">
          <div className="stat-item">
            <div className={`stat-value ${profile.expectancy >= 0 ? 'positive' : 'negative'}`}>{fmtR(profile.expectancy)}</div>
            <div className="stat-label">Expectancy</div>
          </div>
          <div className="stat-item">
            <div className={`stat-value ${profile.expectancy_lb >= 0 ? 'positive' : 'negative'}`}>{fmtR(profile.expectancy_lb)}</div>
            <div className="stat-label">
              Lower Bound
              <InfoDot>95% confidence floor on expectancy — the number the gates actually act on, since a small sample's average is mostly noise.</InfoDot>
            </div>
          </div>
          <div className="stat-item">
            <div className="stat-value">{fmtPct(profile.win_rate * 100, 0)}</div>
            <div className="stat-label">Win Rate</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">{profile.n}</div>
            <div className="stat-label">Trades Analyzed</div>
          </div>
        </div>
        {edgeReport.cost_r != null && (
          <div style={{ marginTop: 12, padding: '10px 12px', background: 'var(--bg-card-alt)', borderRadius: 10, border: '1px solid var(--border)', fontSize: 12, color: 'var(--text-secondary)' }}>
            Round-trip cost ≈ <strong style={{ color: 'var(--accent-gold)' }}>{fmtR(edgeReport.cost_r)}</strong> per trade —
            {' '}the edge {profile.expectancy_lb > edgeReport.cost_r
              ? <strong style={{ color: 'var(--accent-green)' }}> clears costs</strong>
              : <strong style={{ color: 'var(--accent-red)' }}> does not clear costs yet</strong>} on the lower bound.
          </div>
        )}
      </Card>

      {best.length > 0 && (
        <Card title="Conditions Carrying the Edge" right={<InfoDot>Ranked by the 95% lower bound, not the raw average — this is what protects against a lucky slice of trades looking better than it is.</InfoDot>}>
          {best.map(([k, b]) => <BucketRow key={k} bucketKey={k} bucket={b} scaleMax={scaleMax} />)}
        </Card>
      )}

      {bleeding.length > 0 && (
        <Card tone="red" title="Conditions to Stop Trading" right={<InfoDot>Negative lower bound — these conditions have not shown a measured edge after enough trades to trust the number.</InfoDot>}>
          {bleeding.map(([k, b]) => <BucketRow key={k} bucketKey={k} bucket={b} scaleMax={scaleMax} />)}
        </Card>
      )}

      {buckets.some(([, b]) => b.provisional) && (
        <p style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center', margin: '4px 0 14px' }}>
          * Provisional — estimated from recorded market observations rather than a full closed trade, until enough real trades accumulate for that condition.
        </p>
      )}
    </>
  )
}

/* ─────────────────────────── Page shell ─────────────────────────── */

export default function InsightsPage({ botStatus, api }) {
  const [tab, setTab] = useState('now')
  const [edgeReport, setEdgeReport] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const loadEdgeReport = () => {
    setLoading(true)
    setError('')
    api.get('/edge/report')
      .then(r => setEdgeReport(r.data))
      .catch(() => setError('Failed to load the post-mortem report.'))
      .finally(() => setLoading(false))
  }

  useEffect(() => { loadEdgeReport() }, [])

  return (
    <>
      <div className="card" style={{ paddingBottom: 4 }}>
        <h2 className="card-title">Insights</h2>
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 14 }}>
          Two systems watch every market condition, whether or not the bot trades on it, and only let a
          trade through once there's measured evidence of an edge. This page is where you see their reasoning.
        </p>
        <SegmentedControl
          value={tab} onChange={setTab}
          options={[
            { value: 'now', label: 'Right Now' },
            { value: 'gates', label: 'Gate Status' },
            { value: 'postmortem', label: 'Post-Mortem' },
          ]}
        />
      </div>

      {tab === 'now' && <RightNowTab botStatus={botStatus} />}
      {tab === 'gates' && <GateStatusCard botStatus={botStatus} edgeReport={edgeReport} />}
      {tab === 'postmortem' && (
        <PostMortemTab edgeReport={edgeReport} loading={loading} error={error} onRefresh={loadEdgeReport} />
      )}
    </>
  )
}
