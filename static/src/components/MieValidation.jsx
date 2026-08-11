import React, { useEffect, useState } from 'react'
import { Card, Badge, Meter, InfoDot, EmptyState } from './Primitives'
import { fmtPct, fmtSignedPct, horizonLabel } from '../utils/format'

/**
 * The methodology-aligned counterpart to a parameter search: instead of "we
 * tested N configs and this one had the best backtested ROI", this shows
 * what has actually SURVIVED purged walk-forward validation — trained on
 * one period, tested on a later one the model never saw, per time horizon.
 * Nothing here was chosen for looking good in-sample; `validated` is a
 * pass/fail gate, not a ranking.
 */
export default function MieValidationPanel({ api, compact = false }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    api.get('/mie/validation')
      .then(r => setData(r.data))
      .catch(() => setError('Failed to load validation report.'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Card title="Market Intelligence Validation"><EmptyState>Loading…</EmptyState></Card>
  if (error) return <Card title="Market Intelligence Validation"><EmptyState>{error}</EmptyState></Card>
  if (!data?.available) {
    return (
      <Card title="Market Intelligence Validation">
        <EmptyState>{data?.reason || 'Not available on this account.'}</EmptyState>
      </Card>
    )
  }

  return (
    <Card
      title="Market Intelligence Validation"
      right={<InfoDot>
        Each horizon is trained on one period and tested on a later one it never saw. "Validated" means it
        survived that test with positive expectancy across multiple folds — not that it looked good on the
        data it trained on.
      </InfoDot>}
    >
      {!compact && (
        <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 14, lineHeight: 1.5 }}>
          {data.resolved_observations.toLocaleString()} observations recorded
          {data.last_fit_hours_ago != null && ` · last re-validated ${data.last_fit_hours_ago}h ago`}
          {' · '}gate is {data.gate_enabled ? 'enabled' : 'off'} in Settings
        </p>
      )}

      {data.horizons.length === 0 && (
        <EmptyState>No fit attempted yet — needs enough resolved observations first. This fills in automatically.</EmptyState>
      )}

      {data.horizons.map(h => (
        <div key={h.horizon_sec} style={{ padding: '12px 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
            <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>{horizonLabel(h.horizon_sec)} horizon</span>
            <Badge color={h.validated ? 'var(--accent-green)' : 'var(--text-secondary)'} filled={h.validated}>
              {h.validated ? 'VALIDATED' : h.fitted ? 'NOT VALIDATED' : 'NOT FIT YET'}
            </Badge>
          </div>
          {h.fitted ? (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8 }}>
                <span>OOS trades: <strong style={{ color: 'var(--text-primary)' }}>{h.pooled_trades}</strong></span>
                <span>Folds with edge: <strong style={{ color: 'var(--text-primary)' }}>{h.folds_with_edge}/{h.n_splits_run}</strong></span>
                <span>
                  Expectancy/trade
                  <InfoDot>Average net return per trade this model would have taken in held-out data, after the assumed round-trip cost — a raw return, not an R-multiple (that's the separate Quality Gate's unit).</InfoDot>:{' '}
                  <strong style={{ color: (h.pooled_expectancy || 0) >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>{h.pooled_expectancy != null ? fmtSignedPct(h.pooled_expectancy * 100, 3) : '—'}</strong>
                </span>
                <span>Win rate: <strong style={{ color: 'var(--text-primary)' }}>{h.pooled_win_rate != null ? fmtPct(h.pooled_win_rate * 100, 0) : '—'}</strong></span>
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Model reliability</div>
              <Meter value={h.model_confidence * 100} color={h.model_confidence >= 0.5 ? 'var(--accent-green)' : 'var(--accent-gold)'} />
            </>
          ) : (
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{h.reason}</div>
          )}
          {!h.validated && h.fitted && (
            <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)' }}>{h.reason}</div>
          )}
        </div>
      ))}
    </Card>
  )
}
