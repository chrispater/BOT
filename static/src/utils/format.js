// Shared formatting + labeling helpers. Centralized so a number reads the
// same way everywhere in the app instead of each card inventing its own
// rounding/sign/color rules.

export const fmtUsd = (v, opts = {}) => {
  const n = Number(v) || 0
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: 2, ...opts })}`
}

export const fmtSignedUsd = (v) => {
  const n = Number(v) || 0
  return `${n >= 0 ? '+' : ''}${fmtUsd(n)}`
}

export const fmtPct = (v, digits = 1) => {
  const n = Number(v) || 0
  return `${n.toFixed(digits)}%`
}

export const fmtSignedPct = (v, digits = 1) => {
  const n = Number(v) || 0
  return `${n >= 0 ? '+' : ''}${n.toFixed(digits)}%`
}

export const fmtR = (v, digits = 3) => {
  const n = Number(v) || 0
  return `${n >= 0 ? '+' : ''}${n.toFixed(digits)}R`
}

export const coinOf = (symbol) => (symbol ? symbol.split('/')[0] : 'Unknown')

// Traffic-light coloring used across confidence/quality/win-rate displays —
// one place to tune what counts as "good" vs "risky".
export const colorForScore = (score, { good = 70, warn = 50 } = {}) => {
  if (score >= good) return 'var(--accent-green)'
  if (score >= warn) return 'var(--accent-gold)'
  return 'var(--accent-red)'
}

export const colorForSignal = (signal) => {
  if (signal === 1) return 'var(--accent-green)'
  if (signal === -1) return 'var(--accent-red)'
  return 'var(--text-secondary)'
}

export const REGIME_LABELS = {
  bull: 'Bull Market', bear: 'Bear Market', sideways: 'Sideways',
  trend_up: 'Trending Up', trend_down: 'Trending Down', mean_revert: 'Mean-Reverting',
  compression: 'Compressing (Coiled)', expansion: 'Volatility Expanding',
  panic: 'Panic / Cascading', thin: 'Thin Liquidity',
}
export const regimeLabel = (r) => REGIME_LABELS[r] || (r ? r.replace(/_/g, ' ') : 'Unknown')

export const REGIME_COLORS = {
  bull: 'var(--accent-green)', trend_up: 'var(--accent-green)', expansion: 'var(--accent-blue)',
  bear: 'var(--accent-red)', trend_down: 'var(--accent-red)', panic: 'var(--accent-red)',
  sideways: 'var(--accent-gold)', mean_revert: 'var(--accent-gold)',
  compression: 'var(--text-secondary)', thin: 'var(--text-muted)',
}
export const regimeColor = (r) => REGIME_COLORS[r] || 'var(--text-secondary)'

// Turns an edge-analytics bucket key ("regime=trend_up|vol=high") into a
// sentence fragment a non-technical reader can parse at a glance.
export function humanizeBucketKey(key) {
  if (!key) return key
  const parts = key.split('|').map(part => {
    const [field, value] = part.split('=')
    switch (field) {
      case 'regime': return regimeLabel(value)
      case 'vol': return `${value} volatility`
      case 'setup': return value === 'ml' ? 'ML signal (no pattern)' : `"${value.replace(/_/g, ' ')}" setup`
      case 'session': return { asia: 'Asia session', europe: 'Europe session', us: 'US session', late: 'late session' }[value] || value
      case 'symbol': return value
      case 'side': return value === 'long' ? 'Long trades' : 'Short trades'
      case 'btc_agree': return value === 'true' ? 'BTC confirms direction' : 'BTC disagrees'
      case 'book': return { bid_heavy: 'bid-heavy order book', ask_heavy: 'ask-heavy order book', balanced: 'balanced order book' }[value] || value
      default: return part
    }
  })
  return parts.join(' · ')
}

export function timeAgo(iso) {
  if (!iso) return '—'
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return '—'
  const s = Math.max(0, (Date.now() - then) / 1000)
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.round(s / 60)}m ago`
  if (s < 86400) return `${Math.round(s / 3600)}h ago`
  return `${Math.round(s / 86400)}d ago`
}
