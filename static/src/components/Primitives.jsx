import React, { useLayoutEffect, useRef, useState } from 'react'

/* Reusable building blocks — every page composes from these instead of
   re-declaring the same inline style object. Keeps the visual language
   consistent and means a spacing/color tweak happens in one place. */

export function Card({ title, right, children, style, tone }) {
  const toneBorder = {
    green: '1px solid rgba(0,212,170,0.25)',
    red: '1px solid rgba(255,68,68,0.25)',
    gold: '1px solid rgba(245,166,35,0.25)',
  }[tone]
  return (
    <div className="card" style={{ ...(toneBorder ? { border: toneBorder } : {}), ...style }}>
      {(title || right) && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, gap: 8 }}>
          {title && <h2 className="card-title" style={{ margin: 0 }}>{title}</h2>}
          {right}
        </div>
      )}
      {children}
    </div>
  )
}

export function Badge({ children, color = 'var(--text-secondary)', bg, filled = false }) {
  const background = bg || (filled ? color : `color-mix(in srgb, ${color} 15%, transparent)`)
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: '3px 10px', borderRadius: 20, fontSize: 11, fontWeight: 700,
      letterSpacing: 0.3, background,
      color: filled ? '#0a0b0f' : color,
      border: filled ? 'none' : `1px solid color-mix(in srgb, ${color} 35%, transparent)`,
    }}>
      {children}
    </span>
  )
}

export function StatRow({ label, value, color, sub }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '10px 0', borderBottom: '1px solid var(--border)',
    }}>
      <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{label}</span>
      <span style={{ textAlign: 'right' }}>
        <span style={{ fontSize: 14, fontWeight: 700, color: color || 'var(--text-primary)' }}>{value}</span>
        {sub && <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{sub}</div>}
      </span>
    </div>
  )
}

export function InlineStat({ label, value, color }) {
  return (
    <div style={{ padding: '10px 12px', background: 'var(--bg-card-alt)', borderRadius: 10, border: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
      <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{label}</span>
      <span style={{ fontSize: 14, fontWeight: 700, color: color || 'var(--text-primary)' }}>{value}</span>
    </div>
  )
}

// A meter from 0-100 with a colored fill — used for quality/confidence scores
// so a number is always backed by a visual read at a glance.
export function Meter({ value, max = 100, color, height = 6 }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100))
  return (
    <div style={{ height, background: 'var(--border)', borderRadius: height, overflow: 'hidden' }}>
      <div style={{ height: '100%', width: `${pct}%`, background: color || 'var(--accent)', borderRadius: height, transition: 'width 0.4s ease' }} />
    </div>
  )
}

// Click-to-toggle explainer for trading jargon (ADX, Kelly fraction, R-multiple…).
// Tap targets beat hover on a phone, which is the primary surface for this app.
//
// Positioned with `fixed` + a measured left offset rather than a naive
// `absolute; left: 0` — the dot can sit anywhere in a card (often mid-line,
// well past the horizontal midpoint), and a fixed-width popup anchored to
// the dot's own left edge routinely ran off the right side of a phone
// viewport. Clamping against the actual window width keeps it on-screen
// wherever the dot happens to be.
const TOOLTIP_WIDTH = 220
export function InfoDot({ children }) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState(null)
  const btnRef = useRef(null)

  useLayoutEffect(() => {
    if (!open || !btnRef.current) return
    const rect = btnRef.current.getBoundingClientRect()
    const margin = 12
    const left = Math.min(
      Math.max(margin, rect.left),
      window.innerWidth - TOOLTIP_WIDTH - margin
    )
    setPos({ top: rect.bottom + 6, left })
  }, [open])

  return (
    <span style={{ position: 'relative', display: 'inline-block' }}>
      <button
        ref={btnRef}
        type="button"
        onClick={(e) => { e.stopPropagation(); setOpen(o => !o) }}
        aria-label="More info"
        style={{
          width: 16, height: 16, borderRadius: '50%', border: '1px solid var(--text-muted)',
          background: 'transparent', color: 'var(--text-muted)', fontSize: 10, fontWeight: 700,
          lineHeight: '14px', cursor: 'pointer', padding: 0, marginLeft: 5, verticalAlign: 'middle',
        }}
      >i</button>
      {open && pos && (
        <>
          <div onClick={() => setOpen(false)} style={{ position: 'fixed', inset: 0, zIndex: 40 }} />
          <div style={{
            position: 'fixed', zIndex: 41, top: pos.top, left: pos.left, width: TOOLTIP_WIDTH,
            background: '#1c2130', border: '1px solid var(--border-hover)', borderRadius: 10,
            padding: '10px 12px', fontSize: 12, lineHeight: 1.5, color: 'var(--text-secondary)',
            boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
          }}>
            {children}
          </div>
        </>
      )}
    </span>
  )
}

export function EmptyState({ children }) {
  return (
    <p style={{ color: 'var(--text-secondary)', textAlign: 'center', padding: '24px 12px', fontSize: 13, lineHeight: 1.6 }}>
      {children}
    </p>
  )
}

// Segmented control used for the sub-navigation inside a page (e.g. Insights'
// Right Now / Post-Mortem split).
export function SegmentedControl({ options, value, onChange }) {
  return (
    <div style={{ display: 'flex', borderRadius: 10, overflow: 'hidden', border: '1px solid var(--border)', marginBottom: 14 }}>
      {options.map(opt => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          style={{
            flex: 1, padding: '9px 8px', border: 'none', fontSize: 12, fontWeight: 700,
            cursor: 'pointer', letterSpacing: 0.3,
            background: value === opt.value ? 'var(--accent-dim)' : 'var(--bg-card-alt)',
            color: value === opt.value ? 'var(--accent)' : 'var(--text-secondary)',
          }}
        >{opt.label}</button>
      ))}
    </div>
  )
}

// On/off switch bound to a boolean — used for the MIE gate toggle etc.
export function ToggleSwitch({ checked, onChange, disabled, labelOn = 'On', labelOff = 'Off' }) {
  return (
    <button
      type="button"
      onClick={() => !disabled && onChange(!checked)}
      disabled={disabled}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 8, border: 'none', cursor: disabled ? 'not-allowed' : 'pointer',
        background: 'transparent', padding: 0, opacity: disabled ? 0.5 : 1,
      }}
    >
      <span style={{
        width: 40, height: 22, borderRadius: 20, position: 'relative', transition: 'background 0.2s',
        background: checked ? 'var(--accent-green)' : 'var(--border-hover)',
      }}>
        <span style={{
          position: 'absolute', top: 2, left: checked ? 20 : 2, width: 18, height: 18, borderRadius: '50%',
          background: '#fff', transition: 'left 0.2s',
        }} />
      </span>
      <span style={{ fontSize: 13, fontWeight: 600, color: checked ? 'var(--accent-green)' : 'var(--text-secondary)' }}>
        {checked ? labelOn : labelOff}
      </span>
    </button>
  )
}
