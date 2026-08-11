import React, { useEffect, useState } from 'react'
import { Card, EmptyState } from '../components/Primitives'
import { fmtUsd, fmtSignedUsd, coinOf } from '../utils/format'

export default function TradesPage({ botStatus, api }) {
  const [dbTrades, setDbTrades] = useState([])
  useEffect(() => {
    api.get('/trades').then(r => setDbTrades(r.data?.trades || [])).catch(() => {})
  }, [])
  const trades = botStatus?.recent_trades?.length ? botStatus.recent_trades : dbTrades

  return (
    <Card title="Recent Trades">
      {trades.length === 0 ? (
        <EmptyState>No trades yet. Start the bot to begin trading.</EmptyState>
      ) : (
        <div className="trade-list">
          {trades.slice().reverse().map((trade, i) => (
            <div key={i} className="trade-item">
              <div>
                <div className={`trade-side ${trade.side}`}>{trade.type?.toUpperCase()} {trade.side?.toUpperCase()}</div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>
                  {coinOf(trade.symbol)} · {fmtUsd(trade.price)}
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                {trade.pnl !== undefined && (
                  <div className={`trade-pnl ${trade.pnl >= 0 ? 'positive' : 'negative'}`}>{fmtSignedUsd(trade.pnl)}</div>
                )}
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>
                  {trade.reason
                    ? trade.reason
                    : trade.conf_tier
                      ? `${trade.conf_tier} · ${trade.leverage || ''}x`
                      : trade.confidence
                        ? (trade.confidence * 100).toFixed(0) + '% conf'
                        : ''}
                  {trade.quality != null ? ` · quality ${trade.quality}/100` : ''}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}
