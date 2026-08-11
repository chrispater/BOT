"""
Observation store — the engine's memory.

Every market state gets written here the moment it's observed, WITHOUT a label.
Outcomes get attached later, in place, once each horizon has actually matured.
This ordering is the whole point: the store can never contain a label that
peeked at the future, because at write time the future genuinely hasn't
happened yet.

Backed by SQLite by default (zero-ops, works anywhere this bot runs — a laptop,
a $5 VPS, a container) with a schema plain enough to port to Postgres later via
the same table/column shapes the rest of the codebase already uses (see
backend/database.py). Feature vectors are stored as JSON; at the volumes this
system runs at (thousands to low millions of rows) a JSON blob column plus a
handful of indexed scalar columns for filtering is simpler and more debuggable
than a wide sparse table, and avoids a migration every time a feature is added.
"""

import json
import math
import sqlite3
import threading
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from .state import MarketState, HORIZONS_SEC, horizon_label, outcome_columns
from .outcomes import resolve_outcomes, label_maturity_deadline

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH_ENV = 'MIE_DB_PATH'
DEFAULT_DB_PATH = '/tmp/bot_models/mie.sqlite3'


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt) -> str:
    if isinstance(dt, str):
        return dt
    return dt.isoformat()


class ObservationStore:
    """
    One store per process, shared across symbols/users via the `user_id` /
    `symbol` columns. Thread-safe: SQLite in WAL mode + one lock per connection
    call is sufficient at this write volume (one row per symbol per cycle).
    """

    def __init__(self, db_path: Optional[str] = None):
        import os
        self.db_path = db_path or os.environ.get(DEFAULT_DB_PATH_ENV, DEFAULT_DB_PATH)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._lock, self._conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL DEFAULT 0,
                    symbol TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    price REAL NOT NULL,
                    regime TEXT,
                    regime_confidence REAL,
                    features_json TEXT NOT NULL,
                    coverage_json TEXT,
                    outcomes_json TEXT,
                    resolved_through_sec INTEGER NOT NULL DEFAULT 0,
                    fully_resolved INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    UNIQUE(user_id, symbol, as_of)
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_obs_symbol_time ON observations(user_id, symbol, as_of)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_obs_unresolved ON observations(fully_resolved, as_of)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_obs_regime ON observations(regime)')

            # Trade decisions and their eventual outcome — the record the
            # post-mortem / attribution layer reads. Separate from raw
            # observations because not every observation becomes a decision, and
            # not every decision that fires becomes a real trade (paper vs. live,
            # or vetoed downstream by exchange-side risk checks).
            conn.execute('''
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL DEFAULT 0,
                    symbol TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    action TEXT NOT NULL,
                    horizon_sec INTEGER,
                    expected_return REAL,
                    expected_adverse REAL,
                    prob_positive REAL,
                    costs REAL,
                    regime TEXT,
                    regime_confidence REAL,
                    historical_sample INTEGER,
                    expectancy_r REAL,
                    quality INTEGER,
                    size_fraction REAL,
                    reasons_json TEXT,
                    blockers_json TEXT,
                    realized_r REAL,
                    realized_pnl REAL,
                    closed_at TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_dec_symbol_time ON decisions(user_id, symbol, as_of)')

    # ── writing observations ─────────────────────────────────────────────

    def record(self, user_id: int, state: MarketState) -> int:
        """Persist a state the moment it's observed. Idempotent on (user, symbol, as_of)."""
        with self._lock, self._conn() as conn:
            cur = conn.execute('''
                INSERT INTO observations
                    (user_id, symbol, as_of, price, regime, regime_confidence,
                     features_json, coverage_json, outcomes_json, resolved_through_sec,
                     fully_resolved, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}', 0, 0, ?)
                ON CONFLICT(user_id, symbol, as_of) DO NOTHING
            ''', (
                user_id, state.symbol, _iso(state.as_of), float(state.price),
                state.regime, float(state.regime_confidence),
                json.dumps(_clean(state.features)), json.dumps(_clean(state.coverage)),
                _iso(_now_utc()),
            ))
            return cur.lastrowid

    # ── resolving outcomes ────────────────────────────────────────────────

    def backfill_outcomes(self, user_id: int, symbol: str, df, bar_seconds: float,
                          horizons: Optional[List[int]] = None) -> int:
        """
        Given fresh OHLCV for `symbol`, resolve outcomes for any stored,
        not-yet-fully-resolved observation whose `as_of` falls inside `df`'s
        index. Call this after every fetch — cheap when nothing new has
        matured (the SQL WHERE clause does the filtering), essential for
        keeping the store from silently going stale.

        Returns the number of observation rows updated.
        """
        horizons = horizons or HORIZONS_SEC
        outcomes = resolve_outcomes(df, bar_seconds, horizons)
        if outcomes.empty:
            return 0

        updated = 0
        with self._lock, self._conn() as conn:
            rows = conn.execute('''
                SELECT id, as_of, outcomes_json, resolved_through_sec FROM observations
                WHERE user_id = ? AND symbol = ? AND fully_resolved = 0
            ''', (user_id, symbol)).fetchall()

            max_h = max(horizons)
            for row_id, as_of_str, outcomes_json, resolved_through in rows:
                ts = pd.Timestamp(as_of_str)
                if ts not in outcomes.index:
                    continue
                new_vals = outcomes.loc[ts]
                existing = json.loads(outcomes_json or '{}')
                changed = False
                highest_resolved = resolved_through
                for h in horizons:
                    lab = horizon_label(h)
                    ret_col = f'ret_{lab}'
                    if ret_col in existing and existing[ret_col] is not None:
                        continue  # already resolved at this horizon
                    ret_v = new_vals.get(ret_col)
                    if ret_v is None or (isinstance(ret_v, float) and math.isnan(ret_v)):
                        continue  # not mature yet
                    for prefix in ('ret', 'mfe', 'mae'):
                        col = f'{prefix}_{lab}'
                        existing[col] = _clean_scalar(new_vals.get(col))
                    changed = True
                    highest_resolved = max(highest_resolved, h)

                if changed:
                    fully = 1 if highest_resolved >= max_h else 0
                    conn.execute('''
                        UPDATE observations
                        SET outcomes_json = ?, resolved_through_sec = ?, fully_resolved = ?
                        WHERE id = ?
                    ''', (json.dumps(existing), highest_resolved, fully, row_id))
                    updated += 1
        if updated:
            logger.debug(f"MIE store: resolved outcomes for {updated} observation(s) on {symbol}")
        return updated

    # ── reading for training / similarity ────────────────────────────────

    def load_training_frame(self, user_id: Optional[int] = None,
                            symbol: Optional[str] = None,
                            min_resolved_sec: Optional[int] = None,
                            limit: int = 200_000):
        """
        Every observation with at least one resolved horizon, as a DataFrame:
        one column per feature, plus ret_*/mfe_*/mae_* outcome columns, plus
        symbol/regime/as_of. This is the edge model's training set.

        `user_id=None` pools observations across all users — sample size is the
        scarcest resource in this whole system, and there's no reason a bot
        running on one account can't learn from what every account observed of
        the same public market.
        """
        clauses, params = ["resolved_through_sec > 0"], []
        if user_id is not None:
            clauses.append("user_id = ?"); params.append(user_id)
        if symbol is not None:
            clauses.append("symbol = ?"); params.append(symbol)
        if min_resolved_sec is not None:
            clauses.append("resolved_through_sec >= ?"); params.append(min_resolved_sec)
        where = " AND ".join(clauses)

        with self._lock, self._conn() as conn:
            rows = conn.execute(f'''
                SELECT symbol, as_of, price, regime, regime_confidence,
                       features_json, outcomes_json
                FROM observations WHERE {where}
                ORDER BY as_of ASC LIMIT ?
            ''', params + [limit]).fetchall()

        records = []
        for symbol_, as_of, price, regime, regime_conf, feat_json, out_json in rows:
            rec = {'symbol': symbol_, 'as_of': as_of, 'price': price,
                   'regime': regime, 'regime_confidence': regime_conf}
            rec.update(json.loads(feat_json))
            rec.update(json.loads(out_json or '{}'))
            records.append(rec)
        df = pd.DataFrame.from_records(records)
        if not df.empty:
            df['as_of'] = pd.to_datetime(df['as_of'], utc=True)
            df = df.sort_values('as_of').reset_index(drop=True)
        return df

    def similar_observations(self, state: MarketState, feature_columns: List[str],
                             k: int = 50, user_id: Optional[int] = None,
                             max_candidates: int = 20_000) -> Tuple[List[dict], np.ndarray]:
        """
        Nearest neighbors to `state` in standardized feature space, restricted to
        FULLY resolved observations so every neighbor carries a complete outcome
        record. This is the literal mechanism behind "when have I seen something
        statistically similar to this before, and what happened afterward" — a
        capability no discretionary trader has, because no human remembers every
        market condition they've ever watched precisely enough to search it.

        Returns (list of matched records with their outcomes, distances array).
        """
        with self._lock, self._conn() as conn:
            clause = "fully_resolved = 1"
            params: list = []
            if user_id is not None:
                clause += " AND user_id = ?"; params.append(user_id)
            rows = conn.execute(f'''
                SELECT symbol, as_of, regime, features_json, outcomes_json
                FROM observations WHERE {clause}
                ORDER BY as_of DESC LIMIT ?
            ''', params + [max_candidates]).fetchall()

        if not rows:
            return [], np.array([])

        feat_matrix = np.full((len(rows), len(feature_columns)), np.nan)
        records = []
        for i, (symbol_, as_of, regime, feat_json, out_json) in enumerate(rows):
            feats = json.loads(feat_json)
            for j, col in enumerate(feature_columns):
                v = feats.get(col)
                if v is not None:
                    feat_matrix[i, j] = v
            rec = {'symbol': symbol_, 'as_of': as_of, 'regime': regime}
            rec.update(json.loads(out_json or '{}'))
            records.append(rec)

        query = np.array(state.vector(feature_columns), dtype=float)
        # Standardize using the candidate pool's own spread so distance is
        # comparable across features with wildly different scales (a return of
        # 0.01 and an order-book depth of 50,000 should not be compared raw).
        col_mean = np.nanmean(feat_matrix, axis=0)
        col_std = np.nanstd(feat_matrix, axis=0)
        col_std[col_std < 1e-9] = 1.0
        valid_cols = ~np.isnan(query)
        if not valid_cols.any():
            return [], np.array([])

        z_matrix = (feat_matrix - col_mean) / col_std
        z_query = (query - col_mean) / col_std
        z_matrix = np.nan_to_num(z_matrix[:, valid_cols], nan=0.0)
        z_query = np.nan_to_num(z_query[valid_cols], nan=0.0)

        dists = np.linalg.norm(z_matrix - z_query, axis=1)
        order = np.argsort(dists)[:k]
        return [records[i] for i in order], dists[order]

    def count_resolved(self, symbol: Optional[str] = None, user_id: Optional[int] = None) -> int:
        clauses, params = ["fully_resolved = 1"], []
        if symbol is not None:
            clauses.append("symbol = ?"); params.append(symbol)
        if user_id is not None:
            clauses.append("user_id = ?"); params.append(user_id)
        with self._lock, self._conn() as conn:
            return conn.execute(
                f"SELECT COUNT(*) FROM observations WHERE {' AND '.join(clauses)}", params
            ).fetchone()[0]

    # ── decisions log (for post-mortem / attribution) ────────────────────

    def record_decision(self, user_id: int, decision) -> int:
        with self._lock, self._conn() as conn:
            cur = conn.execute('''
                INSERT INTO decisions
                    (user_id, symbol, as_of, action, horizon_sec, expected_return,
                     expected_adverse, prob_positive, costs, regime, regime_confidence,
                     historical_sample, expectancy_r, quality, size_fraction,
                     reasons_json, blockers_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, decision.symbol, _iso(decision.as_of or _now_utc()), decision.action,
                decision.horizon_sec, decision.expected_return, decision.expected_adverse,
                decision.prob_positive, decision.costs, decision.regime,
                decision.regime_confidence, decision.historical_sample,
                decision.expectancy_r, decision.quality, decision.size_fraction,
                json.dumps(decision.reasons), json.dumps(decision.blockers),
                _iso(_now_utc()),
            ))
            return cur.lastrowid

    def close_decision(self, decision_id: int, realized_r: float, realized_pnl: float):
        with self._lock, self._conn() as conn:
            conn.execute('''
                UPDATE decisions SET realized_r = ?, realized_pnl = ?, closed_at = ?
                WHERE id = ?
            ''', (realized_r, realized_pnl, _iso(_now_utc()), decision_id))

    def load_decisions(self, user_id: Optional[int] = None, symbol: Optional[str] = None,
                       since=None, only_closed: bool = False, limit: int = 50_000):
        clauses, params = ["1=1"], []
        if user_id is not None:
            clauses.append("user_id = ?"); params.append(user_id)
        if symbol is not None:
            clauses.append("symbol = ?"); params.append(symbol)
        if since is not None:
            clauses.append("as_of >= ?"); params.append(_iso(since))
        if only_closed:
            clauses.append("closed_at IS NOT NULL")
        with self._lock, self._conn() as conn:
            rows = conn.execute(f'''
                SELECT symbol, as_of, action, horizon_sec, expected_return, expected_adverse,
                       prob_positive, costs, regime, regime_confidence, historical_sample,
                       expectancy_r, quality, size_fraction, reasons_json, blockers_json,
                       realized_r, realized_pnl, closed_at
                FROM decisions WHERE {" AND ".join(clauses)}
                ORDER BY as_of ASC LIMIT ?
            ''', params + [limit]).fetchall()
        cols = ['symbol', 'as_of', 'action', 'horizon_sec', 'expected_return', 'expected_adverse',
                'prob_positive', 'costs', 'regime', 'regime_confidence', 'historical_sample',
                'expectancy_r', 'quality', 'size_fraction', 'reasons_json', 'blockers_json',
                'realized_r', 'realized_pnl', 'closed_at']
        df = pd.DataFrame(rows, columns=cols)
        if not df.empty:
            df['as_of'] = pd.to_datetime(df['as_of'], utc=True)
        return df

    def load_decisions_with_features(self, user_id: Optional[int] = None, symbol: Optional[str] = None,
                                     since=None, only_closed: bool = True,
                                     limit: int = 50_000) -> pd.DataFrame:
        """
        Decisions joined to the exact observation that produced them (matched on
        user_id, symbol, as_of — the reason `decision.as_of` is set to the market
        state's timestamp rather than wall-clock time, see engine.decide()).
        This is what the post-mortem/attribution report reads: it needs the full
        feature vector behind each trade, not just the decision's own summary
        numbers, in order to ask "which CONDITIONS actually carried the P&L".
        """
        clauses, params = ["d.user_id = o.user_id", "d.symbol = o.symbol", "d.as_of = o.as_of"], []
        if user_id is not None:
            clauses.append("d.user_id = ?"); params.append(user_id)
        if symbol is not None:
            clauses.append("d.symbol = ?"); params.append(symbol)
        if since is not None:
            clauses.append("d.as_of >= ?"); params.append(_iso(since))
        if only_closed:
            clauses.append("d.closed_at IS NOT NULL")
        with self._lock, self._conn() as conn:
            rows = conn.execute(f'''
                SELECT d.symbol, d.as_of, d.action, d.horizon_sec, d.expected_return,
                       d.expected_adverse, d.prob_positive, d.costs, d.regime,
                       d.regime_confidence, d.historical_sample, d.expectancy_r, d.quality,
                       d.size_fraction, d.realized_r, d.realized_pnl, d.closed_at,
                       o.features_json
                FROM decisions d JOIN observations o ON {" AND ".join(clauses)}
                ORDER BY d.as_of ASC LIMIT ?
            ''', params + [limit]).fetchall()

        records = []
        for (symbol_, as_of, action, horizon_sec, expected_return, expected_adverse, prob_positive,
             costs, regime, regime_conf, hist_sample, expectancy_r, quality, size_fraction,
             realized_r, realized_pnl, closed_at, feat_json) in rows:
            rec = {'symbol': symbol_, 'as_of': as_of, 'action': action, 'horizon_sec': horizon_sec,
                  'expected_return': expected_return, 'expected_adverse': expected_adverse,
                  'prob_positive': prob_positive, 'costs': costs, 'regime': regime,
                  'regime_confidence': regime_conf, 'historical_sample': hist_sample,
                  'expectancy_r': expectancy_r, 'quality': quality, 'size_fraction': size_fraction,
                  'realized_r': realized_r, 'realized_pnl': realized_pnl, 'closed_at': closed_at}
            rec.update(json.loads(feat_json or '{}'))
            records.append(rec)
        df = pd.DataFrame.from_records(records)
        if not df.empty:
            df['as_of'] = pd.to_datetime(df['as_of'], utc=True)
        return df


def _clean_scalar(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _clean(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _clean_scalar(v) for k, v in d.items()}
