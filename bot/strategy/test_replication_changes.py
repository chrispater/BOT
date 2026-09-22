"""
Tests for the three replication changes (owner instruction 2026-09-22:
"implement all three, dormant and reversible with tests").

  1. The expectancy entry block expires after a cooldown instead of latching.
  2. The ML validation gate charges round-trip cost, and entries require it.
  3. The entry universe expands to 65 symbols behind a config flag.

Each change is pinned twice: that it does what it should when enabled, and
that its config switch restores the pre-change behaviour exactly. The
evidence behind each is in bot/research/REPORT.md.

    python -m bot.strategy.test_replication_changes
"""

import json

import numpy as np
import pandas as pd

from . import engine
from .ml_model import validation_verdict

CFG = json.load(open('bot/config.json'))


def _trades(pnls, last_date='2026-09-22'):
    """Closed-trade records, the last one dated `last_date`."""
    out = [{'event': 'sell', 'pnl_pct': p} for p in pnls]
    if out and last_date:
        out[-1]['ts'] = f'{last_date}T15:00:00Z'
    return out


# ── 1. Expectancy block: cooldown, not a latch ──────────────────────────────

def test_config_ships_with_cooldown():
    assert CFG['strategy']['expectancy_block_expiry_days'] == 3
    print('ok  config ships with a 3-day expectancy cooldown')


def test_block_holds_during_cooldown():
    losing = _trades([-0.5] * 20, '2026-09-21')
    why = engine.expectancy_block(losing, '2026-09-22', 3)
    assert why and 'cooldown, 1/3d' in why, why
    print('ok  a losing record blocks entries inside the cooldown')


def test_block_expires_after_cooldown():
    losing = _trades([-0.5] * 20, '2026-09-18')
    assert engine.expectancy_block(losing, '2026-09-22', 3) is None
    print('ok  the block lifts once the cooldown since the last close has passed')


def test_latch_is_what_expiry_fixes():
    """Without an expiry the same losing record blocks forever — the bug."""
    losing = _trades([-0.5] * 20, '2026-01-01')
    assert engine.expectancy_block(losing, '2026-09-22', None), 'legacy must latch'
    assert engine.expectancy_block(losing, '2026-09-22', 3) is None
    print('ok  expiry_days=None restores the original permanent latch')


def test_positive_or_short_history_never_blocks():
    assert engine.expectancy_block(_trades([0.4] * 20), '2026-09-22', 3) is None
    assert engine.expectancy_block(_trades([-1.0] * 19), '2026-09-22', 3) is None
    print('ok  positive expectancy, or fewer than 20 closes, never blocks')


def test_undated_history_does_not_latch():
    """A record with no datable close cannot anchor a cooldown."""
    losing = _trades([-0.5] * 20, last_date=None)
    assert engine.expectancy_block(losing, '2026-09-22', 3) is None
    print('ok  undated history is treated as expired, not as a permanent block')


def test_uses_latest_dated_close():
    """Old undated lines before a recent dated one must not hide the recent close."""
    t = _trades([-0.5] * 20, '2026-09-22')
    t.insert(0, {'event': 'sell', 'pnl_pct': -0.5})
    assert engine.expectancy_block(t, '2026-09-22', 3)
    print('ok  the cooldown anchors on the most recent dated close')


def test_live_history_is_not_blocked_today():
    """Sanity: the change must not lock the live bot out on deployment."""
    hist = [json.loads(l) for l in open('bot/trade_log.jsonl') if l.strip()]
    closed = [t for t in hist if t.get('event') == 'sell' and t.get('pnl_pct') is not None]
    assert engine.expectancy_block(closed, '2026-09-22', 3) is None
    print(f'ok  live history ({len(closed)} closes) does not trip the block today')


# ── 2. Cost-aware validation and the symbol-level entry gate ────────────────

def test_config_ships_with_cost_gate():
    s = CFG['strategy']
    assert s['validation_cost_pct'] == 0.2
    assert s['validation_cost_pct'] == 2 * s['slippage_pct_per_side'], \
        'the gate should charge exactly the modelled round trip'
    assert s['entry_requires_validation'] is True
    print('ok  config charges 0.2% round-trip cost and requires validation to enter')


def test_verdict_charges_cost():
    edge = [0.001] * 60                     # +0.10% per bar: positive, below cost
    ok_gross, _ = validation_verdict(edge, 4, 0.0)
    ok_net, why = validation_verdict(edge, 4, 0.002)
    assert ok_gross and not ok_net, (ok_gross, ok_net)
    assert 'does not clear cost' in why, why
    assert validation_verdict([0.003] * 60, 4, 0.002)[0], 'an edge above cost must pass'
    print('ok  a positive edge smaller than cost fails; one larger than cost passes')


def test_verdict_zero_cost_is_legacy():
    for data in ([0.0005] * 60, [-0.0005] * 60, [0.0] * 60):
        exp = float(np.mean(data))
        assert validation_verdict(data, 4, 0.0)[0] == (exp > 0), data[0]
    assert not validation_verdict([0.01] * 39, 4, 0.0)[0], 'min pooled bars still applies'
    assert not validation_verdict([0.01] * 60, 1, 0.0)[0], 'min folds still applies'
    print('ok  validation_cost_pct=0 restores the original gross gate exactly')


class _FakeML:
    """Stands in for MLStream so a test controls the validation verdict."""
    VALIDATED = {}
    SEEN_MIN = []

    def __init__(self, *a, **k):
        self.min_expectancy = k.get('min_expectancy', 0.0)
        _FakeML.SEEN_MIN.append(self.min_expectancy)
        self.trained, self.validated, self.validation_summary = True, False, 'fake'

    def train(self, df):
        self.validated = _FakeML.VALIDATED.get(self._sym, False)
        return True

    def predict(self, df):
        return (1, 0.9) if self.validated else (0, 0.5)


def _bars(close, n=90):
    idx = pd.date_range('2026-07-01', periods=n, freq='h', tz='UTC')
    c = np.linspace(close * 0.98, close, n)
    return pd.DataFrame({'open': c, 'high': c * 1.001, 'low': c * 0.999, 'close': c,
                         'volume': np.full(n, 1e6)}, index=idx)


def _entry_run(cfg_patch, validated, universe_syms=('AAPL', 'PLTR'), held=()):
    """Run the engine with every candidate forced to a buy, returning entries."""
    cfg = json.loads(json.dumps(CFG))
    for path, v in cfg_patch.items():
        d = cfg
        *head, last = path.split('.')
        for h in head:
            d = d[h]
        d[last] = v
    snap = {
        'today_et': '2026-09-22', 'config': cfg,
        'state': {'date_et': '2026-09-22', 'start_of_day_equity': 1440.0,
                  'peak_equity': 1440.0, 'positions': {}},
        'portfolio': {'equity': 1440.0, 'buying_power': 5000.0},
        'positions': [{'symbol': s, 'quantity': '1', 'shares_available_for_sells': '1',
                       'average_buy_price': '100'} for s in held],
        'trade_history': [], 'regime_bars': _bars(760, 60),
        'bars': {s: _bars(100.0) for s in universe_syms},
    }
    real = (engine.MLStream, engine.ensemble, engine.entry_filter, engine.entry_ev,
            engine.best_setup)
    _FakeML.VALIDATED = validated
    order = list(universe_syms)

    class Seq(_FakeML):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._sym = order.pop(0) if order else None

    engine.MLStream = Seq
    # Force a high-confidence setup buy whatever the model says, so the only
    # thing standing between a candidate and an entry is the gate under test.
    engine.ensemble = lambda *a, **k: (0, 0.5)
    engine.best_setup = lambda *a, **k: (1, 0.9, 'breakout_long')
    engine.entry_filter = lambda *a, **k: (True, '')
    engine.entry_ev = lambda *a, **k: 0.05
    try:
        out = engine.run(snap)
    finally:
        (engine.MLStream, engine.ensemble, engine.entry_filter, engine.entry_ev,
         engine.best_setup) = real
    return {e['symbol'] for e in out['entries']}, out


def test_gate_blocks_setup_entry_on_unvalidated_name():
    """The ETHA/MSTR case: a setup fired on a model that failed validation."""
    got, out = _entry_run({}, {'AAPL': False, 'PLTR': True})
    assert got == {'PLTR'}, got
    assert out['diagnostics']['symbols']['AAPL']['entry_blocked'] == 'model not validated net of cost'
    print('ok  a setup cannot open a position in a name whose model failed validation')


def test_gate_off_restores_setup_entries():
    got, _ = _entry_run({'strategy.entry_requires_validation': False},
                        {'AAPL': False, 'PLTR': True})
    assert got == {'AAPL', 'PLTR'}, got
    print('ok  entry_requires_validation=false restores setup entries on any name')


def test_engine_passes_cost_to_model():
    """The engine itself must hand every model the configured cost."""
    _FakeML.SEEN_MIN = []
    _entry_run({}, {'AAPL': True, 'PLTR': True})
    assert _FakeML.SEEN_MIN and all(abs(m - 0.002) < 1e-12 for m in _FakeML.SEEN_MIN), _FakeML.SEEN_MIN
    _FakeML.SEEN_MIN = []
    _entry_run({'strategy.validation_cost_pct': 0}, {'AAPL': True, 'PLTR': True})
    assert all(m == 0 for m in _FakeML.SEEN_MIN), _FakeML.SEEN_MIN
    print('ok  the engine hands each model 0.002 minimum expectancy; 0 when switched off')


# ── 3. Universe expansion behind a flag ─────────────────────────────────────

def test_config_ships_with_65_symbols():
    uni = engine.active_universe(CFG)
    assert len(CFG['universe']) == 33 and len(CFG['universe_expansion']) == 32
    assert CFG['universe_expansion_enabled'] is True
    assert len(uni) == 65, len(uni)
    assert not set(CFG['universe']) & set(CFG['universe_expansion']), 'lists must not overlap'
    print('ok  active universe is 65 symbols: 33 core + 32 expansion')


def test_expansion_flag_off_restores_core():
    cfg = dict(CFG, universe_expansion_enabled=False)
    assert engine.active_universe(cfg) == set(CFG['universe'])
    print('ok  universe_expansion_enabled=false restores the 33-symbol core')


def test_entries_limited_to_active_universe():
    both = {'AAPL': True, 'PLTR': True}
    got_on, _ = _entry_run({}, both)
    got_off, out = _entry_run({'universe_expansion_enabled': False}, both)
    assert got_on == {'AAPL', 'PLTR'}, got_on
    assert got_off == {'AAPL'}, got_off
    assert out['diagnostics']['symbols']['PLTR']['entry_blocked'] == 'not in active universe'
    print('ok  expansion names are buyable only while the flag is on')


def test_held_outside_universe_still_managed():
    """A held position must keep exit management even if it leaves the universe."""
    _, out = _entry_run({'universe_expansion_enabled': False},
                        {'AAPL': True, 'PLTR': True}, held=('PLTR',))
    diag = out['diagnostics']['symbols']['PLTR']
    assert 'exit_price' in diag, diag
    print('ok  a held name outside the active universe is still exit-managed')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print('\nall replication-change tests passed')
