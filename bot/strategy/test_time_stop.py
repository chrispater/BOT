"""
Regression tests for retiring the time stop (owner instruction 2026-09-22).

The 14-hour time stop was removed after it proved a net drag: across 20 closed
trades it averaged -0.45% and cost -9.06% in total, while signal_reversal
earned +0.54% over 26. It was cutting positions that had not worked yet rather
than limiting losses.

These pin three things: that a null time_stop_hours never fires, that every
other exit still does, and that cycles_held keeps counting (min_hold_hours and
the reversal hysteresis both read it).

    python -m bot.strategy.test_time_stop
"""

import json

from . import engine

CFG = json.load(open('bot/config.json'))
COST = 100.0


def _cfg(**over):
    cfg = dict(CFG['strategy'], **CFG['risk'])
    cfg.update(over)
    return cfg


def _meta(**over):
    m = {'entry_date_et': '2026-09-21', 'high_water_mark': COST,
         'be_armed': False, 'cycles_held': 0, 'reversal_streak': 0}
    m.update(over)
    return m


def _exit(meta, price, cfg, signal=0, confidence=0.5, held_today=False):
    return engine.exit_decision(meta, COST, price, signal, confidence,
                                cfg, held_today)


def test_config_has_time_stop_disabled():
    """The shipped config must keep the time stop off."""
    assert CFG['strategy']['time_stop_hours'] is None, \
        f"expected null, got {CFG['strategy']['time_stop_hours']!r}"
    print('ok  config ships with time_stop_hours = null')


def test_null_never_fires_however_long_held():
    """A flat position held far past the old 14-cycle threshold must not exit."""
    for held in (13, 14, 15, 100, 5000):
        ok, reason, _ = _exit(_meta(cycles_held=held), COST * 1.001, _cfg())
        assert not ok and reason != 'time_stop', \
            f'held {held} cycles produced {reason!r}'
    print('ok  a null time stop never fires, even at 5000 cycles held')


def test_zero_and_negative_also_disable():
    """Non-positive values disable rather than firing on every cycle."""
    for value in (0, -1):
        ok, reason, _ = _exit(_meta(cycles_held=50), COST,
                              _cfg(time_stop_hours=value))
        assert reason != 'time_stop', f'{value} should disable, got {reason!r}'
    print('ok  zero and negative disable instead of firing immediately')


def test_still_fires_when_configured():
    """The branch is dormant, not deleted — a real value must still work.

    This is what makes the change reversible: setting a number in config is
    enough to bring the time stop back, with no code edit.
    """
    ok, reason, _ = _exit(_meta(cycles_held=13), COST, _cfg(time_stop_hours=14))
    assert ok and reason == 'time_stop', f'expected time_stop, got {reason!r}'
    print('ok  setting a positive value restores the old behaviour')


def test_other_exits_unaffected():
    """Removing the time stop must not blunt any other exit."""
    cases = [
        ('stop_loss',       COST * 0.94, _meta(), {}),
        ('take_profit',     COST * 1.11, _meta(), {}),
        ('breakeven_floor', COST * 1.001, _meta(be_armed=True), {}),
        ('trailing_stop',   COST * 1.02,
         _meta(high_water_mark=COST * 1.10), {}),
    ]
    for expected, price, meta, kw in cases:
        ok, reason, _ = _exit(meta, price, _cfg(), **kw)
        assert ok and reason == expected, \
            f'expected {expected} at {price}, got {reason!r}'
    print('ok  stop_loss, take_profit, breakeven_floor and trailing_stop all fire')


def test_signal_reversal_still_confirms():
    """The earner must survive: two bearish cycles past min_hold still exit."""
    cfg = _cfg()
    meta = _meta(cycles_held=cfg['min_hold_hours'] + 5)
    ok, reason, meta = _exit(meta, COST * 1.01, cfg, signal=-1, confidence=0.9)
    assert not ok and meta['reversal_streak'] == 1, 'first cycle must only arm'
    ok, reason, meta = _exit(meta, COST * 1.01, cfg, signal=-1, confidence=0.9)
    assert ok and reason == 'signal_reversal', f'got {reason!r}'
    print('ok  signal_reversal still needs two cycles and still fires')


def test_cycles_held_still_increments():
    """cycles_held must keep counting — min_hold_hours depends on it."""
    meta = _meta()
    for expected in (1, 2, 3):
        _, _, meta = _exit(meta, COST * 1.001, _cfg())
        assert meta['cycles_held'] == expected, meta['cycles_held']
    print('ok  cycles_held keeps incrementing with the time stop off')


def test_reversal_respects_min_hold_with_time_stop_off():
    """A young position still cannot reverse out before min_hold_hours."""
    cfg = _cfg()
    meta = _meta(cycles_held=0)
    for _ in range(cfg['min_hold_hours'] - 1):
        ok, _, meta = _exit(meta, COST * 1.01, cfg, signal=-1, confidence=0.9)
        assert not ok, 'must not exit before min_hold_hours'
    assert meta['reversal_streak'] == 0, \
        'streak must stay 0 while under min_hold, not bank confirmations'
    print('ok  min_hold_hours still gates reversals, streak does not pre-bank')


def test_same_day_still_deferred():
    """The cash-account GFV guard is untouched: only stop_loss exits today."""
    ok, reason, _ = _exit(_meta(be_armed=True), COST * 1.001, _cfg(),
                          held_today=True)
    assert reason == 'breakeven_floor' and not ok, \
        f'same-day non-stop exit must defer; got ok={ok} {reason!r}'
    ok, reason, _ = _exit(_meta(), COST * 0.94, _cfg(), held_today=True)
    assert ok and reason == 'stop_loss', 'stop_loss must still fire same-day'
    print('ok  same-day GFV deferral unchanged; stop_loss still exempt')


if __name__ == '__main__':
    test_config_has_time_stop_disabled()
    test_null_never_fires_however_long_held()
    test_zero_and_negative_also_disable()
    test_still_fires_when_configured()
    test_other_exits_unaffected()
    test_signal_reversal_still_confirms()
    test_cycles_held_still_increments()
    test_reversal_respects_min_hold_with_time_stop_off()
    test_same_day_still_deferred()
    print('\nall time-stop retirement tests passed')
