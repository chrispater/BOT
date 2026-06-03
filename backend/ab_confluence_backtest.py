#!/usr/bin/env python3
"""
A/B backtest for the GRONKAI signal-confluence layer.

Question it answers: does the 1.25x size boost on "reliable-signal-agrees" setups
actually improve risk-adjusted returns, or is it just adding variance?

It runs the SAME model on the SAME data twice — confluence ON vs OFF — and also
buckets every trade as confluent vs non-confluent to measure whether confluent
trades genuinely win more. If confluent trades don't beat non-confluent trades,
the boost is pure added variance and the layer should be rolled back.

Run on PythonAnywhere (where ccxt + the ML stack + exchange access exist):
    cd ~/cryptoscanner/CryptoQuantScanner/backend
    python3 ab_confluence_backtest.py --days 45 --coins BTC/USDT:USDT,ETH/USDT:USDT
    # or just: python3 ab_confluence_backtest.py    (uses user 1's live settings)
"""
import argparse
import math
import numpy as np
import pandas as pd

from trading_service import (
    TradingService, _decode_y, _encode_y,
    LGBM_AVAILABLE, XGB_AVAILABLE, CONFLUENCE_BOOST,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_OK = True
except Exception:
    SMOTE_OK = False

try:
    from signal_reliability import SignalReliability
except Exception:
    SignalReliability = None

try:
    from trading_service import SVM_PARAMS
except Exception:
    SVM_PARAMS = {'C': [1, 10], 'gamma': ['scale']}

TAKER_FEE = 0.0006
SLIP = 0.0005


def _load_settings(user_id):
    """Best-effort: use the user's persisted live settings so the A/B reflects the
    real config. Falls back to bot defaults if the DB isn't reachable."""
    try:
        import database
        s = database.get_user_settings(user_id)
        if s:
            return s
    except Exception as e:
        print(f"(could not load user settings: {e}; using defaults)")
    return {}


def _build_reliable_map(bot, test_df, s):
    """Replicate the optimizer's confluence map exactly: candle index -> directions
    backed by a RELIABLE signal under this config. Fail-open if too sparse."""
    if SignalReliability is None:
        return None
    try:
        minutes = bot._get_timeframe_minutes()
        horizon = max(12, int(120 / max(1, minutes)))
        sr = SignalReliability(
            leverage=bot.leverage, stop_loss_pct=bot.stop_loss_pct,
            take_profit_pct=bot.take_profit_pct, horizon=horizon,
            min_winrate=bot.reliability_min_winrate, min_samples=4,
            params=s.get('reliability_params'),
        )
        card = sr.score(test_df)
        detected = sr._detect(test_df)
        m = {}
        for name, info in detected.items():
            meta = card.get(name)
            if meta and meta['reliable']:
                for idx in info['idx'].tolist():
                    m.setdefault(idx, set()).add(info['dir'])
        if sum(len(v) for v in m.values()) >= 8:
            return m
    except Exception as e:
        print(f"  (reliability map failed: {e})")
    return None


def _run(bot, model, imp, sc, test_df, X_test, is_tree, reliable_map, use_confluence, start_balance):
    """One pass of the backtest. Returns (trades, final_balance, max_dd).
    Each trade dict carries 'confluent' so we can bucket afterwards. Mirrors the
    live/optimizer entry+exit logic (EV gate, conf-scaled margin, SL/TP/trailing)."""
    balance = start_balance
    peak = start_balance
    max_dd = 0.0
    position = None
    entry_price = 0
    hwm = lwm = 0
    trades = []
    last_trade_candle = -999
    minutes = bot._get_timeframe_minutes()
    cooldown_candles = max(1, math.ceil(bot.trade_cooldown / 60 / minutes))

    lev = max(1, bot.leverage)
    fee_m = 2 * TAKER_FEE * lev
    net_win = max(0.001, bot.take_profit_pct - fee_m)
    net_loss = bot.stop_loss_pct + fee_m

    for i in range(len(test_df)):
        try:
            row = test_df.iloc[i]
            price = row['close']
            X_sc = sc.transform(imp.transform(X_test.iloc[[i]]))
            raw = model.predict(X_sc)[0]
            sig_val = int(_decode_y([raw])[0]) if is_tree else int(raw)
            conf = float(max(model.predict_proba(X_sc)[0]))

            if i - last_trade_candle < cooldown_candles:
                continue

            adx = row.get('adx', 25); adx = 25.0 if pd.isna(adx) else float(adx)
            vol = row.get('volume_ratio', 1.0); vol = 1.0 if pd.isna(vol) else float(vol)
            confluent = reliable_map is not None and sig_val in reliable_map.get(i, ())
            boost = CONFLUENCE_BOOST if (use_confluence and confluent) else 1.0
            ev = conf * net_win - (1 - conf) * net_loss

            if position is None and sig_val != 0 and conf >= bot.min_confidence \
                    and adx >= bot.adx_threshold and vol >= 0.65 and ev > 0:
                conf_rng = max(0.01, 1.0 - bot.min_confidence)
                c_scale = max(0.5, min(1.5, 0.5 + (conf - bot.min_confidence) / conf_rng))
                risk_ceil = min(max(bot.risk_per_trade * 2.0, 0.10), 0.75)
                margin = min(balance * bot.risk_per_trade * c_scale * boost, balance * risk_ceil)
                if margin <= 0 or margin > balance * 0.95:
                    continue
                notional = margin * lev
                position = {
                    'side': 'long' if sig_val == 1 else 'short',
                    'size': notional / price, 'entry_fee': notional * (TAKER_FEE + SLIP),
                    'confluent': confluent,
                }
                entry_price = price
                hwm = lwm = price
                last_trade_candle = i

            elif position is not None:
                if position['side'] == 'long':
                    hwm = max(hwm, price)
                else:
                    lwm = min(lwm, price)
                ppct = ((price - entry_price) / entry_price if position['side'] == 'long'
                        else (entry_price - price) / entry_price)
                mpct = ppct * lev
                trail = bot.trailing_stop_pct / lev
                exit_now = False
                if mpct <= -bot.stop_loss_pct:
                    exit_now = True
                elif mpct >= bot.take_profit_pct:
                    exit_now = True
                else:
                    if ppct > 0:
                        if position['side'] == 'long' and price <= hwm * (1 - trail):
                            exit_now = True
                        elif position['side'] == 'short' and price >= lwm * (1 + trail):
                            exit_now = True
                    if not exit_now and sig_val != 0 and conf >= bot.min_confidence * 0.8:
                        if (sig_val == -1 and position['side'] == 'long') or \
                           (sig_val == 1 and position['side'] == 'short'):
                            exit_now = True
                if exit_now:
                    change = ((price - entry_price) if position['side'] == 'long'
                              else (entry_price - price))
                    exit_fee = position['size'] * price * (TAKER_FEE + SLIP)
                    net = change * position['size'] - position['entry_fee'] - exit_fee
                    balance += net
                    peak = max(peak, balance)
                    dd = (peak - balance) / peak * 100 if peak > 0 else 0
                    max_dd = max(max_dd, dd)
                    trades.append({'pnl': net, 'confluent': position['confluent']})
                    position = None
                    last_trade_candle = i
        except Exception:
            continue
    return trades, balance, max_dd


def _stats(trades, final_balance, start_balance, max_dd):
    n = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    pnl = sum(t['pnl'] for t in trades)
    ret = (final_balance - start_balance) / start_balance * 100
    wr = wins / n * 100 if n else 0
    sharpe = 0.0
    if n > 1:
        r = [t['pnl'] / start_balance for t in trades]
        sharpe = float(np.mean(r) / (np.std(r) + 1e-10) * np.sqrt(n))
    return {'trades': n, 'win_rate': wr, 'return': ret, 'pnl': pnl,
            'max_dd': max_dd, 'sharpe': sharpe}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', type=int, default=1)
    ap.add_argument('--days', type=int, default=45)
    ap.add_argument('--coins', type=str, default=None, help='comma-separated')
    ap.add_argument('--timeframe', type=str, default=None)
    args = ap.parse_args()

    s = _load_settings(args.user)
    coins = (args.coins.split(',') if args.coins
             else s.get('selected_coins') or ['BTC/USDT:USDT', 'ETH/USDT:USDT'])
    timeframe = args.timeframe or s.get('timeframe', '5m')
    start_balance = float(s.get('starting_balance', 10000))
    per_coin = start_balance / len(coins)

    bot = TradingService(
        user_id=args.user, starting_balance=start_balance,
        selected_coins=coins, timeframe=timeframe,
        leverage=int(s.get('leverage', 10)),
        risk_per_trade=float(s.get('risk_per_trade', 0.02)),
        stop_loss_pct=float(s.get('stop_loss_pct', 0.15)),
        take_profit_pct=float(s.get('take_profit_pct', 0.30)),
        trade_cooldown=int(s.get('trade_cooldown', 300)),
        min_confidence=float(s.get('min_confidence', 0.65)),
        trailing_stop_pct=float(s.get('trailing_stop_pct', 0.10)),
        adx_threshold=int(s.get('adx_threshold', 18)),
        reliability_min_winrate=float(s.get('reliability_min_winrate', 0.60)),
        reliability_params=s.get('reliability_params'),
    )

    print("=" * 70)
    print("GRONKAI CONFLUENCE A/B BACKTEST")
    print(f"coins={coins} tf={timeframe} days={args.days} "
          f"lev={bot.leverage}x SL={bot.stop_loss_pct:.0%} TP={bot.take_profit_pct:.0%} "
          f"relwr={bot.reliability_min_winrate:.0%}")
    print(f"model={'LGBM' if LGBM_AVAILABLE else 'XGB' if XGB_AVAILABLE else 'SVM'}")
    print("=" * 70)

    on_trades, off_trades = [], []
    on_bal = off_bal = 0.0
    on_dd = off_dd = 0.0
    minutes = bot._get_timeframe_minutes()

    for symbol in coins:
        print(f"\n[{symbol}] fetching {args.days}d of {timeframe}...")
        periods = int(args.days * 24 * 60 / minutes)
        df = bot.fetch_ohlcv(symbol=symbol, limit=periods)
        if df is None or len(df) < 200:
            print(f"  not enough data ({0 if df is None else len(df)} rows) — skipping")
            on_bal += per_coin; off_bal += per_coin
            continue
        df = bot.calculate_indicators(df)
        df = bot.create_labels(df)
        df = df.dropna()
        half = len(df) // 2
        train_df, test_df = df.iloc[:half], df.iloc[half:]

        X_train, _ = bot.prepare_features(train_df)
        y_train = train_df['signal']
        mask = y_train != 0
        X_train, y_train = X_train[mask], y_train[mask]
        if len(X_train) < 30:
            print("  too few training signals — skipping")
            on_bal += per_coin; off_bal += per_coin
            continue

        imp = SimpleImputer(strategy='mean'); sc = StandardScaler()
        X_sc = sc.fit_transform(imp.fit_transform(X_train))
        if SMOTE_OK:
            try:
                k = min(3, len(y_train[y_train == 1]) - 1, len(y_train[y_train == -1]) - 1)
                X_res, y_res = SMOTE(random_state=42, k_neighbors=max(1, k)).fit_resample(X_sc, y_train)
            except Exception:
                X_res, y_res = X_sc, y_train
        else:
            X_res, y_res = X_sc, y_train

        is_tree = LGBM_AVAILABLE or XGB_AVAILABLE
        model = bot._build_model()
        if is_tree:
            model.fit(X_res, _encode_y(y_res))
        else:
            grid = GridSearchCV(model, SVM_PARAMS, cv=TimeSeriesSplit(n_splits=3),
                                scoring='f1_weighted', n_jobs=1)
            grid.fit(X_res, y_res)
            model = grid.best_estimator_

        X_test, _ = bot.prepare_features(test_df)
        rmap = _build_reliable_map(bot, test_df, s)
        rmap_pts = 0 if rmap is None else sum(len(v) for v in rmap.values())
        print(f"  trained. reliable-signal candle-hits in test window: {rmap_pts}"
              + ("  (confluence INACTIVE — too sparse, fail-open)" if rmap is None else ""))

        t_on, b_on, d_on = _run(bot, model, imp, sc, test_df, X_test, is_tree, rmap, True, per_coin)
        t_off, b_off, d_off = _run(bot, model, imp, sc, test_df, X_test, is_tree, rmap, False, per_coin)
        on_trades += t_on; off_trades += t_off
        on_bal += b_on; off_bal += b_off
        on_dd = max(on_dd, d_on); off_dd = max(off_dd, d_off)

    on = _stats(on_trades, on_bal, start_balance, on_dd)
    off = _stats(off_trades, off_bal, start_balance, off_dd)

    def line(label, a):
        print(f"  {label:18} ret {a['return']:+7.2f}%  win {a['win_rate']:5.1f}%  "
              f"trades {a['trades']:4d}  maxDD {a['max_dd']:5.1f}%  sharpe {a['sharpe']:5.2f}")

    print("\n" + "=" * 70)
    print("AGGREGATE A/B  (same model & data, only the 1.25x boost differs)")
    print("=" * 70)
    line("Confluence ON", on)
    line("Confluence OFF", off)

    # Bucket analysis — the decisive metric. Uses the ON run's tagged trades.
    conf_tr = [t for t in on_trades if t['confluent']]
    nonf_tr = [t for t in on_trades if not t['confluent']]
    def bwr(tr):
        return (sum(1 for t in tr if t['pnl'] > 0) / len(tr) * 100) if tr else 0.0
    def bavg(tr):
        return (sum(t['pnl'] for t in tr) / len(tr)) if tr else 0.0

    print("\n" + "=" * 70)
    print("BUCKET ANALYSIS  (are 'reliable-signal' trades actually better?)")
    print("=" * 70)
    print(f"  Confluent trades     n={len(conf_tr):4d}  win {bwr(conf_tr):5.1f}%  avg PnL ${bavg(conf_tr):+8.2f}")
    print(f"  Non-confluent trades n={len(nonf_tr):4d}  win {bwr(nonf_tr):5.1f}%  avg PnL ${bavg(nonf_tr):+8.2f}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if len(conf_tr) < 10:
        print("  ⚠ Too few confluent trades to judge — the confluence layer is")
        print("    barely firing on this data, so it's near-inert (just overhead).")
        print("    Recommendation: ROLL BACK — it adds complexity for ~no effect.")
    else:
        edge = bwr(conf_tr) - bwr(nonf_tr)
        avg_edge = bavg(conf_tr) - bavg(nonf_tr)
        better_ret = on['return'] - off['return']
        better_sharpe = on['sharpe'] - off['sharpe']
        print(f"  Confluent win-rate edge: {edge:+.1f} pts | avg-PnL edge: ${avg_edge:+.2f}")
        print(f"  ON vs OFF: return {better_ret:+.2f} pts | sharpe {better_sharpe:+.2f} | "
              f"maxDD {on['max_dd'] - off['max_dd']:+.1f} pts")
        if edge > 3 and avg_edge > 0 and better_sharpe >= -0.05:
            print("  ✓ KEEP — confluent trades genuinely win more; the boost earns its")
            print("    keep by concentrating size into better setups.")
        elif edge < 1 or avg_edge <= 0:
            print("  ✗ ROLL BACK — confluent trades are NOT better than the rest, so the")
            print("    1.25x boost is pure added variance (worse risk-adjusted returns).")
        else:
            print("  ~ MARGINAL — small/noisy edge. Given the added complexity, leaning")
            print("    ROLL BACK unless it clearly helps sharpe across more coins/days.")
    print("=" * 70)


if __name__ == '__main__':
    main()
