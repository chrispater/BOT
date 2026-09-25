"""Round-2 lead-lag DISCOVERY (1h / 4h / 1d, lags 1-3). Discovery data only.

Run:   OMP_NUM_THREADS=1 python3 discover.py
Reads: ../discovery_data/*.pkl  (nothing else)
Writes (this dir): all_tests.csv, per_pair_tf.csv, run_meta.json

Every implementation choice is fixed in discovery.md part A (written before any result).
Short version:
  test = (family, tf, lead, target, L, H, direction, threshold)
  s    = lead's return over the last L bars ending at the latest lead mark <= decision time tau
  trade: buy target at next-bar open (+5m for equities; next session open / next UTC day open
         for daily), exit at close of the H-th bar; non-overlapping (one position per rule)
  net  = gross - cost (0.20% equity target, 0.40% crypto target)
  p    = max(per-trade one-sided t p, date-clustered p); BH over ALL tests.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json, pickle, time, datetime as dt
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "discovery_data")
ET = "America/New_York"
H_NS = 3600 * 10**9
M_NS = 60 * 10**9
D_NS = 86400 * 10**9
EQ_COST, CR_COST = 0.0020, 0.0040
SPLIT = {"daily": dt.date(2023, 1, 1), "crypto_intra": dt.date(2025, 8, 1), "eq_intra": dt.date(2026, 4, 27)}
LH = [(1, 1), (2, 1), (2, 2), (3, 1), (3, 3)]
TFS = ["1h", "4h", "1d"]

CRYPTO = ["BTC", "ETH", "SOL", "XRP", "DOGE", "LTC", "BCH", "LINK", "AVAX", "ADA", "DOT", "UNI", "XLM", "SHIB", "ATOM"]
C = lambda s: ("c", s)   # crypto symbol
E = lambda s: ("e", s)   # equity symbol

def lab(x):
    return f"{x[1]}-USD" if x[0] == "c" else x[1]

# ------------------------------------------------------------------ load
t0 = time.time()
eq1d = pickle.load(open(os.path.join(DATA, "equity_1d.pkl"), "rb"))
cr1d = pickle.load(open(os.path.join(DATA, "crypto_1d.pkl"), "rb"))
cr1h = pickle.load(open(os.path.join(DATA, "crypto_1h.pkl"), "rb"))
m5 = pickle.load(open(os.path.join(DATA, "equity_5min.pkl"), "rb"))

def ns(idx):
    return pd.DatetimeIndex(idx).as_unit("ns").asi8

# crypto hourly on a full UTC grid
HG = pd.date_range("2024-09-01 00:00", "2026-06-30 23:00", freq="h", tz="UTC")
HG0 = HG[0].value
NH = len(HG)
CH = {}
for s, d in cr1h.items():
    d = d.copy(); d.index = d.index.as_unit("ns")
    r = d.reindex(HG)
    CH[s] = (r.open.to_numpy(float), r.close.to_numpy(float))

# crypto daily on a full UTC calendar (NaN over gaps, e.g. XRP 2021-01..2023-07)
DG = pd.date_range("2021-01-01", "2024-12-31", freq="D", tz="UTC")
DG0 = DG[0].value
ND = len(DG)
CD = {}
for s, d in cr1d.items():
    d = d.copy(); d.index = d.index.as_unit("ns")
    r = d.reindex(DG)
    CD[s] = (r.open.to_numpy(float), r.close.to_numpy(float))

# equity daily on the SPY session calendar
DSESS = pd.DatetimeIndex(eq1d["SPY"].index).as_unit("ns")
DSESS_DATES = np.array([t.date() for t in DSESS])
ED = {}
for s, d in eq1d.items():
    d = d.copy(); d.index = d.index.as_unit("ns")
    r = d.reindex(DSESS)
    ED[s] = (r.open.to_numpy(float), r.close.to_numpy(float))

def et_ns(dates, hhmm_minutes):
    """dates: list of datetime.date, minutes after midnight ET -> UTC ns array (DST-safe)"""
    naive = pd.DatetimeIndex([pd.Timestamp(d) + pd.Timedelta(minutes=m) for d in dates for m in hhmm_minutes])
    return naive.tz_localize(ET).tz_convert("UTC").as_unit("ns").asi8

# equity 5-minute -> per-session slot arrays (78 slots from 09:30)
spy5 = m5["SPY"].copy(); spy5.index = spy5.index.tz_convert(ET)
SESS5 = sorted(set(spy5.index.date))
NS5 = len(SESS5)
SIDX = {d: i for i, d in enumerate(SESS5)}
M5 = {}
for s, d in m5.items():
    d = d.copy(); d.index = d.index.tz_convert(ET)
    O = np.full((NS5, 78), np.nan); Cl = np.full((NS5, 78), np.nan)
    si = np.array([SIDX.get(x, -1) for x in d.index.date])
    sl = ((d.index.hour * 60 + d.index.minute) - 570) // 5
    ok = (si >= 0) & (sl >= 0) & (sl < 78)
    O[si[ok], sl[ok]] = d.open.to_numpy()[ok]; Cl[si[ok], sl[ok]] = d.close.to_numpy()[ok]
    M5[s] = (O, Cl)

def open_at(sym, slot):
    """first valid 5m open in slots slot..slot+2 (i.e. within 15 min), per session"""
    O = M5[sym][0]; out = np.full(NS5, np.nan)
    for k in range(3):
        j = slot + k
        if j < 78:
            out = np.where(np.isnan(out), O[:, j], out)
    return out

def close_by(sym, m):
    """close of last valid 5m bar starting in [m-3, m-1] slots (i.e. ending by mark m, within 15 min)"""
    Cl = M5[sym][1]; out = np.full(NS5, np.nan)
    for k in range(1, 4):
        j = m - k
        if j >= 0:
            out = np.where(np.isnan(out), Cl[:, j], out)
    return out

# equity intraday mark grids
EQ_MARKS = {"1h": [6, 18, 30, 42, 54, 66, 78], "4h": [48, 78]}  # slot boundaries: 10:00..16:00 ; 13:30,16:00
EQM_T = {tf: et_ns(SESS5, [570 + 5 * m for m in ms]) for tf, ms in EQ_MARKS.items()}
_eqm_cache = {}
def eq_marks(sym, tf):
    key = (sym, tf)
    if key not in _eqm_cache:
        if tf == "1d":
            _eqm_cache[key] = (DSESS_T16, ED[sym][1])
        else:
            vals = np.stack([close_by(sym, m) for m in EQ_MARKS[tf]], axis=1).ravel()
            _eqm_cache[key] = (EQM_T[tf], vals)
    return _eqm_cache[key]
DSESS_T16 = et_ns(list(DSESS_DATES), [960])

BAR_H = {"1h": 1, "4h": 4}

# ------------------------------------------------------------------ signals
def sig(sym, T, L, tf):
    """lead return over last L bars ending at latest mark <= T (T: UTC ns array)."""
    T = np.asarray(T, dtype=np.int64)
    out = np.full(len(T), np.nan)
    if sym[0] == "c":
        if tf == "1d":
            close = CD[sym[1]][1]; k = (T - DG0) // D_NS - 1; step = L
        else:
            close = CH[sym[1]][1]; k = (T - HG0) // H_NS - 1; step = L * BAR_H[tf]
        n = len(close)
    else:
        times, close = eq_marks(sym[1], tf)
        k = np.searchsorted(times, T, side="right") - 1; step = L; n = len(close)
    k0 = k - step
    ok = (k0 >= 0) & (k < n) & (k >= 0)
    out[ok] = close[k[ok]] / close[k0[ok]] - 1
    return out

# ------------------------------------------------------------------ event sets
def to_dates(T, tz):
    return pd.DatetimeIndex(T).tz_localize("UTC").tz_convert(tz).date if tz != "UTC" else \
        pd.DatetimeIndex(T).tz_localize("UTC").date

class EvSet:
    def __init__(self, tau, entry_ts, exit_ts, ret, date, split, ctrl_T, cost, kind):
        o = np.argsort(entry_ts, kind="stable")
        self.tau = np.asarray(tau, np.int64)[o]; self.entry = np.asarray(entry_ts, np.int64)[o]
        self.exit = np.asarray(exit_ts, np.int64)[o]; self.ret = np.asarray(ret, float)[o]
        self.date = np.asarray(date)[o]; self.ctrl_T = np.asarray(ctrl_T, np.int64)[o]
        self.cost = cost; self.kind = kind
        self.h2 = self.date >= split; self.split = split
        # integer day codes for clustering
        self.dcode = np.unique(self.date, return_inverse=True)[1]

_ev_cache = {}
def evset(target, tf, H, grid):
    """grid: 'eq' (equity target), 'cn' (crypto native), 'ce' (crypto target on equity-session grid)"""
    key = (target, tf, H, grid)
    if key in _ev_cache:
        return _ev_cache[key]
    if grid == "eq":
        sym = target[1]
        if tf == "1h":
            taus, exit_tss, rets, dates = [], [], [], []
            for j in range(1, 7):                     # bar starting 10:00 + (j-1)h ; tau = its start
                if j + H - 1 > 6:                     # must exit by the 16:00 close (no overnight 1h)
                    continue
                tau_min = 600 + 60 * (j - 1)
                ent_px = open_at(sym, 7 + 12 * (j - 1))            # tau + 5m
                ex_min = tau_min + 60 * H
                if ex_min == 960:
                    ex_px = close_by(sym, 78); ex_ts_min = 960
                else:
                    ex_px = open_at(sym, (ex_min - 570) // 5 + 1); ex_ts_min = ex_min + 5
                taus.append(et_ns(SESS5, [tau_min])); exit_tss.append(et_ns(SESS5, [ex_ts_min]))
                rets.append(ex_px / ent_px - 1); dates.append(np.array(SESS5))
            tau = np.concatenate(taus); ret = np.concatenate(rets); date = np.concatenate(dates)
            entry = tau + 5 * M_NS; exit_ = np.concatenate(exit_tss)
            ev = EvSet(tau, entry, exit_, ret, date, SPLIT["eq_intra"], tau, EQ_COST, "eq")
        elif tf == "4h":
            nb = 2 * NS5
            o_m, o_a, c_16 = open_at(sym, 1), open_at(sym, 49), close_by(sym, 78)
            t_930, t_1330, t_1335, t_1600 = (et_ns(SESS5, [m]) for m in (570, 810, 815, 960))
            ent_px = np.empty(nb); tau = np.empty(nb, np.int64)
            ent_px[0::2], ent_px[1::2] = o_m, o_a
            tau[0::2], tau[1::2] = t_930, t_1330
            exit_px_bar = np.empty(nb); exit_ts_bar = np.empty(nb, np.int64)   # exit AFTER bar f
            exit_px_bar[0::2], exit_px_bar[1::2] = o_a, c_16
            exit_ts_bar[0::2], exit_ts_bar[1::2] = t_1335, t_1600
            f = np.arange(nb); g = f + H - 1; ok = g < nb
            f, g = f[ok], g[ok]
            ret = exit_px_bar[g] / ent_px[f] - 1
            date = np.array(SESS5)[f // 2]
            ev = EvSet(tau[f], tau[f] + 5 * M_NS, exit_ts_bar[g], ret, date, SPLIT["eq_intra"], tau[f], EQ_COST, "eq")
        else:  # 1d
            o, c = ED[sym]
            n = len(DSESS); f = np.arange(n); g = f + H - 1; ok = g < n; f, g = f[ok], g[ok]
            tau = et_ns(list(DSESS_DATES[f]), [570]); ex_ts = DSESS_T16[g]
            ev = EvSet(tau, tau, ex_ts, c[g] / o[f] - 1, DSESS_DATES[f], SPLIT["daily"], tau, EQ_COST, "eq")
    elif grid == "cn":
        sym = target[1]
        if tf == "1d":
            o, c = CD[sym]; f = np.arange(ND); g = f + H - 1; ok = g < ND; f, g = f[ok], g[ok]
            tau = DG0 + f * D_NS
            ev = EvSet(tau, tau, tau + H * D_NS, c[g] / o[f] - 1, to_dates(tau, "UTC"), SPLIT["daily"], tau, CR_COST, "cr")
        else:
            b = BAR_H[tf]; o, c = CH[sym]
            f = np.arange(0, NH, b); g = f + b * H - 1; ok = g < NH; f, g = f[ok], g[ok]
            tau = HG0 + f * H_NS
            ev = EvSet(tau, tau, tau + b * H * H_NS, c[g] / o[f] - 1, to_dates(tau, "UTC"),
                       SPLIT["crypto_intra"], tau, CR_COST, "cr")
    else:  # 'ce' crypto target, equity-session decision grid (F3)
        sym = target[1]
        if tf == "1d":
            o, c = CD[sym]
            tau = DSESS_T16
            ent_day = np.array([(pd.Timestamp(d) + pd.Timedelta(days=1)).tz_localize("UTC").value for d in DSESS_DATES])
            f = (ent_day - DG0) // D_NS; g = f + H - 1
            ok = (f >= 0) & (g < ND)
            ret = np.full(len(tau), np.nan); ret[ok] = c[g[ok]] / o[f[ok]] - 1
            ev = EvSet(tau, ent_day, ent_day + H * D_NS, ret, DSESS_DATES, SPLIT["daily"], ent_day, CR_COST, "cr")
        else:
            b = BAR_H[tf]; o, c = CH[sym]
            tau = EQM_T[tf]                                  # equity mark times (13:30/16:00 or 10..16)
            ent_ts = -(-(tau - HG0) // H_NS) * H_NS + HG0      # ceil to the hour
            f = (ent_ts - HG0) // H_NS; g = f + b * H - 1
            ok = (f >= 0) & (g < NH)
            ret = np.full(len(tau), np.nan); ret[ok] = c[g[ok]] / o[f[ok]] - 1
            date = np.repeat(np.array(SESS5), len(EQ_MARKS[tf]))
            ev = EvSet(tau, ent_ts, ent_ts + b * H * H_NS, ret, date, SPLIT["eq_intra"], ent_ts, CR_COST, "cr")
    _ev_cache[key] = ev
    return ev

# ------------------------------------------------------------------ evaluation
def tp(x):
    n = len(x)
    if n < 3:
        return np.nan, 1.0
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.nan, 1.0
    t = x.mean() / (sd / np.sqrt(n))
    return float(t), float(stats.t.sf(t, n - 1))

def select_nonoverlap(idx, ev):
    if len(idx) <= 1:
        return idx
    e = ev.entry[idx]; x = ev.exit[idx]
    if np.all(e[1:] >= x[:-1]):
        return idx
    keep = []; last = -2**62
    for j, (a, b) in enumerate(zip(e.tolist(), x.tolist())):
        if a >= last:
            keep.append(j); last = b
    return idx[np.array(keep, dtype=np.int64)]

def trades(ev, s, direction, thr):
    valid = np.isfinite(s) & np.isfinite(ev.ret)
    m = valid & ((s > thr) if direction == "cont" else (s < -thr))
    return select_nonoverlap(np.flatnonzero(m), ev)

def stats_of(ev, idx, cost):
    net = ev.ret[idx] - cost
    n = len(net)
    t, p1 = tp(net)
    if n >= 3:
        dsum = np.bincount(ev.dcode[idx], weights=net)
        dsum = dsum[np.bincount(ev.dcode[idx]) > 0]
        td, p2 = tp(dsum) if len(dsum) >= 3 else (np.nan, 1.0)
        nd = len(dsum)
    else:
        td, p2, nd = np.nan, 1.0, n
    h2 = ev.h2[idx]
    return dict(n=n, n_days=nd, win=float((net > 0).mean()) if n else np.nan,
                mean_net=float(net.mean()) if n else np.nan, sd=float(net.std(ddof=1)) if n > 1 else np.nan,
                t=t, p_trade=p1, t_day=td, p_day=p2, p=max(p1, p2),
                n_h1=int((~h2).sum()), n_h2=int(h2.sum()),
                mean_h1=float(net[~h2].mean()) if (~h2).any() else np.nan,
                mean_h2=float(net[h2].mean()) if h2.any() else np.nan,
                mean_net_2x=float((net - cost).mean()) if n else np.nan)

def thresholds(ev, s):
    valid = np.isfinite(s) & np.isfinite(ev.ret)
    a = np.abs(s[valid])
    if len(a) == 0:
        return {"0": 0.0, "p70": np.nan, "p80": np.nan}, valid
    return {"0": 0.0, "p70": float(np.quantile(a, 0.7)), "p80": float(np.quantile(a, 0.8))}, valid

def evaluate(ev, s):
    """6 results (2 directions x 3 thresholds) + unconditional + IC for one signal on one event set"""
    thr, valid = thresholds(ev, s)
    uncond = float((ev.ret[valid] - ev.cost).mean()) if valid.any() else np.nan
    ic = float(np.corrcoef(s[valid], ev.ret[valid])[0, 1]) if valid.sum() > 10 else np.nan
    dates = ev.date[valid]
    meta = dict(n_events=int(valid.sum()), uncond_net=uncond, ic=ic,
                first_date=str(dates.min()) if len(dates) else "", last_date=str(dates.max()) if len(dates) else "")
    out = {}
    for tn, tv in thr.items():
        for dn in ("cont", "rev"):
            if not np.isfinite(tv):
                idx = np.array([], dtype=np.int64)
            else:
                idx = trades(ev, s, dn, tv)
            r = stats_of(ev, idx, ev.cost); r.update(thr=tv); r.update(meta)
            out[(dn, tn)] = r
    return out

# ------------------------------------------------------------------ pairs
F1_INTRA = [(C("BTC"), E(t)) for t in ["COIN", "MSTR", "MARA", "RIOT", "HOOD", "IBIT", "CONL", "SBIT"]] + \
           [(C("ETH"), E("ETHA")), (C("ETH"), E("COIN")), (C("XRP"), E("XRP")), (C("XRP"), E("XXRP"))]
F1_DAILY = [p for p in F1_INTRA if p[1][1] in eq1d]
F2 = [(C(a), C(b)) for a in CRYPTO for b in CRYPTO if a != b]
F3 = [(E(e), C(c)) for e in ["SPY", "QQQ", "COIN", "MSTR", "NVDA"] for c in CRYPTO]
F4 = [(E("NVDA"), E(t)) for t in ["AMD", "MU", "AVGO", "ARM", "SMH", "SOXL", "NVDL"]] + [(E("TSLA"), E("TSLL"))] + \
     [(E(l), E(t)) for l in ["SPY", "QQQ"] for t in ["TQQQ", "SOXL", "IWM", "ARKK"]]
EQ_TARGETS = sorted(set(t for _, t in F1_INTRA + F4))
CR_TARGETS = [C(c) for c in CRYPTO]

def grid_of(lead, target):
    if target[0] == "e":
        return "eq"
    return "cn" if lead[0] == "c" else "ce"

def market_ctrl(lead, target):
    if target[0] == "e":
        return E("QQQ") if lead == E("SPY") else E("SPY")
    return "own" if lead == C("BTC") else C("BTC")

def ctrl_T(ev, target, ctrl):
    """time at which a control signal is read: equity targets -> tau; crypto targets -> entry time"""
    return ev.tau if target[0] == "e" else ev.ctrl_T

def test_list():
    tl = []
    for tf in TFS:
        f1 = F1_DAILY if tf == "1d" else F1_INTRA
        for fam, pairs in (("F1", f1), ("F2", F2), ("F3", F3), ("F4", F4)):
            for lead, tgt in pairs:
                tl.append((fam, tf, lead, tgt))
        eqt = [t for t in EQ_TARGETS if (tf != "1d" or t[1] in eq1d)]
        for tgt in eqt + CR_TARGETS:
            tl.append(("F5", tf, tgt, tgt))
    return tl

# ------------------------------------------------------------------ main
_ctrl_cache = {}
def run():
    rows = []
    TL = test_list()
    for (fam, tf, lead, tgt) in TL:
        grid = grid_of(lead, tgt) if fam != "F5" else ("eq" if tgt[0] == "e" else "cn")
        for L, H in LH:
            ev = evset(tgt, tf, H, grid)
            # lead signal: equity-target & F3 read at tau; crypto native at tau (= entry)
            s = sig(lead, ev.tau, L, tf) if not (fam == "F5" and tgt[0] == "c") else sig(lead, ev.ctrl_T, L, tf)
            res = evaluate(ev, s)
            # control (ii) and own-momentum control on the SAME event set
            mc = market_ctrl(lead, tgt) if fam != "F5" else (None if tgt in (E("SPY"), C("BTC")) else
                                                              (E("SPY") if tgt[0] == "e" else C("BTC")))
            ctrl_res = {}
            for cname, csym in (("mkt", mc), ("own", tgt)):
                if csym is None:
                    ctrl_res[cname] = None; continue
                csym2 = tgt if csym == "own" else csym
                ck = (tgt, tf, H, grid, L, csym2)
                if ck not in _ctrl_cache:
                    _ctrl_cache[ck] = evaluate(ev, sig(csym2, ctrl_T(ev, tgt, csym2), L, tf))
                ctrl_res[cname] = (csym2, _ctrl_cache[ck])
            for (dn, tn), r in res.items():
                row = dict(family=fam, tf=tf, lead=lab(lead), target=lab(tgt), L=L, H=H, direction=dn,
                           thr_name=tn, grid=grid, cost=ev.cost, split=str(ev.split))
                row.update(r)
                for cname in ("mkt", "own"):
                    cr = ctrl_res[cname]
                    if cr is None:
                        row[f"{cname}_ctrl"] = ""; row[f"{cname}_n"] = 0; row[f"{cname}_mean_net"] = np.nan
                        row[f"{cname}_t"] = np.nan
                    else:
                        cs, cres = cr; c = cres[(dn, tn)]
                        row[f"{cname}_ctrl"] = lab(cs); row[f"{cname}_n"] = c["n"]
                        row[f"{cname}_mean_net"] = c["mean_net"]; row[f"{cname}_t"] = c["t"]
                rows.append(row)
    return pd.DataFrame(rows)

def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    q = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(q, 1.0)
    return out

if __name__ == "__main__":
    R = run()
    R["p"] = R.p.fillna(1.0)
    R["q"] = bh(R.p.values)
    R["beats_uncond"] = R.mean_net > R.uncond_net
    # control (ii): the market-lead rule (or own-lag when the lead is BTC for crypto targets)
    ctrl_mean = np.where(R.mkt_ctrl == R.target, R.own_mean_net, R.mkt_mean_net)
    ctrl_n = np.where(R.mkt_ctrl == R.target, R.own_n, R.mkt_n)
    R["ctrl2_mean_net"] = ctrl_mean; R["ctrl2_n"] = ctrl_n
    R["beats_ctrl2"] = (ctrl_n == 0) | (R.mean_net > np.nan_to_num(ctrl_mean, nan=-np.inf))
    R["robust"] = (R.n >= 30) & (R.mean_h1 > 0) & (R.mean_h2 > 0)
    R["candidate"] = (R.q <= 0.10) & R.robust & R.beats_uncond & R.beats_ctrl2 & (R.family != "F5")
    R.insert(0, "test_id", np.arange(len(R)))
    R.to_csv(os.path.join(HERE, "all_tests.csv"), index=False, float_format="%.6g")
    meta = dict(total_tests=int(len(R)), by_family=R.family.value_counts().to_dict(),
                by_tf=R.tf.value_counts().to_dict(), min_p=float(R.p.min()), min_q=float(R.q.min()),
                n_p_below_05=int((R.p < 0.05).sum()), n_q_le_10=int((R.q <= 0.10).sum()),
                n_robust=int(R.robust.sum()), n_candidates=int(R.candidate.sum()),
                runtime_s=round(time.time() - t0, 1))
    json.dump(meta, open(os.path.join(HERE, "run_meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
