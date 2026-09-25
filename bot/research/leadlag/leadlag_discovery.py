"""Lead-lag DISCOVERY study (discovery data only, ends 2026-06-30).

Run:  OMP_NUM_THREADS=1 python3 leadlag_discovery.py
Outputs (same dir): all_tests.csv, controls.csv, discovery.json

DATA NOTE (important): bars5.pkl is NOT 5-minute data. It contains session-aligned
60-minute bars starting 09:30 ET (09:30,10:30,...,15:30; the 15:30 bar is a half hour).
These are internally consistent (close of bar k == open of bar k+1 to ~1bp), whereas
bars.pkl hourly bars (10:00..15:00 ET) have large gaps between consecutive bars
(COIN median 21bp, 90th pct 117bp) and the 16:00 close differs from bars5 by >0.8% on
10% of days.  So ALL equity prices here come from bars5.pkl, on this grid:
   P_eq(T) for T in 09:30,10:30,...,15:30 = open of the bars5 bar starting at T
   P_eq(16:00) = close of the bars5 bar starting at 15:30
Crypto prices: P_c(T) = open of the hourly Coinbase bar starting at T (T on the hour, ET).
Equity sample is therefore 2026-02-23..2026-06-30 (89 sessions).

Timing rules (signal fully known at t, entry strictly after t):
 A: s = P_c(10:00 ET today)/P_c(16:00 ET previous session day) - 1; entry P_eq(10:30);
    exit P_eq(11:30) [1h], P_eq(12:30) [2h], P_eq(16:00) [close].
 B: for H in 10..14: s = P_c(H:00)/P_c((H-k):00) - 1, k in {1,3}; entry P_eq(H:30),
    exit P_eq((H+1):30).  Pooled over the 5 hourly slots.
 C: s = P_eq(16:00)/P_eq(09:30) - 1 of COIN/MSTR/IBIT/SPY/QQQ (session return);
    entry P_c(17:00 ET same day); exit P_c(10:00 ET next calendar day) [17h] or
    P_c(17:00 ET next calendar day) [24h].
 D: Monday-type sessions only (first session after a gap of >=2 calendar days... i.e.
    previous session is >=3 calendar days earlier): s = P_c(10:00 today)/P_c(16:00 prev
    session) - 1; entry P_eq(10:30); exit P_eq(16:00).
 E: hourly 24/7: s = P_c(T)/P_c(T-k h) - 1 of lead; entry P_c(T+1h); exit P_c(T+2h).
Thresholds: 0, and the 70th / 80th percentile of |s| over the discovery sample (then
frozen as fixed numbers). Directions: 'up' = trade when s > thr, 'down' = when s < -thr.
Each trade = buy the target (long only). Costs: 0.20% equities, 0.40% crypto round trip.
"""
import os, json, pickle
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "discovery_data")
ET = "America/New_York"
SPLIT = pd.Timestamp("2026-04-01").date()
EQ_COST, CR_COST = 0.0020, 0.0040

b5 = pickle.load(open(os.path.join(DATA, "bars5.pkl"), "rb"))
cr = pickle.load(open(os.path.join(DATA, "crypto_bars.pkl"), "rb"))

# ---------------- price grids ----------------
SLOTS = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00"]

def eq_grid(sym):
    d = b5[sym]
    d = d[d.volume > 0].copy()
    d.index = d.index.tz_convert(ET)
    rows = {}
    for ts, r in d.iterrows():
        day = ts.date(); hm = ts.strftime("%H:%M")
        rows.setdefault(day, {})[hm] = r.open
        if hm == "15:30":
            rows[day]["16:00"] = r.close
    g = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=SLOTS)
    g.index = pd.Index(sorted(g.index)); g = g.sort_index()
    return g

EQ = {s: eq_grid(s) for s in b5}
DAYS = sorted(set().union(*[set(g.index) for g in EQ.values()]))

CP = {}
for s, d in cr.items():
    d = d[d.volume > 0]
    ser = d.open.copy(); ser.index = ser.index.tz_convert(ET)
    CP[s] = ser

def cpx(sym, day, hour, add_days=0):
    # wall-clock ET time (DST-safe for the hours used: 10, 16, 17)
    ts =pd.Timestamp(pd.Timestamp(day) + pd.Timedelta(days=add_days) + pd.Timedelta(hours=hour)).tz_localize(ET)
    return CP[sym].get(ts, np.nan)

# ---------------- universe ----------------
PAIRS = [("BTC", t) for t in ["IBIT", "SBIT", "COIN", "MSTR", "MARA", "RIOT", "HOOD", "CONL"]] + \
        [("ETH", t) for t in ["ETHA", "COIN"]] + [("XRP", t) for t in ["XRP", "XXRP"]]
C_LEADS = ["COIN", "MSTR", "IBIT", "SPY", "QQQ"]
C_TARGETS = ["BTC", "ETH"]
E_PAIRS = [("BTC", t) for t in ["ETH", "SOL", "XRP", "DOGE"]] + [(l, "BTC") for l in ["ETH", "SOL", "XRP", "DOGE"]]

# ---------------- evaluation ----------------
def tstat_p(x):
    x = np.asarray(x, float); n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return np.nan, 1.0
    t = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    return t, float(stats.t.sf(t, n - 1))  # one-sided H1: mean net > 0

def evaluate(df, cost):
    """df: columns date, s, ret (gross target return). returns list of result dicts."""
    out = []
    df = df.dropna(subset=["s", "ret"])
    a = df.s.abs()
    thrs = {"0": 0.0, "p70": float(a.quantile(0.7)), "p80": float(a.quantile(0.8))}
    for tn, thr in thrs.items():
        for dn in ["up", "down"]:
            sel = df[df.s > thr] if dn == "up" else df[df.s < -thr]
            net = sel.ret - cost
            n = len(net)
            t, p = tstat_p(net)
            # day-clustered version (sum net per date) - conservative for pooled families
            daily = net.groupby(sel.date).sum()
            tc, pc = tstat_p(daily) if len(daily) >= 3 else (np.nan, 1.0)
            h1 = net[sel.date < SPLIT]; h2 = net[sel.date >= SPLIT]
            out.append(dict(thr_name=tn, thr=thr, direction=dn, n=n, n_days=len(daily),
                            mean_net=net.mean() if n else np.nan,
                            win=(net > 0).mean() if n else np.nan, t=t, p_trade=p, t_day=tc,
                            p=max(p, pc), n_h1=len(h1), n_h2=len(h2),
                            mean_h1=h1.mean() if len(h1) else np.nan,
                            mean_h2=h2.mean() if len(h2) else np.nan,
                            sum_h1=h1.sum(), sum_h2=h2.sum(),
                            uncond_net=(df.ret - cost).mean()))
    return out

def prev_day(i):
    return DAYS[i - 1] if i > 0 else None

# ---- family A & D ----
def build_A(lead, tgt, exit_slot, monday_only=False):
    g = EQ[tgt]; rows = []
    for i, day in enumerate(DAYS):
        if i == 0 or day not in g.index:
            continue
        pd_ = DAYS[i - 1]
        gapdays = (pd.Timestamp(day) - pd.Timestamp(pd_)).days
        if monday_only and gapdays < 3:
            continue
        s = cpx(lead, day, 10) / cpx(lead, pd_, 16) - 1
        ent, ex = g.at[day, "10:30"], g.at[day, exit_slot]
        rows.append(dict(date=day, s=s, ret=ex / ent - 1))
    return pd.DataFrame(rows)

# ---- family B ----
def build_B(lead, tgt, k, lead_is_equity=False):
    g = EQ[tgt]; rows = []
    for day in g.index:
        for H in range(10, 15):
            if lead_is_equity:  # control: lead equity's own return over k slots ending H:30
                lg = EQ[lead]
                if day not in lg.index: continue
                s0 = f"{H-k:02d}:30"
                if s0 not in SLOTS: continue
                s = lg.at[day, f"{H:02d}:30"] / lg.at[day, s0] - 1
            else:
                s = cpx(lead, day, H) / cpx(lead, day, H - k) - 1
            ent, ex = g.at[day, f"{H:02d}:30"], g.at[day, f"{H+1:02d}:30"]
            rows.append(dict(date=day, s=s, ret=ex / ent - 1))
    return pd.DataFrame(rows)

# ---- family C ----
def build_C(lead, tgt, hold, lead_is_crypto_self=False):
    rows = []
    for day in DAYS:
        if lead_is_crypto_self:
            s = cpx(tgt, day, 16) / cpx(tgt, day, 10) - 1
        else:
            lg = EQ[lead]
            if day not in lg.index: continue
            s = lg.at[day, "16:00"] / lg.at[day, "09:30"] - 1
        ent = cpx(tgt, day, 17)
        ex = cpx(tgt, day, 10, add_days=1) if hold == "17h" else cpx(tgt, day, 17, add_days=1)
        rows.append(dict(date=day, s=s, ret=ex / ent - 1))
    return pd.DataFrame(rows)

# ---- family E ----
def build_E(lead, tgt, k):
    L = cr[lead].open.copy(); T = cr[tgt].open.copy()
    idx = pd.date_range(L.index.min(), L.index.max(), freq="h")
    L = L.reindex(idx); T = T.reindex(idx)
    s = L / L.shift(k) - 1
    ret = T.shift(-2) / T.shift(-1) - 1
    df = pd.DataFrame(dict(s=s, ret=ret))
    df["date"] = idx.tz_convert(ET).date
    return df

def main():
    results = []
    def add(meta, df, cost):
        for r in evaluate(df, cost):
            r.update(meta); results.append(r)

    for lead, tgt in PAIRS:
        for ex in ["11:30", "12:30", "16:00"]:
            add(dict(family="A", lead=lead, target=tgt, horizon=ex), build_A(lead, tgt, ex), EQ_COST)
        add(dict(family="D", lead=lead, target=tgt, horizon="16:00"), build_A(lead, tgt, "16:00", True), EQ_COST)
        for k in [1, 3]:
            add(dict(family="B", lead=lead, target=tgt, horizon=f"k{k}_next1h"), build_B(lead, tgt, k), EQ_COST)
    for lead in C_LEADS:
        for tgt in C_TARGETS:
            for h in ["17h", "24h"]:
                add(dict(family="C", lead=lead, target=tgt, horizon=h), build_C(lead, tgt, h), CR_COST)
    for lead, tgt in E_PAIRS:
        for k in [1, 3]:
            add(dict(family="E", lead=lead, target=tgt, horizon=f"k{k}_next1h"), build_E(lead, tgt, k), CR_COST)

    R = pd.DataFrame(results)
    # Benjamini-Hochberg over ALL tests
    p = R.p.fillna(1.0).values; m = len(p); order = np.argsort(p)
    q = np.empty(m); prev = 1.0
    for rank, j in reversed(list(enumerate(order, 1))):
        prev = min(prev, p[j] * m / rank); q[j] = prev
    R["q"] = q
    R["robust"] = (R.n >= 30) & (R.sum_h1 > 0) & (R.sum_h2 > 0) & (R.mean_h1 > 0) & (R.mean_h2 > 0)
    R.to_csv(os.path.join(HERE, "all_tests.csv"), index=False)
    return R

if __name__ == "__main__":
    R = main()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("total tests", len(R))
    print("BH q<=0.10:", (R.q <= 0.10).sum(), " robust (n>=30, both halves +):", R.robust.sum(),
          " both:", ((R.q <= 0.10) & R.robust).sum())
    cols = ["family", "lead", "target", "horizon", "thr_name", "thr", "direction", "n", "win", "mean_net",
            "t", "t_day", "p", "q", "mean_h1", "mean_h2", "n_h1", "n_h2", "uncond_net"]
    print(R.sort_values("p")[cols].head(40).to_string())
