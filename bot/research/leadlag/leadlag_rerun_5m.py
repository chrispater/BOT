"""RERUN of families A, B, D using genuine 5-minute equity bars (bars5min_raw.pkl).
Same pairs / thresholds (0, p70, p80 of |s|) / directions / costs / evaluation as
leadlag_discovery.py.  Only equity entry/exit timing changes (and A/D signal end, so it
is known before the 9:50 entry).

Run: OMP_NUM_THREADS=1 python3 leadlag_rerun_5m.py   (after leadlag_discovery.py)
Outputs: rerun5m_tests.csv, rerun5m_controls.csv, combined_tests.csv

Price rules (ET, 5-minute bar-START timestamps):
  entry_at(T)  = open of the first 5m bar starting at or after T (must start within 15 min of T,
                 else the trade is skipped)
  close_1600   = close of the last 5m bar of the session (must start >= 15:50)
 A: s = P_c(09:00 ET today)/P_c(16:00 ET prev session) - 1   [latest crypto hourly price known
    before 09:50; P_c(T)=open of Coinbase hourly bar starting at T]
    entry = entry_at(09:50); exit = entry_at(10:50) [+1h], entry_at(11:50) [+2h], close_1600.
 D: as A, 16:00 exit, only sessions whose previous session is >= 3 calendar days earlier.
 B: H in 10..14, k in {1,3}: s = P_c(H:00)/P_c((H-k):00) - 1; entry = entry_at(H:05);
    exit = entry_at((H+1):05).
"""
import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import pickle
import numpy as np, pandas as pd
import leadlag_discovery as L

HERE = L.HERE
M5 = pickle.load(open(os.path.join(L.DATA, "bars5min_raw.pkl"), "rb"))
ET = L.ET

def prep(sym):
    d = M5[sym]; d = d[d.volume > 0].copy(); d.index = d.index.tz_convert(ET)
    return {day: g for day, g in d.groupby(d.index.date)}
G = {s: prep(s) for s in set([t for _, t in L.PAIRS] + ["SPY", "QQQ"])}

def at(sym, day, hhmm, maxwait=15):
    g = G[sym].get(day)
    if g is None: return np.nan
    t = pd.Timestamp(f"{day} {hhmm}").tz_localize(ET)
    w = g[(g.index >= t) & (g.index < t + pd.Timedelta(minutes=maxwait))]
    return w.open.iloc[0] if len(w) else np.nan

def close1600(sym, day):
    g = G[sym].get(day)
    if g is None or len(g) == 0: return np.nan
    last = g.iloc[-1]
    return last.close if g.index[-1].strftime("%H:%M") >= "15:50" else np.nan

def px_before(sym, day, hhmm):  # close of last bar ENDING at or before hhmm (for controls)
    g = G[sym].get(day)
    if g is None: return np.nan
    t = pd.Timestamp(f"{day} {hhmm}").tz_localize(ET) - pd.Timedelta(minutes=5)
    w = g[g.index <= t]
    return w.close.iloc[-1] if len(w) else np.nan

EXIT_A = {"+1h": "10:50", "+2h": "11:50", "16:00": None}

def exit_px(sym, day, ex):
    return close1600(sym, day) if ex == "16:00" else at(sym, day, EXIT_A[ex])

def build_A(lead, tgt, ex, monday_only=False, control=None):
    rows = []
    for i, day in enumerate(L.DAYS):
        if i == 0: continue
        pdy = L.DAYS[i - 1]
        if monday_only and (pd.Timestamp(day) - pd.Timestamp(pdy)).days < 3: continue
        if control:  # SPY/QQQ own return prev close -> 09:45 (causal at 09:50)
            s = px_before(control, day, "09:45") / close1600(control, pdy) - 1
        else:
            s = L.cpx(lead, day, 9) / L.cpx(lead, pdy, 16) - 1
        ent = at(tgt, day, "09:50")
        rows.append(dict(date=day, s=s, ret=exit_px(tgt, day, ex) / ent - 1))
    return pd.DataFrame(rows)

def build_B(lead, tgt, k, control=None):
    rows = []
    for day in L.DAYS:
        if day not in G[tgt]: continue
        for H in range(10, 15):
            if control:  # SPY/QQQ own return over (H-k):00 -> H:00 (09:30 open if before open)
                a = f"{H-k:02d}:00" if H - k >= 10 else "09:30"
                s = at(control, day, f"{H:02d}:00") / at(control, day, a) - 1
            else:
                s = L.cpx(lead, day, H) / L.cpx(lead, day, H - k) - 1
            ent, ex = at(tgt, day, f"{H:02d}:05"), at(tgt, day, f"{H+1:02d}:05")
            rows.append(dict(date=day, s=s, ret=ex / ent - 1))
    return pd.DataFrame(rows)

def bh(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); q = np.empty(m); prev = 1.0
    for rank, j in reversed(list(enumerate(order, 1))):
        prev = min(prev, p[j] * m / rank); q[j] = prev
    return q

def main():
    res = []
    def add(meta, df):
        for r in L.evaluate(df, L.EQ_COST):
            r.update(meta); r["run"] = "rerun5m"; res.append(r)
    for lead, tgt in L.PAIRS:
        for ex in EXIT_A:
            add(dict(family="A", lead=lead, target=tgt, horizon=ex), build_A(lead, tgt, ex))
        add(dict(family="D", lead=lead, target=tgt, horizon="16:00"), build_A(lead, tgt, "16:00", True))
        for k in [1, 3]:
            add(dict(family="B", lead=lead, target=tgt, horizon=f"k{k}_next1h"), build_B(lead, tgt, k))
    R5 = pd.DataFrame(res)
    R0 = pd.read_csv(os.path.join(HERE, "all_tests.csv")); R0["run"] = "original"
    R0["date_dummy"] = 0
    C = pd.concat([R0.drop(columns=["q", "robust", "date_dummy"]), R5], ignore_index=True)
    C["q_all"] = bh(C.p.fillna(1.0))                       # BH over ALL tests ever run (primary)
    C["robust"] = (C.n >= 30) & (C.mean_h1 > 0) & (C.mean_h2 > 0)
    C.to_csv(os.path.join(HERE, "combined_tests.csv"), index=False)
    R5 = C[C.run == "rerun5m"].copy()
    R5.to_csv(os.path.join(HERE, "rerun5m_tests.csv"), index=False)
    # sensitivity: BH over 648 tests with A/B/D replaced by the reruns
    alt = pd.concat([C[(C.run == "original") & C.family.isin(["C", "E"])], R5])
    alt_q = bh(alt.p.fillna(1.0))

    # controls for best-ranked rerun rules
    top = R5[(R5.p < 0.2) | R5.robust].sort_values("p")
    out = []
    for _, r in top.iterrows():
        for ctl in ["SPY", "QQQ"]:
            if r.family == "B":
                df = build_B(None, r.target, int(r.horizon[1]), control=ctl)
            else:
                df = build_A(None, r.target, r.horizon, monday_only=(r.family == "D"), control=ctl)
            x = [z for z in L.evaluate(df, L.EQ_COST) if z["thr_name"] == r.thr_name and z["direction"] == r.direction][0]
            out.append(dict(family=r.family, lead=r.lead, target=r.target, horizon=r.horizon, thr_name=r.thr_name,
                            direction=r.direction, rule_n=r.n, rule_mean_net=r.mean_net, rule_p=r.p,
                            uncond_net=r.uncond_net, control=ctl, ctrl_n=x["n"], ctrl_mean_net=x["mean_net"], ctrl_t=x["t"]))
    pd.DataFrame(out).to_csv(os.path.join(HERE, "rerun5m_controls.csv"), index=False)
    return C, R5, alt_q, pd.DataFrame(out)

if __name__ == "__main__":
    C, R5, alt_q, CT = main()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("total tests (combined):", len(C), " rerun tests:", len(R5))
    print("min q_all:", C.q_all.min(), " min q_all among reruns:", R5.q_all.min(), " min q (replacement view, 648):", alt_q.min())
    print("rerun p<0.05:", (R5.p < 0.05).sum(), "expected", 0.05 * len(R5))
    print("survivors (q_all<=0.10 & robust):", ((C.q_all <= 0.10) & C.robust).sum())
    cols = ["family", "lead", "target", "horizon", "thr_name", "thr", "direction", "n", "win", "mean_net", "t", "t_day",
            "p", "q_all", "mean_h1", "mean_h2", "n_h1", "n_h2", "uncond_net"]
    print(R5.sort_values("p")[cols].head(25).to_string())
    print("robust reruns:"); print(R5[R5.robust].sort_values("p")[cols].to_string())
    print(CT.round(4).to_string())
