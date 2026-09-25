"""Beta/placebo controls for the best-ranked discovery rules (diagnostic only; not counted as tests).
Control: replace the crypto lead with SPY or QQQ's own return over the analogous window
(A: P_eq(10:30 today)/P_eq(16:00 prev session)-1; thresholds = same percentile of |control|)."""
import os; os.environ.setdefault("OMP_NUM_THREADS","1")
import pandas as pd, numpy as np
import leadlag_discovery as L
R = pd.read_csv("all_tests.csv")
top = R[(R.family=="A") & ((R.robust) | (R.p < 0.2))].sort_values("p")
def ctrl_A(ctl, tgt, ex):
    g, c = L.EQ[tgt], L.EQ[ctl]; rows=[]
    for i, d in enumerate(L.DAYS):
        if i == 0: continue
        pdy = L.DAYS[i-1]
        rows.append(dict(date=d, s=c.at[d,"10:30"]/c.at[pdy,"16:00"]-1, ret=g.at[d,ex]/g.at[d,"10:30"]-1))
    return pd.DataFrame(rows)
out=[]
for _, r in top.iterrows():
    for ctl in ["SPY","QQQ"]:
        res = [x for x in L.evaluate(ctrl_A(ctl, r.target, r.horizon), L.EQ_COST)
               if x["thr_name"]==r.thr_name and x["direction"]==r.direction][0]
        out.append(dict(lead=r.lead, target=r.target, horizon=r.horizon, thr_name=r.thr_name, direction=r.direction,
                        rule_n=r.n, rule_mean_net=r.mean_net, uncond_net=r.uncond_net, control=ctl,
                        ctrl_n=res["n"], ctrl_mean_net=res["mean_net"], ctrl_t=res["t"]))
C = pd.DataFrame(out); C.to_csv("controls.csv", index=False)
pd.set_option("display.width",250); print(C.round(4).to_string())
# sanity check one trade by hand
import pickle
cr = pickle.load(open(os.path.join(L.DATA,"crypto_bars.pkl"),"rb")); b5 = pickle.load(open(os.path.join(L.DATA,"bars5.pkl"),"rb"))
s = cr["BTC"].open[pd.Timestamp("2026-03-10 14:00",tz="UTC")]/cr["BTC"].open[pd.Timestamp("2026-03-09 20:00",tz="UTC")]-1
ib = b5["IBIT"]; ret = ib.close[pd.Timestamp("2026-03-10 19:30",tz="UTC")]/ib.open[pd.Timestamp("2026-03-10 14:30",tz="UTC")]-1
a = L.build_A("BTC","IBIT","16:00"); print("hand", s, ret, "code", a[a.date==pd.Timestamp("2026-03-10").date()].to_dict("records"))
