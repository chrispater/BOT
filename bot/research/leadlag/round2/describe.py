"""Descriptive diagnostics for round-2 discovery (NOT tests; nothing here is used for selection).

Run after discover.py:  OMP_NUM_THREADS=1 python3 describe.py
Writes: per_pair_tf.csv, ic_summary.csv, placebo_calibration.csv, near_misses.csv

1. Placebo calibration: for every non-F5 cell, the lead signal is circularly shifted by a random
   25-75% of the event-set length (breaks lead timing, keeps signal distribution and regime mix),
   then the same 6 tests are evaluated. Compare the count of small p-values with the real grid.
2. Gross IC summary (Pearson corr of s with the gross window return, all events), H=1 cells.
3. Per-pair best timeframe (smallest-p cell per pair and timeframe) - description only.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, pandas as pd
import discover as D

HERE = D.HERE
R = pd.read_csv(os.path.join(HERE, "all_tests.csv"))
N5 = R[R.family != "F5"]

# ---------------------------------------------------------------- 1. placebo
def placebo(seed):
    rng = np.random.default_rng(seed)
    ps = []
    for (fam, tf, lead, tgt) in D.test_list():
        if fam == "F5":
            continue
        grid = D.grid_of(lead, tgt)
        for L, H in D.LH:
            ev = D.evset(tgt, tf, H, grid)
            s = D.sig(lead, ev.tau, L, tf)
            n = len(s)
            s = np.roll(s, int(rng.integers(n // 4, 3 * n // 4)))
            for (dn, tn), r in D.evaluate(ev, s).items():
                ps.append((fam, tf, r["p"], r["n"], r["mean_h1"], r["mean_h2"]))
    return pd.DataFrame(ps, columns=["family", "tf", "p", "n", "mean_h1", "mean_h2"])

def summ(df, label):
    df = df.assign(p=df.p.fillna(1.0))
    rob = (df.n >= 30) & (df.mean_h1 > 0) & (df.mean_h2 > 0)
    return dict(run=label, tests=len(df), min_p=df.p.min(), p_lt_05=int((df.p < .05).sum()),
                p_lt_01=int((df.p < .01).sum()), p_lt_001=int((df.p < .001).sum()),
                robust=int(rob.sum()), robust_p_lt_01=int((rob & (df.p < .01)).sum()))

rows = [summ(N5, "real")]
for sd in (11, 22, 33):
    rows.append(summ(placebo(sd), f"placebo_seed{sd}"))
PC = pd.DataFrame(rows)
PC.to_csv(os.path.join(HERE, "placebo_calibration.csv"), index=False)
print(PC.to_string())

# ---------------------------------------------------------------- 2. IC summary
cells = R.drop_duplicates(["family", "tf", "lead", "target", "L", "H"])[
    ["family", "tf", "lead", "target", "L", "H", "ic", "n_events", "uncond_net"]].copy()
cells["z"] = cells.ic * np.sqrt(cells.n_events)
h1 = cells[cells.H == 1]
ICS = h1.groupby(["family", "tf"]).agg(cells=("ic", "size"), mean_ic=("ic", "mean"), mean_abs_ic=("ic", lambda x: x.abs().mean()),
                                       share_pos=("ic", lambda x: (x > 0).mean()), n_absz_gt3=("z", lambda x: (x.abs() > 3).sum()),
                                       median_events=("n_events", "median")).round(4)
ICS.to_csv(os.path.join(HERE, "ic_summary.csv"))
print(ICS.to_string())
print("largest |z| H=1 cells (non-F5):")
print(h1[h1.family != "F5"].reindex(h1[h1.family != "F5"].z.abs().sort_values(ascending=False).index).head(15).round(4).to_string())

# ---------------------------------------------------------------- 3. per-pair best timeframe
best = R.loc[R.groupby(["family", "lead", "target", "tf"]).p.idxmin()]
ic11 = cells[(cells.L == 1) & (cells.H == 1)].set_index(["family", "lead", "target", "tf"]).ic
best = best.assign(ic_L1H1=[ic11.get((f, l, t, tf), np.nan) for f, l, t, tf in zip(best.family, best.lead, best.target, best.tf)])
PP = best[["family", "lead", "target", "tf", "L", "H", "direction", "thr_name", "n", "mean_net", "t", "p", "q", "ic_L1H1", "robust"]]
PP = PP.sort_values(["family", "lead", "target", "tf"])
PP.to_csv(os.path.join(HERE, "per_pair_tf.csv"), index=False, float_format="%.5g")
bt = PP.loc[PP.groupby(["family", "lead", "target"]).p.idxmin()]
print("best tf per pair (count of pairs whose smallest-p cell is at that tf):")
print(bt.groupby(["family", "tf"]).size().unstack(fill_value=0))

# ---------------------------------------------------------------- 4. near misses
gates = N5.assign(g_n=N5.n >= 30, g_halves=(N5.mean_h1 > 0) & (N5.mean_h2 > 0))
NM = gates.sort_values("p").head(25)
NM.to_csv(os.path.join(HERE, "near_misses.csv"), index=False, float_format="%.5g")
all_but_bh = gates[gates.robust & gates.beats_uncond & gates.beats_ctrl2]
print("tests passing every gate except BH:", len(all_but_bh), " of which p<0.01:", int((all_but_bh.p < .01).sum()))
