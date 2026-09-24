#!/bin/bash
# Timeframe study, stage A replays — sequential, 4 workers each.
set -e
cd /home/user/BOT
SP=/tmp/claude-0/-home-user-BOT/59c0a99d-d529-57f6-bcb4-f09f32cd2414/scratchpad
export OMP_NUM_THREADS=1
while ! grep -q '^done' $SP/tf/fetch_crypto.log; do sleep 10; done
python3 - <<PY
import pickle, pandas as pd
SP='$SP'
agg={'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
h=pickle.load(open(f'{SP}/bt/bars.pkl','rb'))
eq2={}
for s,df in h.items():
    et=df.index.tz_convert('America/New_York')
    key=et.normalize()+pd.to_timedelta(((et.hour-10)//2)*2+10,unit='h')
    g=df.groupby(key.tz_convert('UTC'))
    o=g.agg(agg); o['n']=g.size(); eq2[s]=o[o['n']==2].drop(columns='n')
pickle.dump(eq2,open(f'{SP}/tf/equity_2h.pkl','wb'))
c=pickle.load(open(f'{SP}/tf/crypto_1h_2y.pkl','rb'))
c4={}
for s,df in c.items():
    r=df.resample('4h'); o=r.agg(agg); o['n']=r.size(); c4[s]=o[o['n']==4].drop(columns='n')
pickle.dump(c4,open(f'{SP}/tf/crypto_4h.pkl','wb'))
print('derived: eq2h', len(eq2['SPY']), 'bars SPY; c4h', len(c4['BTC']), 'bars BTC', flush=True)
PY
echo "== equity 1d $(date -u +%H:%M)"
python3 -m bot.research.tf_precompute $SP/tf/equity_1d.pkl $SP/tf/sig_eq_1d --bar-hours 6.5 --thresh 0.0102 --history 500 --retrain 5 --test-start 2025-09-23 --equity-daily
echo "== crypto 1d $(date -u +%H:%M)"
python3 -m bot.research.tf_precompute $SP/tf/crypto_1d.pkl $SP/tf/sig_c_1d --bar-hours 24 --thresh 0.0196 --history 500 --retrain 5 --test-start 2025-09-23
echo "== crypto 4h $(date -u +%H:%M)"
python3 -m bot.research.tf_precompute $SP/tf/crypto_4h.pkl $SP/tf/sig_c_4h --bar-hours 4 --thresh 0.0080 --history 500 --retrain 12 --test-start 2025-09-23
echo "== equity 2h $(date -u +%H:%M)"
python3 -m bot.research.tf_precompute $SP/tf/equity_2h.pkl $SP/tf/sig_eq_2h --bar-hours 2 --thresh 0.0057 --history 300 --retrain 6 --test-start 2026-01-01
echo "ALL DONE $(date -u +%H:%M)"
