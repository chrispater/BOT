"""Print which ML engine the bot will actually use.

Run this on the target host (e.g. PythonAnywhere) to confirm the bot is using
gradient-boosted trees and not the weak SVM fallback:

    cd ~/cryptoscanner/CryptoQuantScanner && python3 backend/verify_ml_engine.py

Unlike `python3 -c "import lightgbm"`, this imports trading_service first, so the
dask-neutralizing guard runs exactly as it does in production.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import trading_service as ts  # noqa: E402  (guard runs on import)

print("=" * 52)
print(f"  LightGBM available : {ts.LGBM_AVAILABLE}")
print(f"  XGBoost  available : {ts.XGB_AVAILABLE}")
print(f"  ACTIVE ENGINE      : {ts._ACTIVE_ENGINE}")
print("=" * 52)
if not (ts.LGBM_AVAILABLE or ts.XGB_AVAILABLE):
    print("\n  ⚠  Running on SVM fallback. Fix with:")
    print("       pip install lightgbm xgboost")
    sys.exit(1)
print("\n  ✓ Gradient-boosted trees are live.")
