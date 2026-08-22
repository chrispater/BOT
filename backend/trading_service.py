import os
import sys
import math
import pickle
import numpy as np
import pandas as pd
import ccxt
import time
import logging
import threading
import talib
from datetime import datetime
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from .edge_analytics import bucket_keys as _edge_bucket_keys
import warnings
warnings.filterwarnings('ignore')

os.environ['LOKY_MAX_CPU_COUNT'] = '1'

_MODEL_DIR = os.environ.get('BOT_MODEL_DIR', '/tmp/bot_models')
os.makedirs(_MODEL_DIR, exist_ok=True)

# Some environments (e.g. PythonAnywhere) ship a stale system `dask` whose import
# crashes against a newer pandas (AttributeError: pandas.core.strings has no
# attribute 'StringMethods'). LightGBM imports dask.dataframe for its optional
# Dask integration — but an AttributeError escapes LightGBM's `except ImportError`
# guard and takes the whole process down on `import lightgbm`. We don't use dask
# anywhere, so if a broken dask is present we replace it with permissive stub
# modules BEFORE importing lightgbm. The stub satisfies any `from dask... import X`
# LightGBM attempts (returning a dummy type used only for isinstance checks in the
# Dask code path we never exercise), so LightGBM imports cleanly regardless of
# whether it wraps its dask import in try/except. Must run before the LightGBM import.
def _neutralize_broken_dask():
    import types
    import importlib.abc
    import importlib.machinery

    try:
        import dask.dataframe  # noqa: F401  (probe: succeeds on a healthy dask)
        return  # healthy dask — leave it alone
    except ImportError:
        return  # dask not installed — LightGBM degrades gracefully on its own
    except Exception:
        pass    # broken dask — fall through and stub it out

    # Purge any half-initialized dask modules the failed probe may have left behind.
    for _m in [m for m in list(sys.modules) if m == 'dask' or m.startswith('dask.')]:
        del sys.modules[_m]

    class _DaskStub(types.ModuleType):
        __path__ = []  # mark as a package so `import dask.<sub>` keeps resolving
        def __getattr__(self, name):
            # Any symbol LightGBM pulls (DataFrame, Series, Client, delayed, …)
            # resolves to a harmless placeholder type.
            return type(f"_DaskStub_{name}", (), {})

    class _DaskStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path, target=None):
            if fullname == 'dask' or fullname.startswith('dask.'):
                return importlib.machinery.ModuleSpec(fullname, self)
            return None
        def create_module(self, spec):
            return _DaskStub(spec.name)
        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _DaskStubFinder())

_neutralize_broken_dask()

# ── Optional high-performance models ─────────────────────────────────────────
# Priority: LightGBM → XGBoost → HistGradientBoosting (sklearn) → SVM
# Log the actual exception so failures are diagnosable in the web app logs.
_lgbm_import_err = None
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except Exception as _e:
    LGBM_AVAILABLE = False
    _lgbm_import_err = f"{type(_e).__name__}: {_e}"

_xgb_import_err = None
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception as _e:
    XGB_AVAILABLE = False
    _xgb_import_err = f"{type(_e).__name__}: {_e}"

# HistGradientBoostingClassifier — pure numpy, no large C extensions, already
# installed via scikit-learn. Used as the real fallback before SVM so the bot
# never has to resort to a kernel machine for tabular data.
from sklearn.ensemble import HistGradientBoostingClassifier
HGBT_AVAILABLE = True   # always True — it's part of the sklearn dep we require

# ── Optional signal engine (second-opinion ensemble) ──
try:
    _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from signal_engine import SignalEngine
    SIGNAL_ENGINE_AVAILABLE = True
except Exception:
    SIGNAL_ENGINE_AVAILABLE = False

# ── Optional Market Intelligence Engine ──
# A research-first alternative decision layer: instead of an ML BUY/SELL
# classifier, it records unlabeled market state, waits for outcomes to
# resolve, and only lets a model influence a trade after it survives purged
# walk-forward validation. See backend/mie/README.md. Off by default
# (mie_gate_enabled=False) — when enabled it can veto (never force) an entry
# the legacy ML/setup pipeline below wants to take, on the principle that
# DO_NOTHING should be the harder-to-earn-out-of default state, not the
# other way around.
try:
    from backend.mie import MarketIntelligenceEngine, EngineConfig, ObservationStore, ACTION_NOTHING as MIE_DO_NOTHING
    MIE_AVAILABLE = True
except Exception as _mie_err:
    MIE_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Market Intelligence Engine unavailable: {_mie_err}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Definitive startup record of the active ML engine.
if LGBM_AVAILABLE:
    _ACTIVE_ENGINE = 'LightGBM'
elif XGB_AVAILABLE:
    _ACTIVE_ENGINE = 'XGBoost'
else:
    _ACTIVE_ENGINE = 'HistGradientBoosting'   # sklearn — solid fallback, not SVM

logger.info(
    f"ML engine active: {_ACTIVE_ENGINE} "
    f"(LGBM={LGBM_AVAILABLE}, XGB={XGB_AVAILABLE}, HGBT={HGBT_AVAILABLE})"
)
if not LGBM_AVAILABLE:
    logger.warning(f"LightGBM unavailable — {_lgbm_import_err or 'no error captured'}")
if not XGB_AVAILABLE:
    logger.warning(f"XGBoost unavailable — {_xgb_import_err or 'no error captured'}")
if not (LGBM_AVAILABLE or XGB_AVAILABLE):
    logger.warning(
        "Using HistGradientBoosting (sklearn) — gradient-boosted trees, good "
        "calibration, handles NaN natively. Faster to fix: "
        "pip install lightgbm xgboost into the web app virtualenv."
    )

# ── Constants ──────────────────────────────────────────────────────────────
SYMBOL = 'BTC/USDT:USDT'
LEVERAGE = 10
RISK_PER_TRADE = 0.02
STOP_LOSS_PCT = 0.015
TAKE_PROFIT_PCT = 0.03
MIN_CONFIDENCE = 0.65
TIMEFRAME = '5m'
LOOKBACK_PERIODS = 1000
TRADE_COOLDOWN = 300
TAKER_FEE = 0.0006  # Blofin taker fee — market orders (exits)
MAKER_FEE = 0.0002  # Blofin maker fee — post-only limit orders (entries)
ENTRY_LIMIT_WAIT_S = 12   # how long a post-only entry may wait for a fill
MAX_SPREAD_PCT = 0.0005   # skip entries when bid/ask spread > 5 bps (fee killer)

# ── Setup-detector stream ──
# Deterministic technical setups (RSI bounce, divergence, breakout, shakeout)
# trade as a SECOND signal stream alongside the ML model. They have their own
# confidence floor — the ML min_confidence gate applies to ML-originated
# signals, not to pattern setups, which carry their own strict entry conditions.
SETUP_MIN_CONFIDENCE = 0.72

VALID_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']

# Canonical model feature set. Single source of truth so prepare_features() and the
# persisted-model compatibility guard never drift apart. Bump the implied "version"
# simply by changing this list — any model trained on a different set is discarded
# on load rather than crashing the predict path with a feature-count mismatch.
FEATURE_COLUMNS = [
    'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position',
    'atr', 'adx', 'stoch_k', 'stoch_d', 'cci', 'mfi',
    'roc', 'mom', 'trend_sma', 'volatility', 'volume_ratio',
    'vwap_distance', 'vwap_slope',
    'obv_slope', 'ad_slope', 'cmf',
    'volume_price_confirm', 'volume_divergence',
    'breakout_proximity', 'breakout_quality',
    'vol_weighted_mom', 'vol_weighted_roc',
    'bb_squeeze', 'in_squeeze',
    'vol_adj_adx', 'directional_volume',
    'ema200_distance', 'ema200_slope',
    # Temporal / sequence features — recent dynamics, not just snapshots.
    'ret_3', 'ret_5', 'ret_10', 'ret_accel',
    'rsi_delta3', 'macd_hist_delta3', 'adx_delta3',
    'bb_position_delta3', 'volume_ratio_delta3',
    'candle_streak', 'green_frac_10', 'range_pos_5', 'vol_regime',
]

# SVM fallback grid
SVM_PARAMS = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']}

# LightGBM defaults (fast, good crypto performance)
LGBM_PARAMS = {
    'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
    'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'class_weight': 'balanced', 'random_state': 42, 'n_jobs': 1, 'verbose': -1,
}

# XGBoost defaults
XGB_PARAMS = {
    'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': 1,
}


def _encode_y(y):
    """Map {-1, 1} → {0, 1} for tree-based classifiers."""
    return np.where(np.array(y) == -1, 0, 1)


def _decode_y(y_enc):
    """Map {0, 1} → {-1, 1} after tree model prediction."""
    return np.where(np.array(y_enc) == 0, -1, 1)


class TradingService:
    # Minimum trade-quality score required to enter once the gate is armed.
    # 50 = "no measured edge either way"; we demand evidence above neutral.
    QUALITY_MIN_SCORE = 55

    def __init__(self, user_id: int, api_key=None, api_secret=None, api_password=None,
                 starting_balance=10000, leverage=10, selected_coins=None,
                 risk_per_trade=0.02, stop_loss_pct=0.15, take_profit_pct=0.30,
                 trade_cooldown=300, min_confidence=0.65, timeframe='5m',
                 trailing_stop_pct=0.10, max_drawdown_pct=0.20,
                 retrain_every=50, profit_risk_multiplier=1.5,
                 adx_threshold=18,
                 daily_loss_limit=0.08, max_positions=3,
                 mie_gate_enabled=False,
                 on_trade=None, on_signal=None, on_performance=None,
                 on_observation=None, on_backfill=None, get_pending_obs=None):

        self.user_id = user_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_password = api_password
        self.exchange = None
        self.model = None
        self._model_is_tree = False
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.last_trade_times = {}      # symbol → timestamp
        self.positions = {}             # symbol → position dict
        self._reversal_streak = {}      # symbol → consecutive reversal-signal cycles
        self.entry_price = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        self.balance = starting_balance
        self.starting_balance = starting_balance
        self.leverage = leverage
        self.selected_coins = selected_coins or ['BTC/USDT:USDT']
        self.current_symbol_index = 0
        self.simulation_mode = True
        self.running = False
        self.signals_history = []
        self.trades_history = []
        self._cycle_count = 0

        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trade_cooldown = trade_cooldown
        self.min_confidence = min_confidence
        self.timeframe = timeframe if timeframe in VALID_TIMEFRAMES else '5m'
        self.trailing_stop_pct = trailing_stop_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.retrain_every = retrain_every
        self.profit_risk_multiplier = profit_risk_multiplier
        self.adx_threshold = max(5, int(adx_threshold))
        self.daily_loss_limit = max(0.01, min(0.50, float(daily_loss_limit)))
        self.max_positions = max(1, int(max_positions))
        # Daily P&L governor state (resets at UTC midnight)
        self._day_pnl: float = 0.0
        self._day_start_balance: float = starting_balance
        self._day_date: str = ''   # 'YYYY-MM-DD' of last reset

        self._model_file = os.path.join(_MODEL_DIR, f'model_u{user_id}.pkl')
        self._buffer_file = os.path.join(_MODEL_DIR, f'buffer_u{user_id}.pkl')
        self._train_buffer = self._load_train_buffer()
        self._feedback_buffer = self._load_feedback_buffer()
        self._load_persisted_model()
        self._last_candle_ts: dict = {}   # symbol → last candle open timestamp
        self._ohlcv_cache:    dict = {}   # symbol → full DataFrame (1000 rows), refreshed incrementally
        self._drawdown_baseline: float = None  # real exchange balance at first sync — used for drawdown, not starting_balance which may be inflated for sizing
        self._start_time: float = time.time()  # session start epoch — used for velocity metrics
        self._trade_roi_history: list = self._load_roi_history()  # persisted across restarts
        self._last_dynamic_leverage: int = leverage  # effective leverage used on last trade — set by _dynamic_leverage()
        self._trades_since_roi_save: int = 0
        self._last_exchange_total: float = None  # anchor for delta-based balance sync
        self._peak_balance: float = starting_balance  # all-time high — used for max-DD and recovery scaling

        # Market regime state (re-evaluated every 10 cycles)
        self.market_regime = 'sideways'     # 'bull' | 'bear' | 'sideways'
        self._regime_check_every = 10

        # Serializes trade execution: the cycle loop runs on the event-loop thread
        # while manual enter/exit run on a thread-pool thread — without this they
        # could both mutate positions/balance and double-open or double-close.
        self._trade_lock = threading.Lock()

        # Second-opinion signal engine
        self._signal_engine = SignalEngine() if SIGNAL_ENGINE_AVAILABLE else None

        # ── Market Intelligence Engine ──────────────────────────────────────
        # Records market state every cycle (unlabeled — outcomes get attached
        # once the future actually happens), and once enough resolved
        # observations exist, trains + purged-walk-forward-validates a
        # conditional-expectancy model per horizon. `mie_gate_enabled=True`
        # lets a validated MIE reading of DO_NOTHING veto the legacy ML/setup
        # signal below; it can only ever narrow trading, never force a trade
        # the legacy pipeline wasn't already about to take, and it starts
        # silent (no validated model, no observations) until it has earned an
        # opinion. `mie_last_decision` is kept per-symbol purely for status
        # reporting / the UI, independent of whether the gate is enabled.
        self.mie_gate_enabled: bool = bool(mie_gate_enabled)
        self._mie = None
        self._mie_store = None
        self.mie_last_decision: dict = {}     # symbol → TradeDecision
        self._mie_last_fit: float = 0.0
        self._mie_fit_interval: float = 6 * 3600     # re-validate every 6h once enough data exists
        self._mie_min_rows_to_fit: int = 500         # don't even attempt a fit below this many resolved rows
        self._mie_decision_ids: dict = {}            # symbol → last recorded decision row id (for close_decision)
        if MIE_AVAILABLE:
            try:
                self._mie_store = ObservationStore()
                self._mie = MarketIntelligenceEngine(config=EngineConfig(), store=self._mie_store)
            except Exception as e:
                logger.warning(f"User {self.user_id}: Market Intelligence Engine init failed: {e}")
                self._mie = None

        self.on_trade = on_trade
        self.on_signal = on_signal
        self.on_performance = on_performance
        # ── Market Intelligence Engine hooks ──
        # Observations flow OUT through callbacks (keeps this module importable
        # without a database); the computed edge profile flows BACK IN as a plain
        # dict so the quality gate never makes a DB call in the hot path.
        self.on_observation = on_observation
        self.on_backfill = on_backfill
        self.get_pending_obs = get_pending_obs
        self.edge_profile = {}          # refreshed by the host process
        self.quality_gate_enabled = False   # off until the profile has real samples
        self._mkt_ctx_cache = {}        # symbol → (ts, context extras)
        self._last_oi = {}              # symbol → previous open-interest reading
        self._btc_ret_5 = 0.0
        self._last_quality = {}         # symbol → last computed quality score

        self._initialize_exchange()

    # ── Exchange ────────────────────────────────────────────────────────────

    def _initialize_exchange(self):
        try:
            self.exchange = ccxt.blofin({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_password,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap', 'adjustForTimeDifference': True},
            })
            if self.api_key and self.api_secret and self.api_password:
                self.exchange.load_markets()
                self.simulation_mode = False
                logger.info(f"User {self.user_id}: Exchange connected — LIVE MODE")
            else:
                logger.warning(f"User {self.user_id}: No API keys — SIMULATION MODE")
        except Exception as e:
            logger.error(f"User {self.user_id}: Exchange init failed: {e}")
            self.simulation_mode = True

    def get_current_symbol(self):
        if self.selected_coins:
            return self.selected_coins[self.current_symbol_index % len(self.selected_coins)]
        return 'BTC/USDT:USDT'

    def fetch_ohlcv(self, symbol=None, limit=LOOKBACK_PERIODS, timeframe=None):
        symbol = symbol or self.get_current_symbol()
        tf = timeframe or self.timeframe
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception:
            return self._generate_simulated_data(limit)

    def fetch_ohlcv_fast(self, symbol=None):
        """
        Incremental OHLCV fetch: downloads 1000 candles once, then only fetches
        the last 5 candles on subsequent calls and merges them into the cache.
        ~200x less data per cycle than a full fetch — critical for low-latency live trading.
        """
        symbol = symbol or self.get_current_symbol()
        cached = self._ohlcv_cache.get(symbol)

        if cached is None or len(cached) < 200:
            # Cold start: full history needed for indicator calculation
            df = self.fetch_ohlcv(symbol=symbol, limit=LOOKBACK_PERIODS)
            if df is not None and len(df) > 0:
                self._ohlcv_cache[symbol] = df
            return df

        # Hot path: only fetch the last few candles and append
        try:
            raw = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=5)
            new_df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
            new_df.set_index('timestamp', inplace=True)
            combined = pd.concat([cached, new_df])
            combined = combined[~combined.index.duplicated(keep='last')].sort_index().tail(LOOKBACK_PERIODS)
            self._ohlcv_cache[symbol] = combined
            return combined
        except Exception:
            return cached  # Network blip — use stale cache rather than crash

    def _get_timeframe_minutes(self):
        return {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '2h': 120, '4h': 240, '1d': 1440}.get(self.timeframe, 5)

    def _get_pandas_freq(self):
        return {'1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min',
                '30m': '30min', '1h': '1h', '2h': '2h', '4h': '4h', '1d': '1D'}.get(self.timeframe, '5min')

    def _generate_simulated_data(self, limit):
        np.random.seed(int(time.time()) % 1000)
        base_price = 95000
        freq = self._get_pandas_freq()
        timestamps = pd.date_range(end=datetime.now(), periods=limit, freq=freq)
        prices = [base_price]
        for i in range(1, limit):
            change = np.random.randn() * 50 + np.sin(i / 20) * 30
            prices.append(max(prices[-1] + change, base_price * 0.9))
        return pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.randn()) * 0.002) for p in prices],
            'low': [p * (1 - abs(np.random.randn()) * 0.002) for p in prices],
            'close': [p + np.random.randn() * 20 for p in prices],
            'volume': [np.random.uniform(100, 1000) for _ in prices],
        }, index=timestamps)

    # ── Indicators ──────────────────────────────────────────────────────────

    def calculate_indicators(self, df):
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        open_price = df['open'].values.astype(float)
        volume = df['volume'].values.astype(float)

        df['sma_10'] = talib.SMA(close, timeperiod=10)
        df['sma_20'] = talib.SMA(close, timeperiod=20)
        df['ema_12'] = talib.EMA(close, timeperiod=12)
        df['ema_26'] = talib.EMA(close, timeperiod=26)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)
        df['rsi'] = talib.RSI(close, timeperiod=14)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close, timeperiod=20)
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        df['adx'] = talib.ADX(high, low, close, timeperiod=14)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close)
        df['cci'] = talib.CCI(high, low, close, timeperiod=20)
        df['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
        df['roc'] = talib.ROC(close, timeperiod=10)
        df['mom'] = talib.MOM(close, timeperiod=10)
        df['trend_sma'] = np.where(df['sma_10'] > df['sma_20'], 1, -1)
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Rolling 20-period VWAP (avoids cumsum drift across entire dataset)
        tp = pd.Series((high + low + close) / 3, index=df.index)
        vol = pd.Series(volume, index=df.index)
        df['vwap'] = (tp * vol).rolling(20).sum() / vol.rolling(20).sum()
        df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
        df['vwap_slope'] = df['vwap'].pct_change(periods=5) * 100

        df['obv'] = talib.OBV(close, volume)
        df['obv_sma'] = df['obv'].rolling(window=20).mean()
        df['obv_slope'] = df['obv'].pct_change(periods=5) * 100

        df['ad'] = talib.AD(high, low, close, volume)
        df['ad_slope'] = df['ad'].pct_change(periods=5) * 100

        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        mfv = pd.Series(mfm * volume, index=df.index)
        df['cmf'] = mfv.rolling(20).sum() / vol.rolling(20).sum()

        price_ch5 = df['close'].pct_change(periods=5)
        vol_ch5 = df['volume'].pct_change(periods=5)
        df['volume_price_confirm'] = np.sign(price_ch5) * np.sign(vol_ch5)
        exp_vol = abs(price_ch5) * df['volume_sma']
        act_vol = abs(df['volume'] - df['volume_sma'])
        df['volume_divergence'] = (act_vol - exp_vol) / (df['volume_sma'] + 1e-10)

        df['donchian_high'] = df['high'].rolling(20).max()
        df['donchian_low'] = df['low'].rolling(20).min()
        df['donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2
        dc_range = df['donchian_high'] - df['donchian_low']
        df['breakout_proximity'] = (df['close'] - df['donchian_mid']) / (dc_range / 2 + 1e-10)
        df['is_new_high'] = (df['close'] >= df['donchian_high'].shift(1)).astype(int)
        df['is_new_low'] = (df['close'] <= df['donchian_low'].shift(1)).astype(int)
        df['breakout_quality'] = (
            df['is_new_high'] * (1 + df['volume_ratio'].clip(0, 2) - 1) -
            df['is_new_low'] * (1 + df['volume_ratio'].clip(0, 2) - 1)
        )

        df['vol_weighted_mom'] = (df['returns'] * df['volume_ratio']).rolling(10).sum()
        df['vol_weighted_roc'] = df['vol_weighted_mom'].pct_change(periods=5) * 100

        # EMA-200 for long-term trend alignment — used by entry quality filter
        df['ema200'] = talib.EMA(close, timeperiod=200)
        df['ema200_distance'] = (df['close'] - df['ema200']) / (df['ema200'] + 1e-10)
        df['ema200_slope'] = df['ema200'].pct_change(periods=10) * 100

        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_squeeze'] = df['bb_width'] / df['bb_width'].rolling(50).mean()
        df['keltner_upper'] = df['ema_12'] + 2 * df['atr']
        df['keltner_lower'] = df['ema_12'] - 2 * df['atr']
        df['in_squeeze'] = ((df['bb_lower'] > df['keltner_lower']) &
                            (df['bb_upper'] < df['keltner_upper'])).astype(int)

        df['vol_adj_adx'] = df['adx'] * df['volume_ratio'].clip(0.5, 2)
        up_vol = pd.Series(np.where(close > open_price, volume, 0), index=df.index)
        dn_vol = pd.Series(np.where(close < open_price, volume, 0), index=df.index)
        df['directional_volume'] = (
            (up_vol.rolling(10).sum() - dn_vol.rolling(10).sum()) /
            (df['volume'].rolling(10).sum() + 1e-10)
        )

        # ── Temporal / sequence features ────────────────────────────────────────
        # The indicators above are point-in-time snapshots — they tell the model
        # what RSI *is*, not whether it's rising or falling, nor what price did over
        # the last several bars. These features give the tree model short-horizon
        # memory (velocity, acceleration, trajectory, persistence) so it can learn
        # temporal patterns without the overfitting risk of a deep sequence model.
        # All use only past data (pct_change / shift / rolling), so no lookahead.

        # Multi-bar price trajectory — where price has been heading recently.
        df['ret_3']  = df['close'].pct_change(periods=3)
        df['ret_5']  = df['close'].pct_change(periods=5)
        df['ret_10'] = df['close'].pct_change(periods=10)
        # Acceleration: is the most recent 1-bar move speeding up or fading?
        df['ret_accel'] = df['returns'] - df['returns'].shift(1)

        # Indicator velocity (delta over 3 bars) — direction of momentum/trend,
        # which a raw snapshot value cannot convey.
        df['rsi_delta3']          = df['rsi'] - df['rsi'].shift(3)
        df['macd_hist_delta3']    = df['macd_hist'] - df['macd_hist'].shift(3)
        df['adx_delta3']          = df['adx'] - df['adx'].shift(3)          # trend strengthening?
        df['bb_position_delta3']  = df['bb_position'] - df['bb_position'].shift(3)
        df['volume_ratio_delta3'] = df['volume_ratio'] - df['volume_ratio'].shift(3)

        # Momentum persistence: signed run-length of consecutive up/down candles,
        # clipped to keep the scale bounded.
        _dir = np.sign(df['returns'].fillna(0))
        _run_grp = (_dir != _dir.shift()).cumsum()
        _run_len = _dir.groupby(_run_grp).cumcount() + 1
        df['candle_streak'] = (_run_len * _dir).clip(-10, 10)
        # Fraction of the last 10 candles that closed green — bull/bear pressure.
        df['green_frac_10'] = (df['returns'] > 0).rolling(10).mean()

        # Position of close within the recent 5-bar range (short-horizon companion
        # to the 20-bar breakout_proximity above).
        _hi5 = df['high'].rolling(5).max()
        _lo5 = df['low'].rolling(5).min()
        df['range_pos_5'] = (df['close'] - _lo5) / (_hi5 - _lo5 + 1e-10)

        # Volatility regime: current vol relative to its recent norm. Lets the model
        # condition behavior on calm vs turbulent markets.
        df['vol_regime'] = df['volatility'] / (df['volatility'].rolling(50).mean() + 1e-10)

        return df

    def create_labels(self, df, forward_periods=None, threshold=0.005):
        """
        "Clean signal" labeling: only mark a trade LONG if price ends up AND
        never dips below the stop-loss during the forward window — i.e. the
        trade would not have been stopped out before reaching target. Mirrors
        for SHORT. Trains the model on trades that actually survive the exit
        rules, dramatically cutting noise vs a raw threshold filter.

        Fallback: if clean labeling produces < 50 labeled rows (tight SL / low
        volatility), revert to plain threshold labeling so the model always has
        enough training data.
        """
        tf_minutes = self._get_timeframe_minutes()
        if forward_periods is None:
            forward_periods = max(2, min(5, round(25 / tf_minutes)))

        close_arr = df['close'].values.astype(float)
        low_arr   = df['low'].values.astype(float)
        high_arr  = df['high'].values.astype(float)
        n         = len(df)
        # SL/TP are stored as % of margin; convert to % of price for label boundaries.
        # Floor at 0.5% price so that unmigrated old-format values (or high leverage)
        # never produce a boundary so tight that nothing gets labeled.
        sl_price = max(self.stop_loss_pct / max(1, self.leverage), 0.005)

        future_return  = np.full(n, np.nan)
        future_min_pct = np.full(n, np.nan)
        future_max_pct = np.full(n, np.nan)

        for i in range(n - forward_periods):
            c = close_arr[i]
            if c <= 0:
                continue
            future_return[i]  = close_arr[i + forward_periods] / c - 1
            future_min_pct[i] = low_arr[i+1:i+forward_periods+1].min() / c - 1
            future_max_pct[i] = high_arr[i+1:i+forward_periods+1].max() / c - 1

        df['future_return']  = future_return
        df['future_min_pct'] = future_min_pct
        df['future_max_pct'] = future_max_pct
        df['signal'] = 0
        df.loc[(df['future_return'] > threshold) & (df['future_min_pct'] > -sl_price), 'signal'] = 1
        df.loc[(df['future_return'] < -threshold) & (df['future_max_pct'] < sl_price), 'signal'] = -1

        # Fallback: if clean labeling is too sparse OVERALL, or one direction is
        # starved, revert to plain threshold. In trending markets clean SHORT
        # labels (price drops without ever popping above the stop) are far rarer
        # than LONG labels, so the model never learns a real short boundary and
        # the bot effectively trades long-only. Falling back when either side is
        # thin keeps both directions represented before SMOTE rebalancing.
        labeled = int((df['signal'] != 0).sum())
        n_long = int((df['signal'] == 1).sum())
        n_short = int((df['signal'] == -1).sum())
        if labeled < 50 or min(n_long, n_short) < 15:
            logger.warning(
                f"User {self.user_id}: Clean labels imbalanced "
                f"(long={n_long} short={n_short}) — falling back to plain threshold labeling"
            )
            df['signal'] = 0
            df.loc[df['future_return'] > threshold, 'signal'] = 1
            df.loc[df['future_return'] < -threshold, 'signal'] = -1

        return df

    def prepare_features(self, df):
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        return df[available].copy(), available

    # ── Model persistence & incremental learning ────────────────────────────

    def _load_persisted_model(self):
        if not os.path.exists(self._model_file):
            return
        try:
            data = pickle.load(open(self._model_file, 'rb'))
            # Feature-set guard: a model trained on a different feature set would
            # crash (or silently mispredict) when fed the current feature vector.
            # Discard and retrain if features key is missing (old model, pre-versioning)
            # OR if the stored list doesn't match the current canonical list.
            saved_features = data.get('features')
            if saved_features != FEATURE_COLUMNS:
                logger.warning(
                    f"User {self.user_id}: Persisted model feature set is stale "
                    f"({len(saved_features) if saved_features else 'unknown'} feats "
                    f"vs current {len(FEATURE_COLUMNS)}) — discarding and retraining."
                )
                return
            self.model = data['model']
            self.scaler = data['scaler']
            self.imputer = data['imputer']
            self._model_is_tree = data.get('is_tree', True)
            logger.info(f"User {self.user_id}: Loaded persisted model from {data.get('trained_at','?')}")
        except Exception as e:
            logger.warning(f"User {self.user_id}: Could not load persisted model: {e}")

    def _save_persisted_model(self):
        try:
            pickle.dump({
                'model': self.model, 'scaler': self.scaler, 'imputer': self.imputer,
                'is_tree': self._model_is_tree, 'features': FEATURE_COLUMNS,
                'trained_at': datetime.now().isoformat(),
            }, open(self._model_file, 'wb'))
        except Exception as e:
            logger.warning(f"User {self.user_id}: Could not save model: {e}")

    def _load_train_buffer(self):
        if not os.path.exists(self._buffer_file):
            return []
        try:
            return pickle.load(open(self._buffer_file, 'rb'))
        except Exception:
            return []

    def _load_feedback_buffer(self):
        fb_file = self._buffer_file.replace('.pkl', '_fb.pkl')
        if not os.path.exists(fb_file):
            return []
        try:
            return pickle.load(open(fb_file, 'rb'))
        except Exception:
            return []

    def _load_roi_history(self) -> list:
        # Always start fresh — ROI history is for the current trading session only.
        # Loading old data from previous sessions masks performance changes (e.g. old
        # profitable sim runs inflating the average while the bot is currently losing).
        return []

    def _save_buffers(self):
        try:
            with open(self._buffer_file, 'wb') as f:
                pickle.dump(self._train_buffer, f)
            fb_file = self._buffer_file.replace('.pkl', '_fb.pkl')
            with open(fb_file, 'wb') as f:
                pickle.dump(self._feedback_buffer, f)
            roi_file = self._buffer_file.replace('.pkl', '_roi.pkl')
            with open(roi_file, 'wb') as f:
                pickle.dump(self._trade_roi_history, f)
        except Exception as e:
            logger.warning(f"User {self.user_id}: Could not save buffers: {e}")

    def _add_trade_feedback(self, side: str, pnl: float, df_at_entry):
        """Store entry features from a profitable trade as confirmed training signal."""
        if pnl <= 0 or df_at_entry is None:
            return
        try:
            X_entry, feat = self.prepare_features(self.calculate_indicators(df_at_entry.copy()))
            if X_entry.empty:
                return
            row = X_entry.iloc[-1].values
            label = 1 if side == 'long' else -1
            self._feedback_buffer.append({'X': row, 'y': label, 'features': feat})
            # Keep the buffer balanced per-side. Feedback only records WINNING
            # trades, so during a long bull run it would fill with longs and
            # permanently bias the model long — even after the market turns.
            # Capping each side independently prevents that runaway skew.
            longs  = [f for f in self._feedback_buffer if f['y'] == 1][-250:]
            shorts = [f for f in self._feedback_buffer if f['y'] == -1][-250:]
            self._feedback_buffer = longs + shorts
        except Exception:
            pass

    # ── Model ───────────────────────────────────────────────────────────────

    def _build_model(self):
        """Return best available classifier: LightGBM → XGBoost → HistGradientBoosting → SVM."""
        if LGBM_AVAILABLE:
            return lgb.LGBMClassifier(**LGBM_PARAMS)
        if XGB_AVAILABLE:
            return xgb.XGBClassifier(**XGB_PARAMS)
        # HistGradientBoostingClassifier: pure numpy, no large C extensions,
        # handles NaN natively (no imputer needed), well-calibrated probabilities.
        # Far better than SVM for tabular financial data; always available via sklearn.
        return HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=0.1,
            class_weight='balanced', random_state=42,
        )

    def train_model(self, df):
        logger.info(f"User {self.user_id}: Training ML model ({_ACTIVE_ENGINE})...")
        df = self.calculate_indicators(df.copy())
        df = self.create_labels(df)
        df = df.dropna()

        if len(df) < 50:
            return False

        X_fresh, feat = self.prepare_features(df)
        y_fresh = df['signal']
        mask = y_fresh != 0
        X_fresh, y_fresh = X_fresh[mask], y_fresh[mask]

        if len(X_fresh) < 30:
            return False

        # ── Geometric accumulation: merge fresh data with buffered sessions ──
        X_parts = [X_fresh.values]
        y_parts = [y_fresh.values]
        n_feat = X_fresh.shape[1]
        decay = 0.90   # slower decay preserves cyclical crypto patterns across sessions
        for idx, session in enumerate(self._train_buffer[:10]):
            if session.get('n_feat') != n_feat:
                continue
            factor = decay ** (idx + 1)
            sX, sy = session['X'], session['y']
            keep = max(1, int(len(sX) * factor))
            idx_sub = np.random.choice(len(sX), min(keep, len(sX)), replace=False)
            X_parts.append(sX[idx_sub])
            y_parts.append(sy[idx_sub])

        # ── Trade feedback: inject confirmed real-outcome samples ──
        fb_matching = [f for f in self._feedback_buffer if len(f['X']) == n_feat]
        if fb_matching:
            X_parts.append(np.array([f['X'] for f in fb_matching]))
            y_parts.append(np.array([f['y'] for f in fb_matching]))
            logger.info(f"User {self.user_id}: +{len(fb_matching)} feedback samples from real trades")

        X_all = np.vstack(X_parts)
        y_all = np.concatenate(y_parts)

        # Refuse to train on a single class. In a one-sided trend the label set can
        # collapse to all-long (or all-short); a one-class classifier then predicts
        # that direction at a bogus 1.0 confidence on EVERY cycle (and SVC.fit even
        # raises). Skip instead — keep the prior model, or stay FLAT until both
        # directions reappear.
        if len(np.unique(y_all)) < 2:
            logger.warning(
                f"User {self.user_id}: Training set is single-class "
                f"({'long' if y_all[0] == 1 else 'short'} only) — skipping train to avoid a degenerate model"
            )
            return False

        if len(X_parts) > 1:
            logger.info(f"User {self.user_id}: Geometric training — {len(X_fresh)} fresh + {len(X_all)-len(X_fresh)} buffered = {len(X_all)} total")

        X_imp = self.imputer.fit_transform(X_all)
        X_sc = self.scaler.fit_transform(X_imp)

        y_ser = pd.Series(y_all)
        n_pos = int((y_ser == 1).sum())
        n_neg = int((y_ser == -1).sum())
        try:
            kn = min(3, n_pos - 1, n_neg - 1)
            if kn < 1:
                raise ValueError("Insufficient minority class samples for SMOTE")
            smote = SMOTE(random_state=42, k_neighbors=kn)
            X_res, y_res = smote.fit_resample(X_sc, y_ser)
        except Exception:
            X_res, y_res = X_sc, y_ser

        is_tree = LGBM_AVAILABLE or XGB_AVAILABLE or HGBT_AVAILABLE
        model = self._build_model()

        if is_tree:
            model.fit(X_res, _encode_y(y_res))
        else:
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(model, SVM_PARAMS, cv=tscv, scoring='f1_weighted', n_jobs=1)
            grid.fit(X_res, y_res)
            model = grid.best_estimator_

        self.model = model
        self._model_is_tree = is_tree

        X_check = self.scaler.transform(self.imputer.transform(X_fresh.values))
        raw_pred = model.predict(X_check)
        y_pred = _decode_y(raw_pred) if is_tree else raw_pred
        accuracy = accuracy_score(y_fresh, y_pred)
        logger.info(f"User {self.user_id}: Model trained — Accuracy: {accuracy:.2%}")

        # Save this session to buffer (prepend = newest first) and persist
        self._train_buffer.insert(0, {'X': X_fresh.values, 'y': y_fresh.values, 'n_feat': n_feat})
        self._train_buffer = self._train_buffer[:10]
        self._save_buffers()
        self._save_persisted_model()
        return True

    def predict_signal(self, df):
        if self.model is None:
            logger.warning(f"User {self.user_id}: Model not trained yet — returning FLAT signal. Bot will train on next cycle.")
            return 0, 0.5
        df = self.calculate_indicators(df.copy())
        X, _ = self.prepare_features(df)
        if X.empty:
            return 0, 0.5
        X_latest = X.iloc[[-1]]
        X_imp = self.imputer.transform(X_latest)
        X_sc = self.scaler.transform(X_imp)
        raw = self.model.predict(X_sc)[0]
        pred = int(_decode_y([raw])[0]) if self._model_is_tree else int(raw)
        conf = float(max(self.model.predict_proba(X_sc)[0]))
        return pred, conf

    def _ensemble_signal(self, symbol: str, df, ml_signal: int, ml_conf: float):
        """
        Blend ML prediction with SignalEngine second opinion.
        Agreement boosts confidence by up to 20%; disagreement cuts it by 30%.
        Returns (final_signal, final_confidence).
        """
        if self._signal_engine is None:
            return ml_signal, ml_conf

        try:
            se_df = self._signal_engine.calculate_indicators(df.copy())
            sig = self._signal_engine.generate_signal(symbol, se_df)
            if sig is None:
                return ml_signal, ml_conf

            se_signal = 1 if sig.signal_type == 'bullish' else -1
            se_conf = sig.confidence / 100.0  # normalise to 0-1

            if se_signal == ml_signal:
                # Both agree — weighted average + agreement bonus
                blended = (ml_conf * 0.6 + se_conf * 0.4) * 1.2
                return ml_signal, min(0.99, blended)
            else:
                # Disagreement — take ML signal but reduce conviction
                return ml_signal, ml_conf * 0.70
        except Exception as e:
            logger.debug(f"User {self.user_id}: SignalEngine blend failed: {e}")
            return ml_signal, ml_conf

    # ── Setup detectors (second signal stream) ──────────────────────────────

    def detect_setups(self, df):
        """
        Deterministic technical setups evaluated on the LAST CLOSED candle
        (df.iloc[-2] — the last row is the still-forming candle and its close/
        low/high are not final). Returns a list of {'signal', 'confidence',
        'name'} dicts. Each setup encodes a classic, well-documented pattern:

          rsi_bounce    — oversold RSI hooking upward off a stabilizing low
          rsi_fade      — overbought RSI rolling over from a high
          divergence    — price lower-low with RSI higher-low (bullish) / mirror
          breakout      — new 20-bar high/low on strong volume, trend confirmed
          shakeout      — stop-hunt wick through the 20-bar range edge that
                          closes back inside on volume (spring / upthrust)

        All conditions use indicator columns calculate_indicators() computed.
        """
        setups = []
        if df is None or len(df) < 30:
            return setups
        hist = df.iloc[:-1]          # closed candles only
        row = hist.iloc[-1]          # last closed candle
        prev = hist.iloc[-2]

        def g(r, col, default):
            v = r.get(col, default)
            return default if pd.isna(v) else float(v)

        rsi = g(row, 'rsi', 50); rsi_prev2 = g(hist.iloc[-3], 'rsi', 50)
        vol = g(row, 'volume_ratio', 1.0)
        rng5 = g(row, 'range_pos_5', 0.5)
        ema_dist = g(row, 'ema200_distance', 0.0)
        adx = g(row, 'adx', 20)
        close = g(row, 'close', 0); prev_close = g(prev, 'close', 0)
        vol_bonus = 0.04 if vol >= 2.0 else (0.02 if vol >= 1.5 else 0.0)

        try:
            # 1. RSI oversold bounce (long) — hooked up off the low, not a knife
            if (rsi < 32 and rsi > rsi_prev2 + 1.5 and rng5 > 0.25
                    and ema_dist > -0.05 and close > prev_close):
                setups.append({'signal': 1, 'confidence': 0.74 + vol_bonus, 'name': 'rsi_bounce'})
            # RSI overbought fade (short)
            if (rsi > 68 and rsi < rsi_prev2 - 1.5 and rng5 < 0.75
                    and ema_dist < 0.05 and close < prev_close):
                setups.append({'signal': -1, 'confidence': 0.74 + vol_bonus, 'name': 'rsi_fade'})
        except Exception:
            pass

        try:
            # 2. RSI divergence over the last 20 closed candles
            seg = hist.tail(20)
            if len(seg) >= 20 and seg['rsi'].notna().all():
                lo_r, lo_p = seg['low'].iloc[-7:], seg['low'].iloc[:-7]
                rs_r, rs_p = seg['rsi'].iloc[-7:], seg['rsi'].iloc[:-7]
                hi_r, hi_p = seg['high'].iloc[-7:], seg['high'].iloc[:-7]
                # Bullish: price lower low, RSI higher low, bounce started
                if (lo_r.min() < lo_p.min() and rs_r.min() > rs_p.min() + 2
                        and rsi < 45 and close > prev_close):
                    setups.append({'signal': 1, 'confidence': 0.78 + vol_bonus, 'name': 'bull_divergence'})
                # Bearish: price higher high, RSI lower high, rollover started
                if (hi_r.max() > hi_p.max() and rs_r.max() < rs_p.max() - 2
                        and rsi > 55 and close < prev_close):
                    setups.append({'signal': -1, 'confidence': 0.78 + vol_bonus, 'name': 'bear_divergence'})
        except Exception:
            pass

        try:
            # 3. Breakout — new 20-bar extreme on real volume with trend strength
            if g(row, 'is_new_high', 0) >= 1 and vol >= 1.5 and adx >= 18:
                setups.append({'signal': 1, 'confidence': 0.76 + vol_bonus, 'name': 'breakout_long'})
            if g(row, 'is_new_low', 0) >= 1 and vol >= 1.5 and adx >= 18:
                setups.append({'signal': -1, 'confidence': 0.76 + vol_bonus, 'name': 'breakout_short'})
        except Exception:
            pass

        try:
            # 4. Shakeout — wick through the prior range edge, close back inside
            dc_low_prev = g(prev, 'donchian_low', float('nan'))
            dc_high_prev = g(prev, 'donchian_high', float('nan'))
            if not math.isnan(dc_low_prev) and vol >= 1.3:
                if g(row, 'low', 0) < dc_low_prev and close > dc_low_prev:
                    setups.append({'signal': 1, 'confidence': 0.78 + vol_bonus, 'name': 'shakeout_spring'})
            if not math.isnan(dc_high_prev) and vol >= 1.3:
                if g(row, 'high', 0) > dc_high_prev and close < dc_high_prev:
                    setups.append({'signal': -1, 'confidence': 0.78 + vol_bonus, 'name': 'shakeout_upthrust'})
        except Exception:
            pass

        return setups

    def _best_setup(self, symbol, df, ml_signal, ml_conf):
        """
        Resolve fired setups against the ML opinion. Returns (signal, confidence,
        name) or (0, 0.0, None).
          • Conflicting directions in the same candle → stand down.
          • ML strongly disagrees (opposite at ≥80%) → veto.
          • ML agrees → +0.08 confidence boost (pushes the trade into a higher
            dynamic-leverage tier — patterns confirmed by the model ride bigger).
          • Multiple same-direction setups stack +0.03 each.
        """
        fired = self.detect_setups(df)
        if not fired:
            return 0, 0.0, None
        longs  = [s for s in fired if s['signal'] == 1]
        shorts = [s for s in fired if s['signal'] == -1]
        if longs and shorts:
            return 0, 0.0, None
        group = longs or shorts
        sig = group[0]['signal']
        if ml_signal == -sig and ml_conf >= 0.80:
            logger.info(f"User {self.user_id}: [{symbol}] Setup vetoed — ML opposes at {ml_conf:.0%}")
            return 0, 0.0, None
        conf = max(s['confidence'] for s in group) + 0.03 * (len(group) - 1)
        if ml_signal == sig:
            conf += 0.08
        conf = min(0.93, conf)
        name = '+'.join(s['name'] for s in group)
        return sig, conf, name

    # ── Risk helpers ────────────────────────────────────────────────────────

    def _get_drawdown_recovery_scale(self) -> float:
        """
        Graduated position-size scale based on current drawdown from peak.
        Shrinks bets as losses mount so the account can recover faster.
          DD ≤  5% → 1.00× (no penalty)
          DD ≤ 10% → 0.75×
          DD ≤ 15% → 0.50×
          DD ≤ 20% → 0.25×
          DD >  20% → 0.00× (hard stop — handled by _is_drawdown_exceeded)
        """
        baseline = max(self._drawdown_baseline or self.starting_balance, self._peak_balance)
        dd = (baseline - self.balance) / max(baseline, 1)
        if dd <= 0.05:
            return 1.00
        if dd <= 0.10:
            return 0.75
        if dd <= 0.15:
            return 0.50
        return 0.25

    def _check_performance_floor(self) -> bool:
        """
        Auto-pause new entries when the last 20 closed trades have < 40% win rate.
        Requires 20 samples to avoid false triggers early in a session.
        Returns True (= block entry) if performance is too poor to risk capital.
        """
        recent = [t for t in self.trades_history if t.get('type') == 'close' and 'pnl' in t][-20:]
        if len(recent) < 20:
            return False
        wins = sum(1 for t in recent if t['pnl'] > 0)
        wr = wins / len(recent)
        if wr < 0.40:
            logger.warning(
                f"User {self.user_id}: Win rate {wr:.0%} over last 20 trades < 40% floor — pausing new entries"
            )
            return True
        return False

    def _get_adaptive_scale(self) -> float:
        """
        Real-time aggression throttle from recent realized performance. Leans the
        bot in when its recent track record is strong and pulls back when it
        degrades, so it stays in an ideal compounding posture as conditions shift —
        the "lean into what's working" half of the loop (drawdown scale handles the
        capital-protection half). Bounded [0.75, 1.30]; neutral until enough data.
        """
        closes = [t for t in self.trades_history if t.get('type') == 'close' and 'pnl' in t][-20:]
        if len(closes) < 8:
            return 1.0  # too few samples — stay neutral
        wins = sum(1 for t in closes if t['pnl'] > 0)
        wr = wins / len(closes)
        recent_roi = self._trade_roi_history[-20:]
        avg_roi = sum(recent_roi) / len(recent_roi) if recent_roi else 0.0

        scale = 1.0
        if   wr >= 0.65: scale += 0.20
        elif wr >= 0.55: scale += 0.10
        elif wr <  0.45: scale -= 0.15
        elif wr <  0.50: scale -= 0.08

        if   avg_roi > 0.01: scale += 0.10
        elif avg_roi < 0.0:  scale -= 0.10

        return max(0.75, min(1.30, scale))

    def _is_drawdown_exceeded(self):
        baseline = self._drawdown_baseline or self.starting_balance
        drawdown = (baseline - self.balance) / baseline
        if drawdown > self.max_drawdown_pct:
            logger.warning(
                f"User {self.user_id}: Drawdown {drawdown:.1%} > max {self.max_drawdown_pct:.1%} — halting entries"
            )
            return True
        return False

    def _reset_daily_if_needed(self):
        """Reset daily P&L governor at UTC midnight."""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        if today != self._day_date:
            self._day_date = today
            self._day_pnl = 0.0
            self._day_start_balance = self.balance

    def _check_daily_governor(self) -> bool:
        """
        Returns True (block entry) if:
          • Daily loss exceeds daily_loss_limit (e.g. −8% of day-start balance).
        Give-back protection: if today is already up >30%, halve max_positions so
        we don't bet a great day away — still trades, just less aggressively.
        """
        if self._day_start_balance <= 0:
            return False
        day_pct = self._day_pnl / self._day_start_balance
        if day_pct <= -self.daily_loss_limit:
            logger.warning(
                f"User {self.user_id}: Daily loss limit hit ({day_pct:.1%}) — "
                f"halting new entries for today"
            )
            return True
        return False

    def _daily_governor_max_pos(self) -> int:
        """When today is up >30%, be more conservative to protect gains."""
        if self._day_start_balance > 0 and self._day_pnl / self._day_start_balance > 0.30:
            return max(1, self.max_positions // 2)
        return self.max_positions

    def _realized_win_loss(self):
        """Average realized winner and loser (as fraction of margin) from the last
        30 closed trades. Returns (avg_win, avg_loss, n_closed). The realized
        numbers reflect the ACTUAL exit engine (trail/breakeven/reversal), which is
        what the nominal TP/SL settings do not."""
        closed = [t for t in self.trades_history
                  if t.get('type') == 'close' and t.get('pnl_pct') is not None][-30:]
        wins   = [t['pnl_pct'] / 100 for t in closed if t['pnl_pct'] > 0]
        losses = [abs(t['pnl_pct']) / 100 for t in closed if t['pnl_pct'] < 0]
        avg_win  = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        return avg_win, avg_loss, len(closed)

    def _entry_ev(self, confidence: float) -> float:
        """
        Expected value of the next trade as a fraction of margin, after fees.
          EV = p × net_win − (1−p) × net_loss

        Once ≥10 trades are on record, win/loss sizes come from REALIZED history —
        the actual exit engine's captures — so the gate self-calibrates to what the
        bot really earns per winner and loses per loser. Before that, the nominal
        TP is capped at 25% margin: with TP set to 100% (which trailing/reversal
        exits mean it never reaches), the uncapped formula passed any confidence
        above ~14%, making the gate decorative.
        """
        p = max(0.01, min(0.99, float(confidence)))
        fee_m = (MAKER_FEE + TAKER_FEE) * max(1, self.leverage)

        avg_win, avg_loss, n = self._realized_win_loss()
        if n >= 10 and avg_win > 0 and avg_loss > 0:
            # Realized pnl_pct is already net of fees
            return p * avg_win - (1 - p) * avg_loss

        net_win  = min(self.take_profit_pct, 0.25) - fee_m
        net_loss = self.stop_loss_pct + fee_m
        return p * net_win - (1 - p) * net_loss

    def _kelly_fraction(self) -> float:
        """
        Half-Kelly sizing from the last 50 closed trades.
        Falls back to self.risk_per_trade until 20 trades are on record.
        Hard bounds: 0.5% – 5% of balance per trade.
        """
        closed = [t for t in self.trades_history if t.get('type') == 'close' and 'pnl' in t][-50:]
        if len(closed) < 20:
            return self.risk_per_trade

        wins = [t['pnl'] for t in closed if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in closed if t['pnl'] < 0]

        if not wins or not losses:
            return self.risk_per_trade

        p = len(wins) / len(closed)
        b = (sum(wins) / len(wins)) / (sum(losses) / len(losses))
        kelly = (p * b - (1 - p)) / b
        return max(0.001, min(self.risk_per_trade, kelly * 0.5))

    def _dynamic_leverage(self, confidence: float) -> int:
        """
        Scale leverage up/down based on signal confidence.
        No-brainer trades (high conf) use full user-set leverage.
        Borderline trades use a fraction — smaller position, same margin.
        """
        base = self.leverage
        if confidence >= 0.85:
            scale = 1.00   # no-brainer — full throttle
        elif confidence >= 0.75:
            scale = 0.85   # strong signal
        elif confidence >= 0.65:
            scale = 0.65   # moderate — be measured
        else:
            scale = 0.50   # borderline — minimal
        result = max(3, round(base * scale))
        self._last_dynamic_leverage = result
        return result

    def _kelly_fraction_calibrated(self, confidence: float) -> float:
        """
        True half-Kelly using calibrated win probability from predict_proba.
        kelly = p − (1−p) × (net_loss / net_win)
        Half-Kelly halves the bet for practical safety (variance reduction).
        Bounded to [risk_per_trade×0.25 .. risk_per_trade×3].
        """
        p = max(0.05, min(0.95, float(confidence)))
        fee_m = (MAKER_FEE + TAKER_FEE) * max(1, self.leverage)
        # Cap nominal TP at a realistic capture — trailing/breakeven exits mean a
        # 100% TP never hits, and an inflated net_win overstates Kelly.
        net_win  = max(0.001, min(self.take_profit_pct, 0.25) - fee_m)
        net_loss = self.stop_loss_pct + fee_m
        kelly = p - (1 - p) * (net_loss / net_win)
        half_kelly = kelly * 0.5
        return max(self.risk_per_trade * 0.25, min(self.risk_per_trade * 3.0, half_kelly))

    def _calculate_margin(self, confidence: float = None) -> float:
        """
        Profit-tier compounding using calibrated half-Kelly sizing.
          • Base capital: half-Kelly sized to calibrated win probability.
          • Profit tier: gains above starting_balance compounded at profit_risk_multiplier×.
        Hard cap: min(max(risk_per_trade×2, 10%), 75%) of current balance.
        """
        conf = confidence if (confidence is not None and confidence == confidence) else self.min_confidence
        conf = max(0.0, min(1.0, conf))
        k = self._kelly_fraction_calibrated(conf) if conf > 0 else self._kelly_fraction()

        profit = max(0.0, self.balance - self.starting_balance)
        base_cap = min(self.balance, self.starting_balance)
        base_margin = base_cap * k
        profit_margin = profit * k * self.profit_risk_multiplier
        margin = base_margin + profit_margin

        # Real-time posture: drawdown scale protects capital (≤1×), adaptive scale
        # leans into a strong recent track record (up to 1.3×). Applied before the
        # hard ceiling so the boost can never push risk past the configured cap.
        dd_scale = self._get_drawdown_recovery_scale()
        adaptive_scale = self._get_adaptive_scale()
        margin = margin * dd_scale * adaptive_scale

        # Hard ceiling LAST so no upstream multiplier can exceed it. Respects the
        # user's risk setting: ceiling = risk_per_trade × 2.0 (headroom for
        # profit-tier + confidence + lean-in), 10% floor, 75% hard ceiling.
        risk_ceiling = min(max(self.risk_per_trade * 2.0, 0.10), 0.75)
        final = min(margin, self.balance * risk_ceiling)

        # Track peak balance for compound projection accuracy
        self._peak_balance = max(self._peak_balance, self.balance)

        conf_str = f"{confidence:.2f}" if confidence is not None else "n/a"
        logger.info(
            f"User {self.user_id}: [MARGIN] base=${base_cap:.2f} k={k*100:.1f}% | "
            f"profit_tier=${profit:.2f}×{self.profit_risk_multiplier} | "
            f"conf={conf_str} kelly={k:.4f} dd={dd_scale:.2f} adapt={adaptive_scale:.2f} → ${final:.2f}"
        )
        return max(1.0, final)

    def _get_volatility_multiplier(self, df) -> float:
        """Scale size inversely to ATR: high vol → 0.5×, low vol → 1.5×. Baseline 0.15% ATR/price."""
        if 'atr' not in df.columns or df['atr'].isna().all():
            return 1.0
        atr = df['atr'].iloc[-1]
        price = df['close'].iloc[-1]
        if price <= 0 or atr <= 0:
            return 1.0
        return max(0.5, min(1.5, 0.0015 / (atr / price + 1e-10)))

    def _entry_filter(self, signal: int, df) -> bool:
        """
        Pre-entry quality gate — thresholds match the backtest filter exactly so
        optimizer results translate 1:1 to live performance.
          • ADX ≥ adx_threshold — meaningful directional momentum (matches backtest)
          • volume_ratio ≥ 0.65 — solid participation (matches backtest)
        Returns True = entry allowed.
        """
        if df is None or len(df) < 5:
            return True
        latest = df.iloc[-1]
        adx_raw = latest.get('adx', 25)
        vol_raw = latest.get('volume_ratio', 1.0)
        adx = 0.0  if pd.isna(adx_raw) else float(adx_raw)  # NaN → 0 blocks entry (safe default)
        vol = 1.0  if pd.isna(vol_raw) else float(vol_raw)
        if adx < self.adx_threshold:
            logger.info(f"User {self.user_id}: Entry blocked — ADX {adx:.1f} < {self.adx_threshold} (weak trend)")
            return False
        if vol < 0.65:
            logger.info(f"User {self.user_id}: Entry blocked — volume_ratio {vol:.2f} < 0.65 (thin volume)")
            return False
        # Session quality: 02:00–08:00 UTC = Asia-Pac/US gap, historically thin liquidity.
        # Reduce to 50% volume threshold requirement; don't block entirely so on strong
        # vol spikes (news) we still catch the move.
        utc_hour = datetime.utcnow().hour
        if 2 <= utc_hour < 8 and vol < 0.85:
            logger.info(f"User {self.user_id}: Entry skipped — low-liquidity session (UTC {utc_hour:02d}h) vol={vol:.2f}")
            return False
        return True

    def _get_market_regime(self) -> str:
        """
        Detect macro BTC trend from 4h candles (50-period EMA slope + price position).
        Independent of trading timeframe. Cached across cycles.
        """
        try:
            btc_df = self.fetch_ohlcv(symbol='BTC/USDT:USDT', limit=60, timeframe='4h')
            if btc_df is None or len(btc_df) < 50:
                return 'sideways'
            close = btc_df['close'].values.astype(float)
            # Cross-market confirmation input: BTC's own recent 5-bar return.
            # Cached so per-symbol observations can record whether the traded coin
            # is moving with the market or against it.
            try:
                if len(close) >= 6 and close[-6] > 0:
                    self._btc_ret_5 = float(close[-1] / close[-6] - 1)
            except Exception:
                pass
            ema50 = talib.EMA(close, timeperiod=50)
            if np.isnan(ema50[-1]) or np.isnan(ema50[-10]):
                return 'sideways'
            price_vs_ema = (close[-1] - ema50[-1]) / ema50[-1]
            ema_slope = (ema50[-1] - ema50[-10]) / ema50[-10]
            if price_vs_ema > 0.02 and ema_slope > 0.001:
                return 'bull'
            if price_vs_ema < -0.02 and ema_slope < -0.001:
                return 'bear'
            return 'sideways'
        except Exception as e:
            logger.debug(f"User {self.user_id}: Regime detection failed: {e}")
            return 'sideways'

    def _get_regime_multiplier(self, signal: int) -> float:
        """
        Align position size with macro trend.
          Bull: longs get +20%, counter-trend shorts get −60%.
          Bear: shorts get +20%, counter-trend longs get −60%.
          Sideways: −20% across the board (lower conviction).
        """
        r = self.market_regime
        if r == 'bull':
            return 1.2 if signal == 1 else 0.4
        if r == 'bear':
            return 1.2 if signal == -1 else 0.4
        return 0.8

    def analyze_market_direction(self, coins=None):
        """
        Multi-timeframe directional bias scan for the chosen tokens. Read-only and
        ML-free — a fast "where is each token heading" overview to guide the
        optimize → backtest → live workflow and reveal long vs short opportunities.

        Each token is sampled on three horizons (15m, 1h, 4h, macro weighted
        highest). Per horizon we vote on trend (price vs EMA50 + EMA50 slope) and
        momentum (RSI + MACD histogram), and read ADX for trend strength. Voting
        confluence across horizons yields a directional label and a trade bias.
        """
        coins = coins or self.selected_coins or ['BTC/USDT:USDT']
        timeframes = [('15m', 1.0), ('1h', 1.5), ('4h', 2.0)]

        # Refresh macro regime so the summary reflects current conditions.
        try:
            self.market_regime = self._get_market_regime()
        except Exception:
            pass

        results = []
        for symbol in coins:
            tf_views, adx_vals = [], []
            weighted_score = weight_total = 0.0

            for tf, weight in timeframes:
                try:
                    df = self.fetch_ohlcv(symbol=symbol, limit=250, timeframe=tf)
                    if df is None or len(df) < 60:
                        continue
                    close = df['close'].values.astype(float)
                    high  = df['high'].values.astype(float)
                    low   = df['low'].values.astype(float)

                    ema50 = talib.EMA(close, timeperiod=50)
                    rsi   = talib.RSI(close, timeperiod=14)
                    adx   = talib.ADX(high, low, close, timeperiod=14)
                    _, _, macd_hist = talib.MACD(close)

                    price    = close[-1]
                    ema_now  = ema50[-1]
                    ema_prev = ema50[-10] if not np.isnan(ema50[-10]) else ema_now
                    price_vs_ema = (price - ema_now) / ema_now if ema_now else 0.0
                    ema_slope    = (ema_now - ema_prev) / ema_prev if ema_prev else 0.0
                    rsi_now  = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0
                    adx_now  = float(adx[-1]) if not np.isnan(adx[-1]) else 0.0
                    hist_now = float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else 0.0

                    score = 0
                    if price_vs_ema >  0.003: score += 1
                    elif price_vs_ema < -0.003: score -= 1
                    if ema_slope >  0.0005: score += 1
                    elif ema_slope < -0.0005: score -= 1
                    if rsi_now > 55: score += 1
                    elif rsi_now < 45: score -= 1
                    if hist_now > 0: score += 1
                    elif hist_now < 0: score -= 1
                    tf_dir = score / 4.0

                    adx_vals.append(adx_now)
                    weighted_score += tf_dir * weight
                    weight_total   += weight
                    tf_views.append({
                        'timeframe': tf,
                        'direction': 'bullish' if tf_dir > 0.1 else ('bearish' if tf_dir < -0.1 else 'neutral'),
                        'score': round(tf_dir, 2),
                        'rsi': round(rsi_now, 1),
                        'adx': round(adx_now, 1),
                        'price_vs_ema50_pct': round(price_vs_ema * 100, 2),
                    })
                except Exception as e:
                    logger.debug(f"User {self.user_id}: Direction scan {symbol} {tf} failed: {e}")
                    continue

            if weight_total == 0:
                results.append({'symbol': symbol, 'error': 'no_data'})
                continue

            conviction = weighted_score / weight_total
            avg_adx = sum(adx_vals) / len(adx_vals) if adx_vals else 0.0
            nonzero = [1 if v['score'] > 0.1 else (-1 if v['score'] < -0.1 else 0) for v in tf_views]
            active = [d for d in nonzero if d != 0]
            aligned = bool(active) and len(set(active)) == 1

            if conviction >= 0.5:    label = 'STRONG BULLISH'
            elif conviction >= 0.15: label = 'BULLISH'
            elif conviction <= -0.5: label = 'STRONG BEARISH'
            elif conviction <= -0.15: label = 'BEARISH'
            else:                    label = 'NEUTRAL / RANGE'

            trend_ok = avg_adx >= self.adx_threshold
            if not trend_ok or label.startswith('NEUTRAL'):
                bias, tradeable = 'range — wait / mean-revert', False
            elif conviction > 0:
                bias, tradeable = 'long-favored', True
            else:
                bias, tradeable = 'short-favored', True

            results.append({
                'symbol': symbol,
                'label': label,
                'conviction': round(conviction, 3),
                'avg_adx': round(avg_adx, 1),
                'aligned': aligned,
                'trend_strength': 'strong' if avg_adx >= 25 else ('moderate' if trend_ok else 'weak'),
                'recommended_bias': bias,
                'tradeable': tradeable,
                'timeframes': tf_views,
            })

        results.sort(key=lambda r: abs(r.get('conviction', 0)), reverse=True)
        longs  = sum(1 for r in results if r.get('conviction', 0) > 0.15)
        shorts = sum(1 for r in results if r.get('conviction', 0) < -0.15)
        return {
            'scanned_at': datetime.now().isoformat(),
            'market_regime': self.market_regime,
            'adx_threshold': self.adx_threshold,
            'summary': {
                'tokens': len(results),
                'bullish': longs,
                'bearish': shorts,
                'neutral': len(results) - longs - shorts,
            },
            'tokens': results,
        }

    def _sync_live_balance(self):
        """Delta-based balance sync — applies only the change the exchange reports, not the
        absolute total. This lets the user allocate a subset of their Blofin account (e.g.
        $100 of a $503 account) and have the dashboard track just those funds."""
        if self.simulation_mode or self.exchange is None:
            return
        try:
            account = self.exchange.fetch_balance()
            total = float(account.get('USDT', {}).get('total', 0))
            if total <= 0:
                return
            if self._drawdown_baseline is None:
                self._drawdown_baseline = self.balance
                logger.info(f"User {self.user_id}: Drawdown baseline set to ${self.balance:.2f}")
            if self._last_exchange_total is None:
                # First sync after startup — anchor to exchange without changing tracked balance
                self._last_exchange_total = total
                logger.info(f"User {self.user_id}: Exchange anchored at ${total:.2f} | tracking ${self.balance:.2f}")
                return
            delta = total - self._last_exchange_total
            if abs(delta) > 0.005:
                old = self.balance
                self.balance = max(0.0, self.balance + delta)
                self._last_exchange_total = total
                logger.info(f"User {self.user_id}: Balance ${old:.2f} → ${self.balance:.2f} (Δ${delta:+.2f})")
        except Exception as e:
            logger.warning(f"User {self.user_id}: Balance sync failed: {e}")

    # ── Status ──────────────────────────────────────────────────────────────

    def _record_trade_roi(self, pnl: float):
        """Record per-trade ROI as fraction of balance BEFORE this trade closes, then persist."""
        if self.balance > 0:
            roi = pnl / self.balance
            self._trade_roi_history.append(roi)
            if len(self._trade_roi_history) > 200:
                self._trade_roi_history = self._trade_roi_history[-200:]
            self._trades_since_roi_save += 1
            if self._trades_since_roi_save >= 5:
                self._save_buffers()
                self._trades_since_roi_save = 0

    def _compound_projection(self):
        """
        Return compound metrics based on actual realized per-trade ROI.
        Uses last 20 closed trades to compute average ROI per trade, then
        projects trades/days to reach milestone targets from current balance.
        """
        recent = self._trade_roi_history[-20:] if self._trade_roi_history else []
        # Need at least 10 trades before the projection is meaningful.
        if len(recent) < 10:
            return {'insufficient_data': True, 'sample_size': len(recent)}

        avg_roi = sum(recent) / len(recent)  # average ROI per trade (signed fraction)

        # Reality check: if the bot is net-losing this session, don't project
        # a positive path to $1M — that's misleading. Show warning instead.
        session_pnl = self.balance - self.starting_balance
        if avg_roi <= 0 or session_pnl < 0:
            return {
                'avg_trade_roi_pct': round(avg_roi * 100, 3),
                'sample_size': len(recent),
                'warning': True,
                'session_pnl': round(session_pnl, 2),
            }

        hours_running = (time.time() - self._start_time) / 3600
        trades_per_day = self.total_trades / max(1, hours_running / 24)

        def trades_to(target):
            try:
                if self.balance >= target:
                    return 0
                return math.ceil(math.log(target / max(self.balance, 1)) / math.log(1 + avg_roi))
            except (ValueError, ZeroDivisionError):
                return None

        def days_to(trades):
            if trades is None or trades <= 0:
                return None
            rate = max(trades_per_day, 0.5)  # assume at least 0.5 trades/day floor
            return round(trades / rate, 1)

        t10k  = trades_to(10_000)
        t100k = trades_to(100_000)
        t1m   = trades_to(1_000_000)

        return {
            'avg_trade_roi_pct': round(avg_roi * 100, 3),
            'sample_size': len(recent),
            'trades_to_10k':  t10k,
            'trades_to_100k': t100k,
            'trades_to_1m':   t1m,
            'days_to_10k':    days_to(t10k),
            'days_to_100k':   days_to(t100k),
            'days_to_1m':     days_to(t1m),
            'trades_per_day': round(trades_per_day, 2),
        }

    def apply_optimizer_config(self, config: dict):
        """
        Hot-apply a best config from an optimizer run to this live/sim bot instance.
        Clears the OHLCV cache on timeframe change so the next cycle fetches fresh data.
        """
        changed = []
        if 'leverage' in config:
            self.leverage = int(config['leverage'])
            self._last_dynamic_leverage = self.leverage
            changed.append(f"leverage={self.leverage}x")
        if 'timeframe' in config and config['timeframe'] in VALID_TIMEFRAMES:
            if config['timeframe'] != self.timeframe:
                self.timeframe = config['timeframe']
                self._ohlcv_cache.clear()   # force fresh fetch on new TF
                changed.append(f"timeframe={self.timeframe}")
        if 'risk_per_trade' in config:
            self.risk_per_trade = float(config['risk_per_trade'])
            changed.append(f"risk={self.risk_per_trade:.1%}")
        if 'stop_loss_pct' in config:
            self.stop_loss_pct = float(config['stop_loss_pct'])
            changed.append(f"sl={self.stop_loss_pct:.1%}")
        if 'take_profit_pct' in config:
            self.take_profit_pct = float(config['take_profit_pct'])
            changed.append(f"tp={self.take_profit_pct:.1%}")
        if 'trailing_stop_pct' in config:
            self.trailing_stop_pct = float(config['trailing_stop_pct'])
            changed.append(f"trail={self.trailing_stop_pct:.1%}")
        if 'min_confidence' in config:
            self.min_confidence = float(config['min_confidence'])
            changed.append(f"min_conf={self.min_confidence:.0%}")
        if 'adx_threshold' in config:
            self.adx_threshold = max(5, int(config['adx_threshold']))
            changed.append(f"adx={self.adx_threshold}")
        if 'profit_risk_multiplier' in config:
            self.profit_risk_multiplier = float(config['profit_risk_multiplier'])
            changed.append(f"prm={self.profit_risk_multiplier}")
        if 'trade_cooldown' in config:
            self.trade_cooldown = int(config['trade_cooldown'])
            changed.append(f"cooldown={self.trade_cooldown}s")
        if 'daily_loss_limit' in config:
            self.daily_loss_limit = max(0.01, min(0.50, float(config['daily_loss_limit'])))
            changed.append(f"dayloss={self.daily_loss_limit:.0%}")
        if 'max_positions' in config:
            self.max_positions = max(1, int(config['max_positions']))
            changed.append(f"maxpos={self.max_positions}")
        logger.info(f"User {self.user_id}: [OPTIMIZER] Config applied: {', '.join(changed) or 'no changes'}")
        return changed

    def get_status(self):
        coin_signals = {}
        for sig in self.signals_history:
            sym = sig.get('symbol', '')
            coin = sym.split('/')[0] if sym else 'Unknown'
            coin_signals[coin] = sig

        open_positions = list(self.positions.values())

        # ── Compound velocity metrics ──────────────────────────────────────
        hours_running = (time.time() - self._start_time) / 3600
        trades_per_day = self.total_trades / max(1, hours_running / 24)
        daily_pnl_pct = (self.balance / max(self.starting_balance, 1) - 1) * 100

        compound = self._compound_projection()
        projected_days_to_1m = compound.get('days_to_1m') if compound else None

        return {
            'running': self.running,
            'simulation_mode': self.simulation_mode,
            'balance': float(self.balance),
            'starting_balance': float(self.starting_balance),
            'leverage': self.leverage,
            'selected_coins': self.selected_coins,
            'risk_per_trade': self.risk_per_trade,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'trailing_stop_pct': self.trailing_stop_pct,
            'max_drawdown_pct': self.max_drawdown_pct,
            'trade_cooldown': self.trade_cooldown,
            'min_confidence': self.min_confidence,
            'adx_threshold': self.adx_threshold,
            'market_regime': self.market_regime,
            'model_type': _ACTIVE_ENGINE,
            'signal_engine_active': bool(SIGNAL_ENGINE_AVAILABLE),
            'total_pnl': float(self.total_pnl),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'kelly_fraction': self._kelly_fraction(),
            'positions': open_positions,
            'position': open_positions[0] if open_positions else None,
            'last_signals': self.signals_history[-10:] if self.signals_history else [],
            'coin_signals': coin_signals,
            'recent_trades': self.trades_history[-10:] if self.trades_history else [],
            # Compound velocity
            'trades_per_day': round(trades_per_day, 2),
            'daily_pnl_pct': round(daily_pnl_pct, 4),
            'projected_days_to_1m': projected_days_to_1m,
            'compound': compound,
            # Dynamic sizing
            'last_dynamic_leverage': self._last_dynamic_leverage,
            'adaptive_scale': round(self._get_adaptive_scale(), 2),
            # Autonomous self-improvement — this replaces the old parameter-search
            # "autopilot" (which periodically re-ran a curve-fitting grid search and
            # silently hot-applied the winner over SL/TP/leverage/confidence). That
            # mechanism is gone: it optimized for backtested ROI, which is precisely
            # the overfitting risk the Market Intelligence Engine exists to avoid.
            # What replaces it is honest about being narrower — MIE continuously
            # records state and re-validates its own models against unseen data on
            # a fixed schedule (see _maybe_refit_mie), but it NEVER changes the
            # user's configured risk parameters. It can only affect entries, via
            # the gate, and only once a model has actually earned that trust.
            'mie_last_fit_hours_ago': (
                round((time.time() - self._mie_last_fit) / 3600, 1) if self._mie_last_fit else None
            ),
            'mie_fit_interval_hours': round(self._mie_fit_interval / 3600, 1),
            # Daily P&L governor
            'day_pnl': round(float(self._day_pnl), 4),
            'day_pnl_pct': round(self._day_pnl / max(self._day_start_balance, 1) * 100, 2),
            'daily_loss_limit': self.daily_loss_limit,
            'max_positions': self.max_positions,
            # Market Intelligence Engine
            'mie_available': bool(MIE_AVAILABLE and self._mie is not None),
            'mie_gate_enabled': self.mie_gate_enabled,
            'mie_any_validated': bool(self._mie.any_validated) if self._mie is not None else False,
            'mie_resolved_observations': self._mie_store.count_resolved() if self._mie_store is not None else 0,
            'mie_decisions': {sym: {
                'action': d.action, 'quality': d.quality, 'regime': d.regime,
                'expected_return': round(d.expected_return, 5), 'expectancy_r': round(d.expectancy_r, 4),
                'historical_sample': d.historical_sample, 'blockers': d.blockers, 'reasons': d.reasons,
            } for sym, d in self.mie_last_decision.items()},
            # Quality gate (observation/edge-analytics system) — mirrors the
            # richer /api/edge/report profile but exposed here too so the UI's
            # 5-second status poll can show armed/disarmed without a second
            # request on every page.
            'quality_gate_enabled': self.quality_gate_enabled,
        }

    # ── Trade execution ─────────────────────────────────────────────────────

    def _exit_logic(self, position, price, signal, confidence):
        """Shared exit decision for both sim and live. Returns (should_exit, reason).

        SL, TP, and trailing stop are expressed as % of MARGIN (not % of price).
        Dividing by leverage converts each threshold to the equivalent price-move
        trigger, so the margin-risk per trade stays constant regardless of what
        leverage the optimizer selects.

        Exit engine, in priority order:
          1. Stop Loss / Take Profit — hard boundaries.
          2. Breakeven floor — once the trade has cleared round-trip fees plus a
             buffer, arm a floor just above fee-breakeven. A winner can never
             round-trip into a loser (the single biggest scalping edge-saver).
          3. Trailing stop — while in profit, exit on retrace from peak.
          4. Signal reversal — only with FULL entry-grade confidence, only after a
             minimum hold, and only on the 2nd consecutive reversal cycle.
             Cuts fee-churn from transient model flips on noisy candles.
        """
        entry = position['entry_price']
        leverage = max(1, position.get('leverage', self.leverage))

        # Raw price move as a fraction of entry
        price_pnl_pct = (
            (price - entry) / entry if position['side'] == 'long'
            else (entry - price) / entry
        )
        # Scale to margin-relative PnL
        margin_pnl_pct = price_pnl_pct * leverage

        if margin_pnl_pct <= -self.stop_loss_pct:
            return True, 'Stop Loss'
        if margin_pnl_pct >= self.take_profit_pct:
            return True, 'Take Profit'

        # ── Breakeven floor ──────────────────────────────────────────────────
        # Round-trip fee cost in margin terms (maker entry + taker exit).
        fee_m = (MAKER_FEE + TAKER_FEE) * leverage
        be_trigger = max(4 * fee_m, 0.05)     # arm once pnl ≥ max(4× fees, 5% margin)
        be_floor   = 2 * fee_m                # lock in ≥ 2× fees = real net profit
        if not position.get('be_armed') and margin_pnl_pct >= be_trigger:
            position['be_armed'] = True
        if position.get('be_armed') and margin_pnl_pct <= be_floor:
            return True, 'Breakeven Floor'

        # ── Trailing stop ────────────────────────────────────────────────────
        # trailing_stop_pct is also margin-%, so price trail = setting / leverage.
        trail_price_pct = self.trailing_stop_pct / leverage
        if price_pnl_pct > 0:
            if position['side'] == 'long':
                if price <= position['high_water_mark'] * (1 - trail_price_pct):
                    return True, 'Trailing Stop'
            else:
                if price >= position['low_water_mark'] * (1 + trail_price_pct):
                    return True, 'Trailing Stop'

        # ── Time stop: thesis expiry ─────────────────────────────────────────
        # The model's label horizon is ~5 candles. A position that hasn't reached
        # the breakeven trigger within 8 candles is uninformed exposure — the
        # signal's predictive window has passed. Exit and recycle the capital into
        # the next fresh signal (this is what makes high frequency compound: margin
        # is never parked in dead trades).
        candle_s = self._get_timeframe_minutes() * 60
        if not position.get('be_armed'):
            age_s = time.time() - position.get('opened_at', 0)
            if position.get('opened_at') and age_s >= 8 * candle_s:
                return True, 'Time Stop'

        # ── Signal reversal (with hysteresis) ────────────────────────────────
        is_reversal = ((signal == -1 and position['side'] == 'long') or
                       (signal == 1 and position['side'] == 'short'))
        symbol = position.get('symbol', '')
        if is_reversal and confidence >= self.min_confidence:
            # Minimum hold: the label horizon is ~5 candles; give the thesis at
            # least 3 candles before a reversal can cut it (SL/TP/trail still live).
            min_hold_s = 3 * candle_s
            held = time.time() - position.get('opened_at', 0)
            if held >= min_hold_s:
                # Streak advances at most once per candle — live cycles run every
                # 30s, and two intra-candle flickers must not count as two candles
                # of confirmation.
                now = time.time()
                last_inc = position.get('rev_streak_ts', 0)
                if now - last_inc >= candle_s * 0.8:
                    streak = self._reversal_streak.get(symbol, 0) + 1
                    self._reversal_streak[symbol] = streak
                    position['rev_streak_ts'] = now
                    if streak >= 2:
                        return True, 'Signal Reversal'
            # armed but not yet confirmed — wait for the next candle
        else:
            # Any non-reversal cycle resets the persistence counter
            if symbol in self._reversal_streak:
                self._reversal_streak[symbol] = 0

        return False, ''

    def manual_enter(self, symbol: str, side: str) -> dict:
        """Force-open a position on demand, bypassing confidence/cooldown/filter gates."""
        side = side.lower()
        if side not in ('long', 'short'):
            return {'ok': False, 'error': f"side must be 'long' or 'short', got '{side}'"}
        with self._trade_lock:
            if symbol in self.positions:
                return {'ok': False, 'error': f'Position already open for {symbol}'}

            confidence = 0.75
            pos_side = side  # 'long' or 'short'
            conf_tier = 'MANUAL'

            if not self.simulation_mode:
                # Sync to real balance first so the manual size compounds off the
                # latest funds, then cap to available margin so the order can't be
                # rejected for insufficient funds.
                self._sync_live_balance()
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    price = float(ticker['last'])
                except Exception as e:
                    return {'ok': False, 'error': f'Price fetch failed: {e}'}
                try:
                    acct = self.exchange.fetch_balance()
                    avail = float(acct.get('USDT', {}).get('free', 0)) or float(acct.get('free', {}).get('USDT', 0))
                except Exception:
                    avail = self.balance
                margin = min(self._calculate_margin(confidence), avail * 0.95)
                if margin <= 0:
                    return {'ok': False, 'error': 'No available margin'}
                dyn_lev = self._dynamic_leverage(confidence)
                size = (margin * dyn_lev) / price
                try:
                    market = self.exchange.market(symbol)
                    contract_size = market.get('contractSize', 1) or 1
                    amount = size / contract_size
                    order_side = 'buy' if side == 'long' else 'sell'
                    try:
                        self.exchange.set_margin_mode('cross', symbol)
                    except Exception:
                        pass
                    self.exchange.set_leverage(dyn_lev, symbol, params={'marginMode': 'cross'})
                    self.exchange.create_order(
                        symbol=symbol, type='market', side=order_side, amount=amount,
                        params={'posSide': pos_side},
                    )
                except Exception as e:
                    return {'ok': False, 'error': f'Order failed: {e}'}
            else:
                try:
                    price = self.fetch_ohlcv(symbol=symbol, limit=5)['close'].iloc[-1]
                except Exception as e:
                    return {'ok': False, 'error': f'Price fetch failed: {e}'}
                margin = self._calculate_margin(confidence)
                dyn_lev = self._dynamic_leverage(confidence)
                size = (margin * dyn_lev) / price

            self.positions[symbol] = {
                'side': pos_side, 'size': size, 'entry_price': price,
                'symbol': symbol, 'margin': margin, 'leverage': dyn_lev,
                'high_water_mark': price, 'low_water_mark': price,
                'conf_tier': conf_tier, 'opened_at': time.time(),
            }
            self.entry_price = price
            self.last_trade_times[symbol] = time.time()
            self._reversal_streak[symbol] = 0
            self.trades_history.append({
                'type': 'open', 'side': pos_side, 'size': float(size), 'price': float(price),
                'confidence': float(confidence), 'symbol': symbol, 'margin': float(margin),
                'leverage': dyn_lev, 'conf_tier': conf_tier,
                'regime': self.market_regime, 'time': datetime.now().isoformat(),
            })
            if self.on_trade:
                self.on_trade(self.user_id, symbol, pos_side, 'open', size, price, None, confidence, None)
            mode = 'SIM' if self.simulation_mode else 'LIVE'
            logger.info(f"User {self.user_id}: [MANUAL/{mode}] Open {pos_side.upper()} {symbol} @ ${price:.2f} | Margin ${margin:.2f} · {dyn_lev}x")
            return {'ok': True, 'side': pos_side, 'price': price, 'margin': margin, 'leverage': dyn_lev}

    def manual_exit(self, symbol: str) -> dict:
        """Force-close an open position on demand."""
        with self._trade_lock:
            position = self.positions.get(symbol)
            if position is None:
                return {'ok': False, 'error': f'No open position for {symbol}'}

            try:
                ticker = self.exchange.fetch_ticker(symbol) if not self.simulation_mode else None
                price = float(ticker['last']) if ticker else self.fetch_ohlcv(symbol=symbol, limit=5)['close'].iloc[-1]
            except Exception as e:
                return {'ok': False, 'error': f'Price fetch failed: {e}'}

            if not self.simulation_mode:
                try:
                    market = self.exchange.market(symbol)
                    contract_size = market.get('contractSize', 1) or 1
                    amount = position['size'] / contract_size
                    close_side = 'sell' if position['side'] == 'long' else 'buy'
                    self.exchange.create_order(
                        symbol=symbol, type='market', side=close_side, amount=amount,
                        params={'posSide': position['side'], 'reduceOnly': True},
                    )
                except Exception as e:
                    return {'ok': False, 'error': f'Close order failed: {e}'}

            price_change = (price - position['entry_price']) if position['side'] == 'long' else (position['entry_price'] - price)
            pnl = price_change * position['size']
            self.total_pnl += pnl
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
            self._record_trade_roi(pnl)
            self.balance += pnl
            self._peak_balance = max(self._peak_balance, self.balance)

            margin_used = position.get('margin', 1)
            lev_pct = (pnl / margin_used * 100) if margin_used > 0 else 0
            exit_reason = 'Manual Exit'

            self.trades_history.append({
                'type': 'close', 'side': position['side'], 'price': float(price),
                'pnl': float(pnl), 'pnl_pct': round(lev_pct, 2),
                'reason': exit_reason, 'symbol': symbol, 'time': datetime.now().isoformat(),
            })
            if self.on_trade:
                self.on_trade(self.user_id, symbol, position['side'], 'close',
                              position['size'], price, pnl, None, exit_reason,
                              self._close_context(position, price, lev_pct))
            if self.on_performance:
                self.on_performance(self.user_id, self.balance, self.total_pnl,
                                    self.total_trades, self.winning_trades)
            mode = 'SIM' if self.simulation_mode else 'LIVE'
            logger.info(f"User {self.user_id}: [MANUAL/{mode}] Close {symbol} @ ${price:.2f} PnL ${pnl:.2f} ({lev_pct:.1f}%)")
            del self.positions[symbol]
            self._reversal_streak[symbol] = 0
            if not self.simulation_mode:
                self._sync_live_balance()
            return {'ok': True, 'pnl': pnl, 'pnl_pct': lev_pct, 'price': price}

    def _close_context(self, position, exit_price, lev_pct):
        """R-multiple accounting for a closed trade.

        R is the risk unit: one R = the configured stop distance in margin terms.
        Expressing results in R rather than dollars or percent is what makes
        expectancy comparable across position sizes, leverage settings and coins —
        it is the unit the whole edge analysis is built on.

        MFE/MAE (maximum favorable/adverse excursion, also in R) record how far
        the trade ran in each direction while open, which lets us judge the SETUP
        separately from the EXIT rule that happened to be in force.
        """
        risk = max(self.stop_loss_pct, 1e-6)
        lev = max(1, position.get('leverage', self.leverage))
        entry = position.get('entry_price') or 0
        ctx = dict(position.get('entry_ctx') or {})
        mfe_r = mae_r = None
        if entry > 0:
            if position['side'] == 'long':
                best  = (position.get('high_water_mark', entry) - entry) / entry
                worst = (position.get('low_water_mark', entry) - entry) / entry
            else:
                best  = (entry - position.get('low_water_mark', entry)) / entry
                worst = (entry - position.get('high_water_mark', entry)) / entry
            mfe_r = round(best * lev / risk, 4)
            mae_r = round(worst * lev / risk, 4)
        ctx.update({
            'r_multiple': round((lev_pct / 100.0) / risk, 4),
            'mfe_r': mfe_r, 'mae_r': mae_r,
            'leverage': lev,
            'hold_secs': int(time.time() - position.get('opened_at', time.time())),
            'quality_score': position.get('quality_score'),
            'setup_name': position.get('setup_name'),
        })
        return ctx

    def simulate_trade(self, signal, price, confidence, symbol=None, df=None, setup_name=None,
                       obs_ctx=None, quality=None):
        symbol = symbol or self.get_current_symbol()
        current_time = time.time()
        # Cooldown gates ENTRIES only — exit logic must run on every cycle so an
        # open position is never left without stop-loss/trailing protection.
        in_cooldown = current_time - self.last_trade_times.get(symbol, 0) < self.trade_cooldown

        position = self.positions.get(symbol)

        if position is None and in_cooldown:
            return

        # Setup-originated signals carry their own confidence floor (checked at
        # the call site); ML-originated signals use the user's min_confidence.
        conf_ok = (confidence >= self.min_confidence) or (setup_name is not None)
        if position is None and signal != 0 and conf_ok:
            if self._is_drawdown_exceeded():
                return
            if self._check_performance_floor():
                return
            if self._check_daily_governor():
                return
            if len(self.positions) >= self._daily_governor_max_pos():
                logger.info(f"User {self.user_id}: [{symbol}] Entry blocked — max positions ({self.max_positions}) reached")
                return
            ev = self._entry_ev(confidence)
            if ev <= 0:
                logger.info(f"User {self.user_id}: [{symbol}] Entry skipped — negative EV {ev:.3f}")
                return
            if not self._entry_filter(signal, df):
                return

            margin = self._calculate_margin(confidence) * self._get_regime_multiplier(signal)
            if margin <= 0:
                return

            dyn_lev = self._dynamic_leverage(confidence)
            notional = margin * dyn_lev
            size = notional / price
            side = 'long' if signal == 1 else 'short'
            if setup_name:
                conf_tier = f'SETUP:{setup_name}'
            else:
                conf_tier = 'NO-BRAINER' if confidence >= 0.85 else ('STRONG' if confidence >= 0.75 else 'MODERATE')

            self.positions[symbol] = {
                'side': side, 'size': size, 'entry_price': price,
                'symbol': symbol, 'margin': margin, 'leverage': dyn_lev,
                'high_water_mark': price, 'low_water_mark': price,
                'opened_at': current_time,
                # Market state at entry — this is what the post-mortem slices on.
                'entry_ctx': obs_ctx or {}, 'setup_name': setup_name,
                'quality_score': quality,
            }
            self.entry_price = price
            self.last_trade_times[symbol] = current_time
            self._reversal_streak[symbol] = 0

            trade = {
                'type': 'open', 'side': side, 'size': float(size), 'price': float(price),
                'confidence': float(confidence), 'symbol': symbol, 'margin': float(margin),
                'leverage': dyn_lev, 'conf_tier': conf_tier, 'setup': setup_name,
                'regime': self.market_regime, 'time': datetime.now().isoformat(),
            }
            self.trades_history.append(trade)
            if self.on_trade:
                self.on_trade(self.user_id, symbol, side, 'open', size, price, None, confidence, None,
                              obs_ctx or {})
            logger.info(
                f"User {self.user_id}: [SIM] Open {side.upper()} {symbol} @ ${price:.2f} | "
                f"Margin ${margin:.2f} · {dyn_lev}x lev [{conf_tier}]"
            )

        elif position is not None:
            # Track BOTH extremes regardless of side — the trailing stop reads the
            # favorable one, while MAE (adverse excursion) needs the other.
            position['high_water_mark'] = max(position['high_water_mark'], price)
            position['low_water_mark'] = min(position['low_water_mark'], price)

            should_exit, exit_reason = self._exit_logic(position, price, signal, confidence)

            if should_exit:
                price_change = (price - position['entry_price']) if position['side'] == 'long' else (position['entry_price'] - price)
                # Maker fee on entry (post-only limit), taker on exit (market) —
                # mirrors the live execution model.
                fee = (position['entry_price'] * MAKER_FEE + price * TAKER_FEE) * position['size']
                pnl = price_change * position['size'] - fee
                self.total_pnl += pnl
                self._day_pnl += pnl
                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
                self._record_trade_roi(pnl)
                self.balance += pnl
                self._peak_balance = max(self._peak_balance, self.balance)

                if pnl > 0:
                    self._add_trade_feedback(position['side'], pnl, df)

                margin_used = position.get('margin', 1)
                lev_pct = (pnl / margin_used * 100) if margin_used > 0 else 0

                self.trades_history.append({
                    'type': 'close', 'side': position['side'], 'price': float(price),
                    'pnl': float(pnl), 'pnl_pct': round(lev_pct, 2),
                    'reason': exit_reason, 'symbol': symbol, 'time': datetime.now().isoformat(),
                })
                if self.on_trade:
                    self.on_trade(self.user_id, symbol, position['side'], 'close',
                                  position['size'], price, pnl, None, exit_reason,
                                  self._close_context(position, price, lev_pct))
                if self.on_performance:
                    self.on_performance(self.user_id, self.balance, self.total_pnl,
                                        self.total_trades, self.winning_trades)
                logger.info(
                    f"User {self.user_id}: [SIM] Close {symbol} — {exit_reason} "
                    f"PnL ${pnl:.2f} ({lev_pct:.1f}%) fee=${fee:.2f} | Balance ${self.balance:.2f}"
                )
                del self.positions[symbol]
                # Cooldown restarts from CLOSE too — prevents the reversal-exit →
                # instant-re-entry ping-pong that burns a round trip of fees each time.
                self.last_trade_times[symbol] = current_time
                self._reversal_streak[symbol] = 0

    def _place_entry_order(self, symbol, side, pos_side, amount, ref_price):
        """
        Maker-first entry execution. Entries are optional — unlike exits, we never
        have to cross the spread for one, and at 8-10x leverage the taker fee plus
        spread is a large share of a scalp's edge (0.06% taker vs 0.02% maker per
        leg, ×leverage on margin).

          1. Spread guard: skip entirely if bid/ask spread > MAX_SPREAD_PCT.
          2. Post-only limit at the near touch (bid for long, ask for short).
          3. Poll up to ENTRY_LIMIT_WAIT_S for a fill.
          4. Full fill → done (maker). Partial ≥ 20% → cancel rest, keep the part.
             Less → cancel, skip this entry (the signal can re-fire next cycle).
          5. Any error in the maker path → market-order fallback (old behavior).

        Returns (order, avg_fill_price, filled_amount_contracts, maker: bool)
        or None if the entry should be skipped.
        """
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=5)
            bid = float(ob['bids'][0][0]); ask = float(ob['asks'][0][0])
            spread_pct = (ask - bid) / bid if bid > 0 else 1.0
            if spread_pct > MAX_SPREAD_PCT:
                logger.info(
                    f"User {self.user_id}: [{symbol}] Entry skipped — spread "
                    f"{spread_pct*10000:.1f} bps > {MAX_SPREAD_PCT*10000:.0f} bps limit"
                )
                return None

            limit_price = bid if side == 'buy' else ask
            order = self.exchange.create_order(
                symbol=symbol, type='limit', side=side, amount=amount, price=limit_price,
                params={'posSide': pos_side, 'postOnly': True},
            )
            order_id = order.get('id')

            deadline = time.time() + ENTRY_LIMIT_WAIT_S
            filled = 0.0
            while time.time() < deadline:
                time.sleep(2)
                o = self.exchange.fetch_order(order_id, symbol)
                filled = float(o.get('filled') or 0)
                status = o.get('status')
                if status == 'closed' or (amount > 0 and filled >= amount * 0.999):
                    avg = float(o.get('average') or limit_price)
                    logger.info(f"User {self.user_id}: [{symbol}] Maker entry filled @ ${avg:.4f}")
                    return o, avg, filled, True
                if status in ('canceled', 'rejected', 'expired'):
                    # postOnly rejected because it would have crossed — price moved.
                    logger.info(f"User {self.user_id}: [{symbol}] Maker entry {status} — skipping this cycle")
                    return None

            # Timeout: cancel the remainder
            try:
                self.exchange.cancel_order(order_id, symbol)
            except Exception:
                pass
            if amount > 0 and filled >= amount * 0.20:
                o = self.exchange.fetch_order(order_id, symbol)
                avg = float(o.get('average') or limit_price)
                logger.info(
                    f"User {self.user_id}: [{symbol}] Maker entry PARTIAL "
                    f"{filled/amount:.0%} @ ${avg:.4f} — keeping filled portion"
                )
                return o, avg, filled, True
            logger.info(f"User {self.user_id}: [{symbol}] Maker entry unfilled in {ENTRY_LIMIT_WAIT_S}s — skipped")
            return None

        except Exception as e:
            logger.warning(
                f"User {self.user_id}: [{symbol}] Maker entry path failed "
                f"({type(e).__name__}: {e}) — falling back to market order"
            )
            try:
                order = self.exchange.create_order(
                    symbol=symbol, type='market', side=side, amount=amount,
                    params={'posSide': pos_side},
                )
                return order, ref_price, amount, False
            except Exception as e2:
                logger.error(f"User {self.user_id}: [{symbol}] Market fallback FAILED: {e2}")
                return None

    def execute_live_trade(self, signal, price, confidence, symbol=None, df=None, setup_name=None,
                           obs_ctx=None, quality=None):
        symbol = symbol or self.get_current_symbol()
        current_time = time.time()
        # Cooldown gates ENTRIES only — an open position must have its exit logic
        # (SL/TP/trailing/breakeven) evaluated on every single cycle.
        cooldown_remaining = self.trade_cooldown - (current_time - self.last_trade_times.get(symbol, 0))
        if cooldown_remaining > 0 and self.positions.get(symbol) is None:
            if confidence >= self.min_confidence:
                logger.info(f"User {self.user_id}: [{symbol}] Signal skipped (conf={confidence:.1%}) — cooldown {cooldown_remaining:.0f}s remaining")
            return

        try:
            position = self.positions.get(symbol)
            sig_label = 'LONG' if signal == 1 else ('SHORT' if signal == -1 else 'FLAT')
            logger.info(f"User {self.user_id}: [{symbol}] Cycle — signal={sig_label} conf={confidence:.1%} threshold={self.min_confidence:.1%} position={'open' if position else 'none'}")

            # Setup-originated signals carry their own confidence floor (checked
            # at the call site); ML-originated signals use min_confidence.
            conf_ok = (confidence >= self.min_confidence) or (setup_name is not None)
            if position is None and signal != 0 and conf_ok:
                if self._is_drawdown_exceeded():
                    logger.warning(f"User {self.user_id}: [{symbol}] Entry blocked — max drawdown limit reached")
                    return
                if self._check_performance_floor():
                    return
                if self._check_daily_governor():
                    return
                if len(self.positions) >= self._daily_governor_max_pos():
                    logger.info(f"User {self.user_id}: [{symbol}] Entry blocked — max positions ({self.max_positions}) reached")
                    return
                ev = self._entry_ev(confidence)
                if ev <= 0:
                    logger.info(f"User {self.user_id}: [{symbol}] Entry skipped — negative EV {ev:.3f}")
                    return
                if not self._entry_filter(signal, df):
                    return  # _entry_filter already logs the reason

                # Sync balance from exchange before sizing — catches deposits/settlements
                # that occurred since the last close so every entry compounds off the
                # latest real balance.
                self._sync_live_balance()

                try:
                    acct = self.exchange.fetch_balance()
                    avail = float(acct.get('USDT', {}).get('free', 0))
                    if avail <= 0:
                        avail = float(acct.get('free', {}).get('USDT', 0))
                except Exception as e:
                    logger.warning(f"User {self.user_id}: Balance fetch failed: {e}")
                    avail = self.balance

                margin = self._calculate_margin(confidence) * self._get_regime_multiplier(signal)
                margin = min(margin, avail * 0.95)

                dyn_lev = self._dynamic_leverage(confidence)
                notional = margin * dyn_lev
                size = notional / price
                side = 'buy' if signal == 1 else 'sell'
                pos_side = 'long' if signal == 1 else 'short'
                if setup_name:
                    conf_tier = f'SETUP:{setup_name}'
                else:
                    conf_tier = 'NO-BRAINER' if confidence >= 0.85 else ('STRONG' if confidence >= 0.75 else 'MODERATE')

                market = self.exchange.market(symbol)
                contract_size = market.get('contractSize', 1) or 1
                amount = size / contract_size

                try:
                    # Cross margin: set mode first, then dynamic leverage
                    try:
                        self.exchange.set_margin_mode('cross', symbol)
                    except Exception:
                        pass  # already set, or exchange doesn't need explicit call
                    self.exchange.set_leverage(dyn_lev, symbol, params={'marginMode': 'cross'})
                    logger.info(f"User {self.user_id}: Leverage set to {dyn_lev}x (cross, {conf_tier}) on {symbol}")
                except Exception as e:
                    logger.error(f"User {self.user_id}: set_leverage({dyn_lev}x, {symbol}) FAILED — {type(e).__name__}: {e}.")

                fill = self._place_entry_order(symbol, side, pos_side, amount, price)
                if fill is None:
                    return  # spread too wide or maker order unfilled — skip, retry next cycle
                order, fill_price, fill_size_contracts, maker_entry = fill
                filled_size = fill_size_contracts * contract_size

                self.positions[symbol] = {
                    'side': pos_side, 'size': filled_size, 'entry_price': fill_price,
                    'symbol': symbol, 'margin': margin * (filled_size / size if size > 0 else 1),
                    'leverage': dyn_lev,
                    'order_id': order.get('id'),
                    'high_water_mark': fill_price, 'low_water_mark': fill_price,
                    'opened_at': current_time, 'maker_entry': maker_entry,
                    # Market state at entry — what the post-mortem slices on.
                    'entry_ctx': obs_ctx or {}, 'setup_name': setup_name,
                    'quality_score': quality,
                }
                price, size = fill_price, filled_size  # for history/log below
                self.entry_price = price
                self.last_trade_times[symbol] = current_time
                self._reversal_streak[symbol] = 0

                self.trades_history.append({
                    'type': 'open', 'side': pos_side, 'size': float(size), 'price': float(price),
                    'confidence': float(confidence), 'symbol': symbol, 'margin': float(margin),
                    'leverage': dyn_lev, 'conf_tier': conf_tier, 'setup': setup_name,
                    'regime': self.market_regime, 'time': datetime.now().isoformat(),
                })
                if self.on_trade:
                    self.on_trade(self.user_id, symbol, pos_side, 'open', size, price, None, confidence, None,
                                  obs_ctx or {})
                logger.info(
                    f"User {self.user_id}: [LIVE] Open {pos_side.upper()} {symbol} @ ${price:.2f} | "
                    f"Margin ${margin:.2f} · {dyn_lev}x lev [{conf_tier}] · notional ${notional:.2f}"
                )

            elif position is not None:
                # Use real-time ticker price for exit decisions — much faster than
                # waiting for a full OHLCV candle to update
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    live_price = float(ticker.get('last') or ticker.get('close') or price)
                    if live_price > 0:
                        price = live_price
                except Exception:
                    pass  # fall back to OHLCV close price

                # Track BOTH extremes regardless of side — the trailing stop reads
                # the favorable one, MAE (adverse excursion) needs the other.
                position['high_water_mark'] = max(position['high_water_mark'], price)
                position['low_water_mark'] = min(position['low_water_mark'], price)

                should_exit, exit_reason = self._exit_logic(position, price, signal, confidence)

                if should_exit:
                    pos_sym = position.get('symbol', symbol)
                    close_side = 'sell' if position['side'] == 'long' else 'buy'
                    market = self.exchange.market(pos_sym)
                    amount = position['size'] / (market.get('contractSize', 1) or 1)

                    try:
                        self.exchange.create_order(
                            symbol=pos_sym, type='market', side=close_side, amount=amount,
                            params={'posSide': position['side'], 'reduceOnly': True},
                        )
                    except Exception as close_err:
                        logger.error(f"User {self.user_id}: Close order FAILED for {pos_sym}: {close_err}. Removing position and resyncing.")
                        del self.positions[symbol]
                        self._sync_live_balance()
                        return

                    price_change = (price - position['entry_price']) if position['side'] == 'long' else (position['entry_price'] - price)
                    # Maker entry (post-only limit) + taker exit (market)
                    entry_fee_rate = MAKER_FEE if position.get('maker_entry') else TAKER_FEE
                    fee = (position['entry_price'] * entry_fee_rate + price * TAKER_FEE) * position['size']
                    pnl = price_change * position['size'] - fee
                    self.total_pnl += pnl
                    self._day_pnl += pnl
                    self.total_trades += 1
                    if pnl > 0:
                        self.winning_trades += 1
                    self._record_trade_roi(pnl)
                    self.balance += pnl
                    self._peak_balance = max(self._peak_balance, self.balance)

                    if pnl > 0:
                        self._add_trade_feedback(position['side'], pnl, df)

                    margin_used = position.get('margin', 1)
                    lev_pct = (pnl / margin_used * 100) if margin_used > 0 else 0

                    self.trades_history.append({
                        'type': 'close', 'side': position['side'], 'price': float(price),
                        'pnl': float(pnl), 'pnl_pct': round(lev_pct, 2),
                        'reason': exit_reason, 'symbol': pos_sym, 'time': datetime.now().isoformat(),
                    })
                    if self.on_trade:
                        self.on_trade(self.user_id, pos_sym, position['side'], 'close',
                                      position['size'], price, pnl, None, exit_reason,
                                      self._close_context(position, price, lev_pct))
                    if self.on_performance:
                        self.on_performance(self.user_id, self.balance, self.total_pnl,
                                            self.total_trades, self.winning_trades)
                    logger.info(
                        f"User {self.user_id}: [LIVE] Close {pos_sym} — {exit_reason} "
                        f"PnL ${pnl:.2f} ({lev_pct:.1f}%) fee=${fee:.2f}"
                    )
                    del self.positions[symbol]
                    # Cooldown restarts from CLOSE — no instant re-entry churn.
                    self.last_trade_times[symbol] = current_time
                    self._reversal_streak[symbol] = 0
                    # Don't sync immediately after close — the next entry's pre-trade
                    # sync (line ~1580) will pick up the settled balance accurately.

            elif position is None and signal == 0:
                logger.debug(f"User {self.user_id}: [{symbol}] No trade — model output FLAT (signal=0)")
            elif position is None and signal != 0 and confidence < self.min_confidence:
                logger.info(f"User {self.user_id}: [{symbol}] No trade — conf {confidence:.1%} below threshold {self.min_confidence:.1%}")

        except Exception as e:
            logger.error(f"User {self.user_id}: Live trade execution FAILED: {type(e).__name__}: {e}", exc_info=True)

    # ── Backtest ─────────────────────────────────────────────────────────────

    def run_backtest_single_coin(self, symbol, days=30):
        TAKER_FEE = 0.0006
        minutes_per_candle = self._get_timeframe_minutes()
        periods = int(days * 24 * 60 / minutes_per_candle)
        df = self.fetch_ohlcv(symbol=symbol, limit=min(periods, LOOKBACK_PERIODS * 3))

        if df is None or len(df) < 100:
            return {'symbol': symbol, 'error': 'Not enough data', 'total_return': 0, 'win_rate': 0, 'total_trades': 0}

        df = self.calculate_indicators(df)
        df = self.create_labels(df)
        df = df.dropna()

        if len(df) < 50:
            return {'symbol': symbol, 'error': 'Not enough data after processing', 'total_return': 0, 'win_rate': 0, 'total_trades': 0}

        train_size = len(df) // 2
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        X_train, _ = self.prepare_features(train_df)
        y_train = train_df['signal']
        mask = y_train != 0
        X_train, y_train = X_train[mask], y_train[mask]

        if len(X_train) < 20:
            return {'symbol': symbol, 'error': 'Not enough training signals', 'total_return': 0, 'win_rate': 0, 'total_trades': 0}

        try:
            imp = SimpleImputer(strategy='mean')
            sc = StandardScaler()
            X_imp = imp.fit_transform(X_train)
            X_sc = sc.fit_transform(X_imp)

            try:
                smote = SMOTE(random_state=42,
                              k_neighbors=min(3, len(y_train[y_train == 1]) - 1, len(y_train[y_train == -1]) - 1))
                X_res, y_res = smote.fit_resample(X_sc, y_train)
            except Exception:
                X_res, y_res = X_sc, y_train

            is_tree = LGBM_AVAILABLE or XGB_AVAILABLE or HGBT_AVAILABLE
            model = self._build_model()
            if is_tree:
                model.fit(X_res, _encode_y(y_res))
            else:
                tscv = TimeSeriesSplit(n_splits=3)
                grid = GridSearchCV(model, SVM_PARAMS, cv=tscv, scoring='f1_weighted', n_jobs=1)
                grid.fit(X_res, y_res)
                model = grid.best_estimator_
        except Exception as e:
            return {'symbol': symbol, 'error': f'Model training failed: {e}', 'total_return': 0, 'win_rate': 0, 'total_trades': 0}

        balance_per_coin = self.starting_balance / len(self.selected_coins)
        balance = balance_per_coin
        position = None
        entry_price = 0
        hwm = lwm = 0
        trades = []
        last_trade_candle = -999
        entry_candle = -999
        reversal_streak = 0
        be_armed = False
        cooldown_candles = max(1, math.ceil(self.trade_cooldown / 60 / minutes_per_candle))
        SLIPPAGE = 0.0005  # 5 bps slippage on the taker (exit) leg
        bt_lev = max(1, getattr(self, 'leverage', 1))
        fee_m_bt = (MAKER_FEE + TAKER_FEE) * bt_lev
        be_trigger = max(4 * fee_m_bt, 0.05)
        be_floor = 2 * fee_m_bt
        X_test, _ = self.prepare_features(test_df)

        for i in range(len(test_df)):
            try:
                row = test_df.iloc[i]
                price = row['close']
                X_row = X_test.iloc[[i]]
                X_imp2 = imp.transform(X_row)
                X_sc2 = sc.transform(X_imp2)
                raw = model.predict(X_sc2)[0]
                sig_val = int(_decode_y([raw])[0]) if is_tree else int(raw)
                conf = float(max(model.predict_proba(X_sc2)[0]))

                # Entry quality filter — same gate as live trading
                adx_raw = row.get('adx', 25); adx_f = 25.0 if pd.isna(adx_raw) else float(adx_raw)
                vol_raw = row.get('volume_ratio', 1.0); vol_f = 1.0 if pd.isna(vol_raw) else float(vol_raw)
                if position is None and sig_val != 0 and conf >= self.min_confidence and adx_f >= self.adx_threshold and vol_f >= 0.65:
                    # Cooldown gates ENTRIES only — matches live engine
                    if i - last_trade_candle < cooldown_candles:
                        continue
                    # Margin formula mirrors live bot exactly: confidence scaling + risk_per_trade ceiling
                    profit = max(0, balance - balance_per_coin)
                    k = self.risk_per_trade  # Fixed fraction in backtest (Kelly needs live history)
                    margin = balance_per_coin * k + profit * k * self.profit_risk_multiplier
                    # Confidence scaling — matches _calculate_margin() exactly
                    conf_rng = max(0.01, 1.0 - self.min_confidence)
                    c_scale = max(0.5, min(1.5, 0.5 + (conf - self.min_confidence) / conf_rng))
                    margin *= c_scale
                    # Risk ceiling — same formula as live
                    risk_ceil = min(max(self.risk_per_trade * 2.0, 0.10), 0.75)
                    margin = min(margin, balance * risk_ceil)
                    if margin <= 0:
                        continue

                    notional = margin * self.leverage
                    size = notional / price
                    entry_fee = notional * MAKER_FEE   # post-only limit entry
                    position = {
                        'side': 'long' if sig_val == 1 else 'short',
                        'size': size, 'margin': margin, 'entry_fee': entry_fee,
                    }
                    entry_price = price
                    hwm = lwm = price
                    last_trade_candle = i
                    entry_candle = i
                    reversal_streak = 0
                    be_armed = False

                elif position is not None:
                    if position['side'] == 'long':
                        hwm = max(hwm, price)
                    else:
                        lwm = min(lwm, price)

                    price_pnl_pct = (
                        (price - entry_price) / entry_price if position['side'] == 'long'
                        else (entry_price - price) / entry_price
                    )
                    margin_pnl_pct = price_pnl_pct * bt_lev
                    trail_price_pct = self.trailing_stop_pct / bt_lev

                    should_exit = False
                    exit_reason = ''
                    if margin_pnl_pct <= -self.stop_loss_pct:
                        should_exit, exit_reason = True, 'Stop Loss'
                    elif margin_pnl_pct >= self.take_profit_pct:
                        should_exit, exit_reason = True, 'Take Profit'
                    else:
                        # Breakeven floor — winner can't round-trip into a loser
                        if not be_armed and margin_pnl_pct >= be_trigger:
                            be_armed = True
                        if be_armed and margin_pnl_pct <= be_floor:
                            should_exit, exit_reason = True, 'Breakeven Floor'
                        # Time stop — thesis expired, recycle capital (matches live)
                        if not should_exit and not be_armed and (i - entry_candle) >= 8:
                            should_exit, exit_reason = True, 'Time Stop'
                        if not should_exit and price_pnl_pct > 0:
                            if position['side'] == 'long' and price <= hwm * (1 - trail_price_pct):
                                should_exit, exit_reason = True, 'Trailing Stop'
                            elif position['side'] == 'short' and price >= lwm * (1 + trail_price_pct):
                                should_exit, exit_reason = True, 'Trailing Stop'
                        if not should_exit:
                            # Reversal hysteresis — full entry confidence, min-hold,
                            # 2 consecutive reversal candles (matches live)
                            is_rev = (sig_val != 0 and
                                      ((sig_val == -1 and position['side'] == 'long') or
                                       (sig_val == 1 and position['side'] == 'short')))
                            if is_rev and conf >= self.min_confidence and (i - entry_candle) >= 3:
                                reversal_streak += 1
                                if reversal_streak >= 2:
                                    should_exit, exit_reason = True, 'Signal Reversal'
                            else:
                                reversal_streak = 0

                    if should_exit:
                        price_change = (price - entry_price) if position['side'] == 'long' else (entry_price - price)
                        pnl_amount = price_change * position['size']
                        exit_fee = position['size'] * price * (TAKER_FEE + SLIPPAGE)
                        total_fees = position['entry_fee'] + exit_fee
                        net_pnl = pnl_amount - total_fees
                        balance += net_pnl

                        margin_used = position['margin']
                        lev_pct = (net_pnl / margin_used * 100) if margin_used > 0 else 0

                        trades.append({
                            'symbol': symbol, 'coin': symbol.split('/')[0],
                            'side': position['side'],
                            'entry': round(entry_price, 2), 'exit': round(price, 2),
                            'pnl': round(net_pnl, 2), 'pnl_pct': round(lev_pct, 2),
                            'fees': round(total_fees, 4), 'reason': exit_reason,
                        })
                        position = None
                        last_trade_candle = i
                        reversal_streak = 0
                        be_armed = False
            except Exception:
                continue

        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        total_fees = sum(t.get('fees', 0) for t in trades)
        total_return_pct = ((balance - balance_per_coin) / balance_per_coin) * 100

        # Max drawdown
        max_drawdown = 0
        peak = balance_per_coin
        running = balance_per_coin
        for t in trades:
            running += t['pnl']
            peak = max(peak, running)
            dd = (peak - running) / peak * 100
            max_drawdown = max(max_drawdown, dd)

        monthly_roi = round(((1 + total_return_pct / 100) ** (30 / max(days, 1)) - 1) * 100, 2) if total_return_pct > 0 else 0.0
        sharpe_ratio = 0.0
        if len(trades) > 1:
            tret = [t['pnl'] / max(balance_per_coin, 1) for t in trades]
            sharpe_ratio = round(float(np.mean(tret) / (np.std(tret) + 1e-10) * np.sqrt(len(tret))), 3)
        calmar_ratio = round(total_return_pct / max(max_drawdown, 0.01), 2) if total_return_pct > 0 else 0.0

        return {
            'symbol': symbol, 'coin': symbol.split('/')[0],
            'starting_balance': round(balance_per_coin, 2),
            'final_balance': round(balance, 2),
            'total_return': round(total_return_pct, 2),
            'monthly_roi': monthly_roi,
            'total_pnl': round(sum(t['pnl'] for t in trades), 2),
            'total_fees': round(total_fees, 4),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': round((winning_trades / total_trades * 100) if total_trades > 0 else 0, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'trades': trades,
        }

    def run_backtest(self, days=30):
        coin_results = []
        all_trades = []
        for symbol in self.selected_coins:
            result = self.run_backtest_single_coin(symbol, days)
            coin_results.append(result)
            if 'trades' in result:
                all_trades.extend(result['trades'])

        total_starting = self.starting_balance
        total_final = sum(r.get('final_balance', r.get('starting_balance', 0)) for r in coin_results)
        total_pnl = total_final - total_starting
        total_trades_count = sum(r.get('total_trades', 0) for r in coin_results)
        winning_trades = sum(r.get('winning_trades', 0) for r in coin_results)
        max_drawdown = max((r.get('max_drawdown', 0) for r in coin_results), default=0)
        total_fees = sum(r.get('total_fees', 0) for r in coin_results)
        total_return_pct = ((total_final - total_starting) / total_starting * 100) if total_starting > 0 else 0

        monthly_roi = round(((1 + total_return_pct / 100) ** (30 / max(days, 1)) - 1) * 100, 2) if total_return_pct > 0 else 0.0
        sharpe_ratio = 0.0
        if len(all_trades) > 1:
            tret = [t['pnl'] / max(total_starting, 1) for t in all_trades]
            sharpe_ratio = round(float(np.mean(tret) / (np.std(tret) + 1e-10) * np.sqrt(len(tret))), 3)
        calmar_ratio = round(total_return_pct / max(max_drawdown, 0.01), 2) if total_return_pct > 0 else 0.0
        mr = monthly_roi / 100
        compound_projection = {
            '1m':  int(1000 * (1 + mr) ** 1),
            '3m':  int(1000 * (1 + mr) ** 3),
            '6m':  int(1000 * (1 + mr) ** 6),
            '12m': int(1000 * (1 + mr) ** 12),
            '24m': int(1000 * (1 + mr) ** 24),
            '36m': int(1000 * (1 + mr) ** 36),
        }
        months_to_1m = math.ceil(math.log(1000) / math.log(1 + mr)) if mr > 0 else None

        return {
            'period_days': days,
            'starting_balance': total_starting,
            'final_balance': round(total_final, 2),
            'total_return': round(total_return_pct, 2),
            'monthly_roi': monthly_roi,
            'total_pnl': round(total_pnl, 2),
            'total_fees': round(total_fees, 4),
            'total_trades': total_trades_count,
            'winning_trades': winning_trades,
            'win_rate': round((winning_trades / total_trades_count * 100) if total_trades_count > 0 else 0, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'compound_projection': compound_projection,
            'months_to_1m': months_to_1m,
            'leverage': self.leverage,
            'risk_per_trade': self.risk_per_trade,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'trailing_stop_pct': self.trailing_stop_pct,
            'min_confidence': self.min_confidence,
            'timeframe': self.timeframe,
            'selected_coins': self.selected_coins,
            'coin_results': coin_results,
            'all_trades': all_trades,
        }

    # ── Market Intelligence Engine (backend/mie package) ────────────────────
    # Standalone, evidence-gated decision layer: records unlabeled state,
    # validates conditional-expectancy models against held-out data before
    # trusting them, defaults to DO_NOTHING. See backend/mie/README.md. Kept
    # deliberately separate from the observation/edge-analytics layer just
    # below, which is the inline, database-backed counterpart built the same
    # day in a parallel session — the two overlap in purpose but not in
    # implementation, and reconciling them into one system is a follow-up,
    # not something to collapse silently in a merge.

    def _mie_cycle(self, symbol: str, df_ind, new_candle: bool):
        """
        Record this cycle's market state (unlabeled) into the engine's
        observation store, resolve any outcomes that have matured since the
        last new candle, periodically re-fit + re-validate the per-horizon
        edge models, and return the current TradeDecision for `symbol`.

        Never allowed to raise into the trading loop — a research subsystem
        misbehaving must not be able to take live trading down with it. On any
        failure this simply returns None, which reads to callers as "MIE has
        no opinion right now", the same as its legitimate cold-start state.
        """
        if self._mie is None:
            return None
        try:
            decision = self._mie.decide(symbol, df_ind, user_id=self.user_id)
            self.mie_last_decision[symbol] = decision
            if new_candle:
                bar_seconds = self._get_timeframe_minutes() * 60
                self._mie_store.backfill_outcomes(self.user_id, symbol, df_ind, bar_seconds=bar_seconds)
                self._maybe_refit_mie(bar_seconds)
            return decision
        except Exception as e:
            logger.debug(f"User {self.user_id}: MIE cycle failed for {symbol}: {e}")
            return None

    def _maybe_refit_mie(self, bar_seconds: float):
        """
        Re-fit + re-validate every horizon's edge model against everything the
        store has accumulated (pooled across users — see
        ObservationStore.load_training_frame), on a fixed schedule and only
        once there's enough resolved history for a fit attempt to mean
        anything. This is what lets a freshly-started bot go from "no opinion"
        to "validated" over its first few hours/days of operation without any
        manual optimizer run.
        """
        now = time.time()
        if now - self._mie_last_fit < self._mie_fit_interval:
            return
        if self._mie_store.count_resolved() < self._mie_min_rows_to_fit:
            return
        try:
            training_df = self._mie_store.load_training_frame()
            if len(training_df) < self._mie_min_rows_to_fit:
                return
            summaries = self._mie.fit(training_df, bar_seconds=bar_seconds)
            self._mie_last_fit = now
            logger.info(f"User {self.user_id}: MIE re-fit across {len(training_df)} observations — "
                       f"{ {h: ('validated' if m.validated else m.report.reason) for h, m in self._mie.edge_models.items()} }")
        except Exception as e:
            logger.warning(f"User {self.user_id}: MIE re-fit failed: {e}")

    # ── Observation / edge-analytics layer (inline, database-backed) ───────

    def _volatility_percentile(self, df) -> float:
        """Where current realized volatility sits within its own recent history
        (0 = calmest in 200 bars, 1 = most volatile). Regime conditioning needs a
        RELATIVE measure — absolute ATR means nothing across coins or eras."""
        try:
            v = df['volatility'].dropna().tail(200)
            if len(v) < 30:
                return 0.5
            cur = float(v.iloc[-1])
            return float((v < cur).sum()) / len(v)
        except Exception:
            return 0.5

    def _collect_market_context(self, symbol: str, df) -> dict:
        """Order-flow and derivatives-positioning snapshot for one symbol.

        This is the honest, host-appropriate subset of what a professional desk
        watches: book imbalance and spread from a depth snapshot, plus funding
        and open-interest change. It is NOT tick-level tape — reconstructing real
        order flow needs a persistent websocket feed and tick storage, which this
        host cannot run. Every field is best-effort and individually guarded; a
        market-data hiccup must never interrupt trading.

        Cached per candle so repeated intra-candle cycles don't re-poll.
        """
        tf_s = self._get_timeframe_minutes() * 60
        now = time.time()
        cached = self._mkt_ctx_cache.get(symbol)
        if cached and (now - cached[0]) < tf_s * 0.5:
            return cached[1]

        ctx = {'spread_bps': None, 'book_imbalance': None,
               'funding_rate': None, 'oi_change': None}
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=20)
            bids, asks = ob.get('bids') or [], ob.get('asks') or []
            if bids and asks:
                bid, ask = float(bids[0][0]), float(asks[0][0])
                if bid > 0:
                    ctx['spread_bps'] = round((ask - bid) / bid * 10000, 4)
                bid_depth = sum(float(b[1]) for b in bids[:10])
                ask_depth = sum(float(a[1]) for a in asks[:10])
                tot = bid_depth + ask_depth
                if tot > 0:
                    # 0.5 = balanced, >0.5 = bid-heavy (buy pressure)
                    ctx['book_imbalance'] = round(bid_depth / tot, 4)
        except Exception:
            pass
        try:
            fr = self.exchange.fetch_funding_rate(symbol)
            v = fr.get('fundingRate')
            if v is not None:
                ctx['funding_rate'] = float(v)
        except Exception:
            pass
        try:
            oi = self.exchange.fetch_open_interest(symbol)
            val = oi.get('openInterestAmount') or oi.get('openInterestValue')
            if val:
                val = float(val)
                prev = self._last_oi.get(symbol)
                if prev and prev > 0:
                    ctx['oi_change'] = round((val - prev) / prev, 6)
                self._last_oi[symbol] = val
        except Exception:
            pass

        self._mkt_ctx_cache[symbol] = (now, ctx)
        return ctx

    def _observation_context(self, symbol, df, signal, confidence, setup_name, traded):
        """Assemble the full market-state context for one observation/entry."""
        ctx = dict(self._collect_market_context(symbol, df))
        try:
            last_ret = float(df['returns'].iloc[-1]) if 'returns' in df else 0.0
        except Exception:
            last_ret = 0.0
        btc_agree = None
        if abs(self._btc_ret_5) > 1e-9 and abs(last_ret) > 1e-9:
            btc_agree = (self._btc_ret_5 > 0) == (last_ret > 0)
        ctx.update({
            'regime': self.market_regime,
            'vol_pct': round(self._volatility_percentile(df), 4),
            'btc_ret_5': round(self._btc_ret_5, 6),
            'btc_agree': btc_agree,
            'hour_utc': datetime.utcnow().hour,
            'ml_signal': int(signal), 'ml_conf': round(float(confidence), 4),
            'setup_name': setup_name, 'traded': traded,
        })
        return ctx

    def _record_observation(self, symbol, df, signal, confidence, setup_name, traded, ctx=None):
        """Record market state for this candle. Outcomes are backfilled later.
        Pass `ctx` to record the exact context the quality gate scored, so the
        stored row reflects the decision that was actually made."""
        if not self.on_observation:
            return None
        try:
            X, _ = self.prepare_features(df)
            if X.empty:
                return None
            feats = {k: (None if pd.isna(v) else round(float(v), 6))
                     for k, v in X.iloc[-1].items()}
            if ctx is None:
                ctx = self._observation_context(symbol, df, signal, confidence, setup_name, traded)
            ctx = dict(ctx)
            ctx['traded'] = traded
            self.on_observation(self.user_id, symbol, df.index[-1],
                                float(df['close'].iloc[-1]), self.timeframe, feats, ctx)
            return ctx
        except Exception as e:
            logger.debug(f"User {self.user_id}: observation record failed: {e}")
            return None

    def _backfill_observations(self, symbol, df):
        """Fill forward outcomes for observations old enough to have resolved.

        Computes returns at 1/3/5/15 candles plus maximum favorable and adverse
        excursion over the 15-candle window, reading straight from the OHLCV
        already in memory — no extra API calls. MFE/MAE are what let us judge a
        setup independently of whatever exit rule happened to be in force.
        """
        if not (self.get_pending_obs and self.on_backfill):
            return
        try:
            pending = self.get_pending_obs(self.user_id, symbol, 200)
            if not pending:
                return
            idx = df.index
            rows = []
            for obs in pending:
                ts = obs['candle_ts']
                try:
                    pos = idx.get_loc(pd.Timestamp(ts))
                except Exception:
                    continue  # candle outside the loaded window — retry later
                if not isinstance(pos, int):
                    continue
                if pos + 15 >= len(df):
                    continue  # not enough forward candles yet
                p0 = float(df['close'].iloc[pos])
                if p0 <= 0:
                    continue
                fwd = df.iloc[pos + 1: pos + 16]
                r = lambda n: round(float(df['close'].iloc[pos + n]) / p0 - 1, 8)
                rows.append((
                    r(1), r(3), r(5), r(15),
                    round(float(fwd['high'].max()) / p0 - 1, 8),   # MFE
                    round(float(fwd['low'].min()) / p0 - 1, 8),    # MAE
                    obs['id'],
                ))
            if rows:
                n = self.on_backfill(rows)
                logger.debug(f"User {self.user_id}: [{symbol}] backfilled {n} observation outcomes")
        except Exception as e:
            logger.debug(f"User {self.user_id}: observation backfill failed: {e}")

    def _trade_quality(self, symbol, ctx):
        """
        Score this opportunity 0-100 from the historical conditional edge profile,
        and decide whether we have measured evidence of an advantage right now.

        The profile is computed out-of-band (see edge_analytics) and holds, per
        condition bucket, the LOWER CONFIDENCE BOUND on expectancy in R together
        with the sample size behind it. Using the lower bound rather than the
        point estimate is the guard against multiple-testing: slice trades by
        regime x volatility x setup x hour and some bucket will look brilliant by
        luck, but luck does not survive a confidence bound built on its own
        sample size.

        Returns (score, reason). NO TRADE is the default — an unknown bucket does
        not get the benefit of the doubt once the gate is armed.
        """
        prof = self.edge_profile or {}
        buckets = prof.get('buckets') or {}
        if not buckets:
            return 50, 'no profile yet'

        # Build the exact same "row" shape edge_analytics.bucket_keys() reads
        # when it built these buckets from trade history, so the live lookup
        # can see every dimension the profile actually differentiates on —
        # symbol, side and session included. A hand-rolled subset of keys here
        # (the previous version checked only setup/regime/vol/btc_agree) meant
        # every coin and every direction collapsed onto the same shared
        # regime-level bucket, silently discarding the profile's per-coin and
        # per-side evidence and making every symbol score identically.
        side = 'long' if ctx.get('ml_signal') == 1 else ('short' if ctx.get('ml_signal') == -1 else None)
        row = dict(ctx)
        row['symbol'] = symbol
        row['side'] = side
        row['context'] = {'btc_agree': ctx.get('btc_agree'), 'book_imbalance': ctx.get('book_imbalance')}
        keys = _edge_bucket_keys(row)

        applicable = [(k, buckets[k]) for k in keys if k in buckets
                      and buckets[k].get('n', 0) >= prof.get('min_sample', 25)]
        if not applicable:
            return 45, 'no comparable history'

        # Worst applicable bucket governs — a condition known to be unprofitable
        # is not rescued by a different slice that happens to look fine.
        #
        # Name the condition that drove the verdict. Several coins genuinely CAN
        # score identically: when the worst applicable bucket is a market-wide
        # slice (regime / volatility band / session) they are all being judged by
        # the same shared evidence, and that is the correct answer, not a bug.
        # But an unattributed bare number ("-0.420R (n=30)") repeated across every
        # coin is indistinguishable from a broken gate — naming the bucket is the
        # difference between that reading as an explanation and reading as a fault.
        worst_key, worst = min(applicable, key=lambda kv: float(kv[1]['expectancy_lb']))
        lb = float(worst['expectancy_lb'])
        # The DRIVING bucket's own sample size. Previously this was min(n) across
        # every applicable bucket, which could report a sample belonging to a
        # completely different slice than the expectancy shown beside it.
        n = int(worst['n'])
        prov = ', provisional' if worst.get('provisional') else ''
        score = int(max(0, min(100, 50 + lb * 100)))
        return score, f'{worst_key} · lower-bound {lb:+.3f}R (n={n}{prov})'

    # ── Main cycle ───────────────────────────────────────────────────────────

    def run_cycle(self):
        if not self.running:
            return None

        self._cycle_count += 1
        self._reset_daily_if_needed()

        # Re-check market regime every N cycles (uses a separate 4h BTC fetch)
        if self._cycle_count % self._regime_check_every == 1:
            self.market_regime = self._get_market_regime()
            logger.info(f"User {self.user_id}: Market regime → {self.market_regime}")

        results = []
        for symbol in self.selected_coins:
            df = self.fetch_ohlcv_fast(symbol=symbol)
            if df is None or len(df) < 50:
                continue

            # Detect whether the latest candle is new (used to gate entries in live mode)
            latest_ts = df.index[-1] if hasattr(df.index, '__len__') else None
            new_candle = (latest_ts != self._last_candle_ts.get(symbol))
            self._last_candle_ts[symbol] = latest_ts

            # Only retrain on a new candle (avoids redundant training on intra-candle cycles)
            should_train = new_candle and (self.model is None or (self._cycle_count % self.retrain_every == 0))
            if should_train:
                logger.info(
                    f"User {self.user_id}: {'Initial' if self.model is None else 'Periodic'} "
                    f"model training (cycle {self._cycle_count})"
                )
                self.train_model(df)

            df_ind = self.calculate_indicators(df)
            ml_signal, ml_conf = self.predict_signal(df_ind)

            # Blend ML signal with SignalEngine second opinion
            signal, confidence = self._ensemble_signal(symbol, df_ind, ml_signal, ml_conf)

            latest = df_ind.iloc[-1]
            price = float(latest['close'])

            # ── Setup-detector stream ────────────────────────────────────────
            # When the ML stream isn't trading this candle (flat or below its
            # confidence gate), check the deterministic pattern setups. Evaluated
            # once per NEW candle, on the last closed candle — a forming candle's
            # wick/close isn't final and must not trigger pattern entries.
            setup_name = None
            if (new_candle and self.positions.get(symbol) is None
                    and (signal == 0 or confidence < self.min_confidence)):
                s_sig, s_conf, s_name = self._best_setup(symbol, df_ind, signal, confidence)
                if s_sig != 0 and s_conf >= SETUP_MIN_CONFIDENCE:
                    signal, confidence = s_sig, s_conf
                    setup_name = s_name
                    logger.info(
                        f"User {self.user_id}: [{symbol}] SETUP fired: {s_name} "
                        f"{'LONG' if s_sig == 1 else 'SHORT'} conf={s_conf:.0%}"
                    )

            # ── Market Intelligence Engine ────────────────────────────────
            # Runs every cycle regardless of `mie_gate_enabled` so it keeps
            # accumulating observations and re-validating models even while
            # silent — otherwise turning the gate on would start it from zero
            # history at whatever moment someone flips the flag. Only actually
            # allowed to veto a fresh ENTRY (never an exit/reversal check on an
            # open position — see simulate_trade/_exit_logic) and only once it
            # has a validated model to back the opinion up.
            mie_decision = self._mie_cycle(symbol, df_ind, new_candle)
            if (self.mie_gate_enabled and mie_decision is not None and signal != 0
                    and self.positions.get(symbol) is None
                    and mie_decision.action == MIE_DO_NOTHING and mie_decision.forecast is not None):
                logger.info(
                    f"User {self.user_id}: [{symbol}] MIE gate vetoed entry "
                    f"(quality {mie_decision.quality}/100): "
                    f"{'; '.join(mie_decision.blockers) or 'no validated positive-EV setup'}")
                signal, confidence = 0, 0.5

            signal_data = {
                'signal': int(signal),
                'confidence': float(confidence),
                'price': price,
                'symbol': symbol,
                'rsi': float(latest.get('rsi', 0)) if pd.notna(latest.get('rsi')) else 0,
                'macd': float(latest.get('macd', 0)) if pd.notna(latest.get('macd')) else 0,
                'adx': float(latest.get('adx', 0)) if pd.notna(latest.get('adx')) else 0,
                'regime': self.market_regime,
                'time': datetime.now().isoformat(),
            }
            self.signals_history.append(signal_data)
            if len(self.signals_history) > 100:
                self.signals_history = self.signals_history[-100:]

            if self.on_signal:
                self.on_signal(self.user_id, signal, confidence, price,
                               signal_data['rsi'], signal_data['macd'], signal_data['adx'])

            # ── Market Intelligence: score, gate, then observe ────────────────
            # Order matters. The quality gate runs BEFORE the observation is
            # written so the `traded` flag records what actually happened rather
            # than what we intended — a vetoed candle mislabelled as traded would
            # corrupt the very decisions we most want to learn from.
            obs_ctx = self._observation_context(
                symbol, df_ind, signal, confidence, setup_name, False)

            quality, q_reason = self._trade_quality(symbol, obs_ctx)
            obs_ctx['quality'] = quality
            self._last_quality[symbol] = {'score': quality, 'reason': q_reason}
            signal_data['quality'] = quality
            signal_data['quality_reason'] = q_reason

            # NO TRADE is the default state: without measured evidence of an
            # advantage under these conditions, standing down is the correct
            # outcome, not a missed opportunity.
            if (self.quality_gate_enabled and signal != 0
                    and self.positions.get(symbol) is None
                    and quality < self.QUALITY_MIN_SCORE):
                logger.info(
                    f"User {self.user_id}: [{symbol}] NO TRADE — quality {quality}/100 "
                    f"< {self.QUALITY_MIN_SCORE} ({q_reason})"
                )
                signal = 0

            # Record every candle, traded or not. Observations restricted to
            # entries would be selected on the decision under evaluation; the
            # skipped candles are the control group that makes the analysis valid.
            if new_candle:
                will_trade = (signal != 0 and self.positions.get(symbol) is None
                              and (confidence >= self.min_confidence or setup_name))
                self._record_observation(
                    symbol, df_ind, signal, confidence, setup_name,
                    bool(will_trade), ctx=obs_ctx)
                self._backfill_observations(symbol, df_ind)

            with self._trade_lock:
                if self.simulation_mode:
                    self.simulate_trade(signal, price, confidence, symbol=symbol, df=df_ind,
                                        setup_name=setup_name, obs_ctx=obs_ctx, quality=quality)
                else:
                    self.execute_live_trade(signal, price, confidence, symbol=symbol, df=df_ind,
                                            setup_name=setup_name, obs_ctx=obs_ctx, quality=quality)

            results.append(signal_data)

        self.current_symbol_index += 1
        return results[-1] if results else None


# ── Parameter Optimizer ──────────────────────────────────────────────────────

class ParameterOptimizer:
    """
    Stratified random search across timeframes.
    Trains model once per (coin, timeframe) pair, then sweeps parameter combinations.
    Scores by ROI × win-rate × trade-count, penalised for excess drawdown.
    """

    # Intraday swing focus: leverage capped at 10x (above this, an adverse intraday
    # wick liquidates the position before the thesis can play out), timeframes capped
    # at 2h (4h+ is positional, not intraday). Tighter search space also cuts overfitting.
    LEVERAGES = [3, 5, 8, 10]
    RISK_PER_TRADE = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
    STOP_LOSS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]   # % of margin
    TAKE_PROFIT = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.50]  # % of margin
    COOLDOWNS = [60, 180, 300, 600, 900]
    CONFIDENCES = [x / 100 for x in range(60, 90, 5)]  # raised floor from 55% to 60%
    TIMEFRAMES = ['5m', '15m', '30m', '1h', '2h']
    TRAILING_STOPS = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25]  # % of margin
    PROFIT_MULTIPLIERS = [1.0, 1.25, 1.5, 2.0, 2.5]
    ADX_THRESHOLDS = [8, 10, 13, 16, 18, 20, 23, 25]  # per-token optimized entry trend filter

    MIN_TRADES = 20

    # Walk-forward validation: train on an expanding window, test on the next
    # out-of-sample segment, and roll forward. A config is only good if it works
    # across MULTIPLE forward windows — this is what separates a durable edge from
    # a curve-fit to one lucky period.
    WALK_FORWARD_FOLDS = 4           # number of out-of-sample test segments
    MIN_WALK_FORWARD_FOLDS = 2       # fall back to single split below this

    # ── Two-phase search budget ────────────────────────────────────────────────
    SAMPLES_PHASE1      = 50    # broad random sweep per timeframe
    SAMPLES_PHASE2      = 70    # neighborhood refinement around top-phase1 configs
    TOP_FOR_REFINEMENT  = 7     # take top-N from phase-1 for neighbor generation

    # ── SL-aware model training ────────────────────────────────────────────────
    # Labels are trained with bucket-representative params so the model actually
    # learns the right exit-regime boundary.  Three buckets cover the whole grid.
    _SL_TIGHT_THRESHOLD  = 0.010   # effective price-move SL < 1%  (tight stop)
    _SL_MEDIUM_THRESHOLD = 0.030   # effective price-move SL 1%-3% (medium)
    # ≥ 3% = wide stop
    # Representative price-move SL per bucket (SL%/lev): tight≈0.5%, medium≈2%, wide≈5%.
    # All use lev=10 (the new ceiling) so labels reflect the regimes we actually trade.
    _BUCKET_SL  = {'tight': 0.05, 'medium': 0.20, 'wide': 0.50}
    _BUCKET_LEV = {'tight': 10,   'medium': 10,   'wide': 10  }
    _BUCKET_TP  = {'tight': 0.20, 'medium': 0.30, 'wide': 0.50}

    @staticmethod
    def _sl_bucket(params: dict) -> str:
        """Map a param set to its SL sensitivity bucket."""
        ratio = params['stop_loss_pct'] / max(1, params['leverage'])
        if ratio < ParameterOptimizer._SL_TIGHT_THRESHOLD:
            return 'tight'
        if ratio < ParameterOptimizer._SL_MEDIUM_THRESHOLD:
            return 'medium'
        return 'wide'

    def __init__(self, user_id: int, selected_coins: list, starting_balance: float = 10000,
                 api_key: str = None, api_secret: str = None, api_password: str = None,
                 max_leverage: int = None, fixed_params: dict = None):
        self.user_id = user_id
        self.selected_coins = selected_coins
        self.starting_balance = starting_balance
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_password = api_password
        # Never let the optimizer test leverage above the user's configured cap.
        self._max_leverage = max_leverage
        self.ohlcv_cache = {}
        self.labeled_cache = {}   # (symbol, timeframe, bucket) → labeled df
        self.model_cache = {}     # (symbol, timeframe, bucket) → fold artifacts
        self.progress = 0
        self.total_tests = 0
        self.current_test = 0
        self.results = []
        self.phase = 'idle'

        # ── Curve-fit knobs, frozen ──────────────────────────────────────────
        # Grid-searching risk_per_trade/stop_loss/take_profit/leverage/confidence
        # etc. for the best backtested ROI is exactly the overfitting the Market
        # Intelligence Engine exists to move away from — a 1,350-config sweep
        # over free parameters will always find SOMETHING that looks good on
        # historical data, whether or not it generalizes. `fixed_params` pins
        # every one of those knobs to the caller's actual configured values, so
        # this class only ever searches TIMEFRAME — a single, low-dimensional,
        # walk-forward-validated comparison, not a curve fit. Passing None
        # preserves the old unconstrained behavior for callers that still want
        # it (none remain in this codebase, but the option stays honest rather
        # than silently unreachable).
        self._fixed_params = dict(fixed_params) if fixed_params else {}
        if self._fixed_params:
            # Nothing left to randomly vary once every risk knob is pinned —
            # sampling the search space 120x/timeframe would just re-run the
            # identical backtest 120 times. One deterministic pass per
            # timeframe is both correct and honest about what changed.
            self.SAMPLES_PHASE1 = 1
            self.SAMPLES_PHASE2 = 0

    def _random_params(self):
        import random
        # Respect the user's leverage cap — never test above what they configured.
        valid_leverages = [l for l in self.LEVERAGES if self._max_leverage is None or l <= self._max_leverage]
        if not valid_leverages:
            valid_leverages = [min(self.LEVERAGES)]
        for _ in range(50):
            params = {
                'leverage': random.choice(valid_leverages),
                'risk_per_trade': random.choice(self.RISK_PER_TRADE),
                'stop_loss_pct': random.choice(self.STOP_LOSS),
                'take_profit_pct': random.choice(self.TAKE_PROFIT),
                'trade_cooldown': random.choice(self.COOLDOWNS),
                'min_confidence': random.choice(self.CONFIDENCES),
                'trailing_stop_pct': random.choice(self.TRAILING_STOPS),
                'profit_risk_multiplier': random.choice(self.PROFIT_MULTIPLIERS),
                'adx_threshold': random.choice(self.ADX_THRESHOLDS),
            }
            params.update(self._fixed_params)
            # Enforce minimum 1.5:1 reward:risk — required for sustainable compounding
            if params['take_profit_pct'] / params['stop_loss_pct'] >= 1.5:
                return params
        return params  # fallback (extremely rare after 50 attempts)

    def _neighbour_params(self, base: dict) -> list:
        """All valid one-step perturbations of a config (one param at a time ±1 step).
        Used in phase-2 of the search to concentrate budget near high-scoring configs
        found in phase-1, rather than continuing to throw uniform random darts."""
        import random as _random
        valid_leverages = [l for l in self.LEVERAGES
                           if self._max_leverage is None or l <= self._max_leverage]
        spaces = [
            ('leverage',              valid_leverages),
            ('risk_per_trade',        self.RISK_PER_TRADE),
            ('stop_loss_pct',         self.STOP_LOSS),
            ('take_profit_pct',       self.TAKE_PROFIT),
            ('trade_cooldown',        self.COOLDOWNS),
            ('min_confidence',        self.CONFIDENCES),
            ('trailing_stop_pct',     self.TRAILING_STOPS),
            ('profit_risk_multiplier', self.PROFIT_MULTIPLIERS),
            ('adx_threshold',         self.ADX_THRESHOLDS),
        ]
        results = []
        for key, space in spaces:
            val = base.get(key)
            if val not in space:
                continue
            if key in self._fixed_params:
                continue   # frozen — not a dimension this search is allowed to move
            idx = space.index(val)
            for new_idx in (idx - 1, idx + 1):
                if 0 <= new_idx < len(space):
                    candidate = {**base, key: space[new_idx]}
                    candidate.update(self._fixed_params)
                    if candidate['take_profit_pct'] / candidate['stop_loss_pct'] >= 1.5:
                        results.append(candidate)
        _random.shuffle(results)
        return results

    def _cache_ohlcv(self, symbol: str, timeframe: str, days: int = 30):
        """Fetch OHLCV and compute indicators — labels are NOT applied here.
        Labels are bucket-specific (different SL/lev → different exit boundaries)
        and are applied in _get_labeled_df."""
        key = (symbol, timeframe)
        if key not in self.ohlcv_cache:
            bot = TradingService(
                user_id=self.user_id, starting_balance=self.starting_balance,
                selected_coins=[symbol], timeframe=timeframe,
                api_key=self.api_key, api_secret=self.api_secret, api_password=self.api_password,
            )
            minutes = bot._get_timeframe_minutes()
            periods = int(days * 24 * 60 / minutes)
            df = bot.fetch_ohlcv(symbol=symbol, limit=min(periods, LOOKBACK_PERIODS * 3))
            if df is not None:
                df = bot.calculate_indicators(df)
                # Drop only rows with missing price/volume; keep label NaNs for now
                df = df.dropna(subset=['close', 'volume', 'open', 'high', 'low'])
            self.ohlcv_cache[key] = df
        return self.ohlcv_cache.get(key)

    def _get_labeled_df(self, symbol: str, timeframe: str, bucket: str):
        """Apply create_labels with bucket-representative SL/leverage, then dropna.
        Cached per (symbol, timeframe, bucket) — label quality matches the exit
        regime the model is being asked to predict."""
        key = (symbol, timeframe, bucket)
        if key in self.labeled_cache:
            return self.labeled_cache[key]
        raw_df = self.ohlcv_cache.get((symbol, timeframe))
        if raw_df is None:
            self.labeled_cache[key] = None
            return None
        label_bot = TradingService(
            user_id=self.user_id,
            starting_balance=self.starting_balance,
            selected_coins=[symbol],
            timeframe=timeframe,
            stop_loss_pct=self._BUCKET_SL[bucket],
            leverage=self._BUCKET_LEV[bucket],
            take_profit_pct=self._BUCKET_TP[bucket],
        )
        df = raw_df.copy()
        df = label_bot.create_labels(df)
        df = df.dropna()
        self.labeled_cache[key] = df
        return df

    def _fit_fold(self, bot, train_df, test_df, is_tree):
        """Train one walk-forward fold and return its cached artifacts, or None if
        the training window is too thin to fit a meaningful model."""
        X_train, _ = bot.prepare_features(train_df)
        y_train = train_df['signal']
        mask = y_train != 0
        X_train, y_train = X_train[mask], y_train[mask]
        if len(X_train) < 20 or len(y_train[y_train == 1]) < 2 or len(y_train[y_train == -1]) < 2:
            return None
        try:
            imp = SimpleImputer(strategy='mean')
            sc = StandardScaler()
            X_imp = imp.fit_transform(X_train)
            X_sc = sc.fit_transform(X_imp)
            try:
                smote = SMOTE(random_state=42,
                              k_neighbors=min(3, len(y_train[y_train == 1]) - 1, len(y_train[y_train == -1]) - 1))
                X_res, y_res = smote.fit_resample(X_sc, y_train)
            except Exception:
                X_res, y_res = X_sc, y_train

            model = bot._build_model()
            if is_tree:
                model.fit(X_res, _encode_y(y_res))
            else:
                tscv = TimeSeriesSplit(n_splits=3)
                grid = GridSearchCV(model, SVM_PARAMS, cv=tscv, scoring='f1_weighted', n_jobs=1)
                grid.fit(X_res, y_res)
                model = grid.best_estimator_

            X_test, _ = bot.prepare_features(test_df)
            return {'model': model, 'imputer': imp, 'scaler': sc,
                    'test_df': test_df, 'X_test': X_test, 'is_tree': is_tree}
        except Exception:
            return None

    def _train_and_cache_model(self, symbol: str, timeframe: str, days: int = 30):
        """Train one walk-forward ensemble per SL bucket for (symbol, timeframe).

        Labels differ by bucket (different SL/leverage → different exit boundaries),
        so the model for each bucket actually learns the right target for the param
        combos that land in that bucket.  Three models per (symbol, tf) instead of
        one: ~3× training time, but results translate to production correctly.
        """
        # Shared feature/model bot — prepare_features and _build_model don't depend on SL/lev
        feature_bot = TradingService(
            user_id=self.user_id, starting_balance=self.starting_balance,
            selected_coins=[symbol], timeframe=timeframe,
        )
        is_tree = LGBM_AVAILABLE or XGB_AVAILABLE or HGBT_AVAILABLE

        for bucket in ('tight', 'medium', 'wide'):
            key = (symbol, timeframe, bucket)
            if key in self.model_cache:
                continue

            df = self._get_labeled_df(symbol, timeframe, bucket)
            if df is None or len(df) < 100:
                self.model_cache[key] = None
                continue

            n_blocks = self.WALK_FORWARD_FOLDS + 1
            block = len(df) // n_blocks
            folds = []
            if block >= 40:
                for k in range(self.WALK_FORWARD_FOLDS):
                    train_df = df.iloc[: (k + 1) * block]
                    test_df  = df.iloc[(k + 1) * block : (k + 2) * block]
                    if len(test_df) < 20:
                        continue
                    fold = self._fit_fold(feature_bot, train_df, test_df, is_tree)
                    if fold is not None:
                        folds.append(fold)

            if len(folds) < self.MIN_WALK_FORWARD_FOLDS:
                half = len(df) // 2
                fold = self._fit_fold(feature_bot, df.iloc[:half], df.iloc[half:], is_tree)
                folds = [fold] if fold is not None else []

            if not folds:
                self.model_cache[key] = None
                continue

            self.model_cache[key] = {'folds': folds, 'temp_bot': feature_bot, 'is_tree': is_tree}
            logger.info(f"Optimizer: {symbol} {timeframe} [{bucket}] — {len(folds)} fold(s)")

    def _backtest_segment(self, fold, bot, params, start_balance):
        """Backtest one walk-forward fold's out-of-sample test_df under `params`.
        Returns (trades, final_balance, max_dd_pct). Mirrors the live entry/exit
        engine exactly — maker entries, breakeven floor, reversal hysteresis,
        cooldown gating entries only — so results translate 1:1 to production."""
        SLIP = 0.0005  # 5 bps slippage on the taker (exit) leg
        model = fold['model']; imp = fold['imputer']; sc = fold['scaler']
        test_df = fold['test_df']; X_test = fold['X_test']; is_tree = fold['is_tree']

        balance = start_balance
        peak_balance = start_balance
        max_dd = 0.0
        position = None
        entry_price = 0
        hwm = lwm = 0
        trades = []
        last_trade_candle = -999
        entry_candle = -999
        reversal_streak = 0
        be_armed = False
        minutes = bot._get_timeframe_minutes()
        cooldown_candles = max(1, math.ceil(params['trade_cooldown'] / 60 / minutes))
        MIN_HOLD_CANDLES = 3   # matches live: 3 × timeframe before reversal exits

        opt_lev = max(1, params.get('leverage', 1))
        # Maker entry + taker exit — matches the live execution model
        fee_m = (MAKER_FEE + TAKER_FEE) * opt_lev
        net_win = max(0.001, min(params['take_profit_pct'], 0.25) - fee_m)
        net_loss = params['stop_loss_pct'] + fee_m
        be_trigger = max(4 * fee_m, 0.05)
        be_floor = 2 * fee_m

        for i in range(len(test_df)):
            try:
                row = test_df.iloc[i]
                price = row['close']
                X_row = X_test.iloc[[i]]
                X_sc = sc.transform(imp.transform(X_row))
                raw = model.predict(X_sc)[0]
                sig_val = int(_decode_y([raw])[0]) if is_tree else int(raw)
                conf = float(max(model.predict_proba(X_sc)[0]))

                if position is None:
                    # Cooldown gates ENTRIES only (matches live fix)
                    if i - last_trade_candle < cooldown_candles:
                        continue
                    adx_o = row.get('adx', 25); adx_o = 25.0 if pd.isna(adx_o) else float(adx_o)
                    vol_o = row.get('volume_ratio', 1.0); vol_o = 1.0 if pd.isna(vol_o) else float(vol_o)
                    ev_o = conf * net_win - (1 - conf) * net_loss
                    if sig_val != 0 and conf >= params['min_confidence'] and adx_o >= params['adx_threshold'] and vol_o >= 0.65 and ev_o > 0:
                        conf_rng_o = max(0.01, 1.0 - params['min_confidence'])
                        c_scale_o = max(0.5, min(1.5, 0.5 + (conf - params['min_confidence']) / conf_rng_o))
                        risk_ceil_o = min(max(params['risk_per_trade'] * 2.0, 0.10), 0.75)
                        margin = min(balance * params['risk_per_trade'] * c_scale_o, balance * risk_ceil_o)
                        if margin <= 0 or margin > balance * 0.95:
                            continue
                        notional = margin * params['leverage']
                        size = notional / price
                        entry_fee = notional * MAKER_FEE   # post-only limit entry
                        position = {'side': 'long' if sig_val == 1 else 'short',
                                    'size': size, 'margin': margin, 'entry_fee': entry_fee}
                        entry_price = price
                        hwm = lwm = price
                        last_trade_candle = i
                        entry_candle = i
                        reversal_streak = 0
                        be_armed = False

                else:
                    if position['side'] == 'long':
                        hwm = max(hwm, price)
                    else:
                        lwm = min(lwm, price)

                    price_pnl_pct = ((price - entry_price) / entry_price if position['side'] == 'long'
                                     else (entry_price - price) / entry_price)
                    margin_pnl_pct = price_pnl_pct * opt_lev
                    trail_price_pct = params.get('trailing_stop_pct', 0.01) / opt_lev
                    should_exit = False

                    if margin_pnl_pct <= -params['stop_loss_pct']:
                        should_exit = True
                    elif margin_pnl_pct >= params['take_profit_pct']:
                        should_exit = True
                    else:
                        # Breakeven floor — winner can't become a loser
                        if not be_armed and margin_pnl_pct >= be_trigger:
                            be_armed = True
                        if be_armed and margin_pnl_pct <= be_floor:
                            should_exit = True
                        # Time stop — thesis expired, recycle capital (matches live)
                        if not should_exit and not be_armed and (i - entry_candle) >= 8:
                            should_exit = True
                        if not should_exit and price_pnl_pct > 0:
                            if position['side'] == 'long' and price <= hwm * (1 - trail_price_pct):
                                should_exit = True
                            elif position['side'] == 'short' and price >= lwm * (1 + trail_price_pct):
                                should_exit = True
                        if not should_exit:
                            # Reversal hysteresis: full entry confidence, min-hold,
                            # 2 consecutive reversal candles (matches live)
                            is_rev = (sig_val != 0 and
                                      ((sig_val == -1 and position['side'] == 'long') or
                                       (sig_val == 1 and position['side'] == 'short')))
                            if is_rev and conf >= params['min_confidence'] and (i - entry_candle) >= MIN_HOLD_CANDLES:
                                reversal_streak += 1
                                if reversal_streak >= 2:
                                    should_exit = True
                            else:
                                reversal_streak = 0

                    if should_exit:
                        price_change = ((price - entry_price) if position['side'] == 'long'
                                        else (entry_price - price))
                        pnl_amount = price_change * position['size']
                        exit_fee = position['size'] * price * (TAKER_FEE + SLIP)
                        net_pnl = pnl_amount - position['entry_fee'] - exit_fee
                        balance += net_pnl
                        peak_balance = max(peak_balance, balance)
                        dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
                        max_dd = max(max_dd, dd)
                        trades.append({'pnl': net_pnl})
                        position = None
                        last_trade_candle = i
                        reversal_streak = 0
                        be_armed = False
            except Exception:
                continue

        return trades, balance, max_dd

    def _run_cached_backtest(self, timeframe: str, params: dict):
        """
        Walk-forward backtest: for each coin, run every cached fold's out-of-sample
        segment, then aggregate. Uses the bucket-matched model so the signal was
        trained on the same exit-regime we're now testing. Per-fold returns are
        MARGINAL (balance change within the fold, not cumulative from start) so the
        walk-forward consistency metric is meaningful across all folds.
        """
        bucket = self._sl_bucket(params)
        all_trades = []
        total_balance = 0
        overall_max_dd = 0
        balance_per_coin = self.starting_balance / len(self.selected_coins)

        # fold index → marginal P&L and starting balance (for consistency scoring)
        fold_pnl:     dict = {}
        fold_start_b: dict = {}

        for symbol in self.selected_coins:
            # Pick the bucket-matched model; fall back to adjacent buckets rather
            # than skipping the coin entirely.
            cached = self.model_cache.get((symbol, timeframe, bucket))
            if cached is None:
                for fb in ('medium', 'tight', 'wide'):
                    cached = self.model_cache.get((symbol, timeframe, fb))
                    if cached is not None:
                        break
            if cached is None:
                total_balance += balance_per_coin
                continue

            bot = cached['temp_bot']
            folds = cached['folds']
            # Compound each fold off the previous fold's ending balance — true
            # forward simulation, not N independent restarts.
            balance = balance_per_coin
            for fi, fold in enumerate(folds):
                b_before = balance
                trades, balance, fold_dd = self._backtest_segment(fold, bot, params, balance)
                all_trades.extend(trades)
                overall_max_dd = max(overall_max_dd, fold_dd)
                # Marginal: how much did THIS fold earn relative to its own start?
                fold_pnl[fi]     = fold_pnl.get(fi, 0.0)     + (balance - b_before)
                fold_start_b[fi] = fold_start_b.get(fi, 0.0) + b_before
            total_balance += balance

        total_trades = len(all_trades)
        winning = len([t for t in all_trades if t['pnl'] > 0])
        total_pnl = sum(t['pnl'] for t in all_trades)
        total_return_pct = ((total_balance - self.starting_balance) / self.starting_balance) * 100
        win_rate = (winning / total_trades * 100) if total_trades > 0 else 0

        sharpe_ratio = 0.0
        if total_trades > 1:
            tret = [t['pnl'] / max(self.starting_balance, 1) for t in all_trades]
            sharpe_ratio = float(np.mean(tret) / (np.std(tret) + 1e-10) * np.sqrt(total_trades))

        # Walk-forward consistency: per-fold MARGINAL % return across all coins.
        # Marginal means we measure how much each fold earned relative to its own
        # starting balance — not cumulative from the beginning — so a bad fold can't
        # hide behind the gains of earlier folds.
        fold_returns = [
            round(fold_pnl.get(fi, 0.0) / max(fold_start_b.get(fi, 1.0), 1.0) * 100, 3)
            for fi in sorted(fold_pnl.keys())
        ]
        if fold_returns:
            profitable_folds = sum(1 for r in fold_returns if r > 0)
            wf_consistency = profitable_folds / len(fold_returns)
            wf_return_std = float(np.std(fold_returns))
            wf_worst_fold = float(min(fold_returns))
        else:
            wf_consistency = 0.0
            wf_return_std = 0.0
            wf_worst_fold = 0.0

        return {
            'total_return': total_return_pct,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': winning,
            'win_rate': win_rate,
            'final_balance': total_balance,
            'max_drawdown': overall_max_dd,
            'sharpe_ratio': sharpe_ratio,
            'wf_folds': len(fold_returns),
            'wf_consistency': round(wf_consistency, 3),
            'wf_return_std': round(wf_return_std, 3),
            'wf_worst_fold': round(wf_worst_fold, 3),
            'fold_returns': [round(r, 3) for r in fold_returns],
        }

    def _calculate_score(self, result: dict) -> float:
        if result['total_trades'] < self.MIN_TRADES:
            return -999
        if result['win_rate'] < 42:        # below this it's statistical noise
            return -999
        if result['total_return'] < -5:    # small tolerance for rounding / fee noise
            return -999

        # Walk-forward gate: reject configs where ANY marginal forward fold lost more
        # than 5% (uses true marginal fold returns, so a bad fold can't hide behind
        # earlier compounding). Relaxed from -3% to -5% to reflect that fold windows
        # are ~7.5 days of test data — a single streak of losing trades is enough to
        # breach -3% even in a durable regime.
        wf_folds = result.get('wf_folds', 0)
        if wf_folds >= self.MIN_WALK_FORWARD_FOLDS and result.get('wf_worst_fold', 0) < -5.0:
            return -999

        # Log-scale ROI: properly differentiates 50% vs 200% vs 400% returns.
        # Calibrated so ~300% total return → roi_score = 1.0.  Linear capping at
        # 100% was discarding the most interesting high-leverage regimes.
        roi_score = min(math.log1p(max(0, result['total_return'])) / math.log1p(300), 1.0)

        # Win rate normalized: 42% floor → 0, 90% → 1
        winrate_score = max(0.0, (result['win_rate'] - 42) / 48)

        # Trade activity: enough signal to be meaningful, but not overfit to high-freq
        trade_score = min(result['total_trades'] / 80, 1.0)

        # Calmar ratio: return / max_drawdown — the compounding metric.
        # Most important single number: a 100% return with 5% drawdown beats a
        # 300% return with 60% drawdown for sustainable compounding.
        max_dd = max(result.get('max_drawdown', 0.1), 0.1)
        calmar_score = min(result['total_return'] / max_dd, 8.0) / 8.0

        # Sharpe ratio: trade-level consistency (capped at Sharpe 3.0 → 1.0)
        sharpe_score = max(0.0, min(result.get('sharpe_ratio', 0) / 3.0, 1.0))

        # Walk-forward consistency: fraction of forward folds that were profitable.
        wf_score = result.get('wf_consistency', 0.0) if wf_folds >= self.MIN_WALK_FORWARD_FOLDS else 0.5

        dd_penalty = max(0.0, (result.get('max_drawdown', 0) - 15) / 100)
        wf_penalty = min(0.15, result.get('wf_return_std', 0.0) / 200) if wf_folds >= self.MIN_WALK_FORWARD_FOLDS else 0.0

        return (
            0.20 * roi_score        # log-scale ROI
            + 0.10 * winrate_score  # win rate (floored at 42%)
            + 0.05 * trade_score    # activity (secondary signal)
            + 0.30 * calmar_score   # return/drawdown — primary compounding metric
            + 0.15 * sharpe_score   # risk-adjusted consistency
            + 0.20 * wf_score       # forward generalization
            - dd_penalty
            - wf_penalty
        )

    def optimize(self, days: int = 30, progress_callback=None):
        import random
        import time as _time
        print(f"[OPT] User {self.user_id}: optimize() started, days={days}", flush=True)
        random.seed(42)

        self.results = []
        total_per_tf = self.SAMPLES_PHASE1 + self.SAMPLES_PHASE2
        self.total_tests = len(self.TIMEFRAMES) * total_per_tf
        self.current_test = 0

        logger.info(
            f"Starting optimization: {self.total_tests} tests across {len(self.TIMEFRAMES)} "
            f"timeframes ({self.SAMPLES_PHASE1} random + {self.SAMPLES_PHASE2} guided per tf)"
        )

        # ── Step 1: fetch OHLCV + indicators, one (coin, tf) pair ─────────────
        total_fetches = len(self.TIMEFRAMES) * len(self.selected_coins)
        fetch_count = 0
        self.ohlcv_cache  = {}
        self.labeled_cache = {}
        self.model_cache  = {}
        self.phase = 'fetching'

        for tf_idx, timeframe in enumerate(self.TIMEFRAMES):
            logger.info(f"Fetching data: {timeframe} ({tf_idx + 1}/{len(self.TIMEFRAMES)})")
            for symbol in self.selected_coins:
                fetch_count += 1
                self.progress = (fetch_count / total_fetches) * 15
                if progress_callback:
                    progress_callback(self.progress)
                try:
                    self._cache_ohlcv(symbol, timeframe, days)
                    _time.sleep(0.2)
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol} {timeframe}: {e}")

        logger.info(f"Data fetch complete: {len(self.ohlcv_cache)} datasets cached")

        # ── Step 2: train one model per (coin, tf, bucket) ────────────────────
        self.phase = 'training'
        total_models = len(self.TIMEFRAMES) * len(self.selected_coins)
        model_count = 0
        for tf_idx, timeframe in enumerate(self.TIMEFRAMES):
            for symbol in self.selected_coins:
                model_count += 1
                self.progress = 15 + (model_count / total_models) * 20
                if progress_callback:
                    progress_callback(self.progress)
                self._train_and_cache_model(symbol, timeframe, days)

        logger.info(f"Models trained: {sum(1 for v in self.model_cache.values() if v is not None)} valid buckets")

        # ── Step 3: two-phase param search ────────────────────────────────────
        self.phase = 'testing'

        def _record(params, result, score):
            self.results.append({
                'timeframe':             timeframe,
                'leverage':              params['leverage'],
                'risk_per_trade':        params['risk_per_trade'],
                'stop_loss_pct':         params['stop_loss_pct'],
                'take_profit_pct':       params['take_profit_pct'],
                'trailing_stop_pct':     params['trailing_stop_pct'],
                'profit_risk_multiplier': params['profit_risk_multiplier'],
                'trade_cooldown':        params['trade_cooldown'],
                'min_confidence':        params['min_confidence'],
                'adx_threshold':         params['adx_threshold'],
                'total_return':   round(result['total_return'], 2),
                'win_rate':       round(result['win_rate'], 2),
                'total_trades':   result['total_trades'],
                'total_pnl':      round(result['total_pnl'], 2),
                'max_drawdown':   round(result.get('max_drawdown', 0), 2),
                'wf_folds':       result.get('wf_folds', 0),
                'wf_consistency': result.get('wf_consistency', 0.0),
                'wf_worst_fold':  result.get('wf_worst_fold', 0.0),
                'fold_returns':   result.get('fold_returns', []),
                'score':          round(score, 4),
            })

        for tf_idx, timeframe in enumerate(self.TIMEFRAMES):
            logger.info(f"Searching timeframe: {timeframe} ({tf_idx + 1}/{len(self.TIMEFRAMES)})")

            # Phase A: broad random sweep
            phase1_hits = []
            for _ in range(self.SAMPLES_PHASE1):
                self.current_test += 1
                self.progress = 35 + (self.current_test / self.total_tests) * 65
                if progress_callback:
                    progress_callback(self.progress)
                params = self._random_params()
                try:
                    result = self._run_cached_backtest(timeframe, params)
                    score  = self._calculate_score(result)
                    if score > -999:
                        _record(params, result, score)
                        phase1_hits.append((score, params))
                except Exception as e:
                    logger.debug(f"Backtest failed: {e}")

            # Phase B: neighborhood refinement around the top-scoring configs.
            # Skipped entirely once every risk knob is frozen (SAMPLES_PHASE2=0,
            # see __init__) — there is no neighborhood left to refine when
            # timeframe is the only thing this search is allowed to move.
            if self.SAMPLES_PHASE2 > 0:
                phase1_hits.sort(key=lambda x: x[0], reverse=True)
                seen = set()
                candidates = []
                for _, top_params in phase1_hits[:self.TOP_FOR_REFINEMENT]:
                    for nb in self._neighbour_params(top_params):
                        key = (nb['leverage'], nb['stop_loss_pct'], nb['take_profit_pct'],
                               nb['trade_cooldown'], nb['min_confidence'])
                        if key not in seen:
                            seen.add(key)
                            candidates.append(nb)

                # Pad with fresh random samples if there weren't enough valid neighbours
                while len(candidates) < self.SAMPLES_PHASE2:
                    candidates.append(self._random_params())

                for params in candidates[:self.SAMPLES_PHASE2]:
                    self.current_test += 1
                    self.progress = 35 + (self.current_test / self.total_tests) * 65
                    if progress_callback:
                        progress_callback(self.progress)
                    try:
                        result = self._run_cached_backtest(timeframe, params)
                        score  = self._calculate_score(result)
                        if score > -999:
                            _record(params, result, score)
                    except Exception as e:
                        logger.debug(f"Backtest failed: {e}")

            logger.info(
                f"  {timeframe}: {sum(1 for r in self.results if r['timeframe'] == timeframe)} "
                f"valid configs found so far"
            )

        self.results.sort(key=lambda x: x['score'], reverse=True)
        top_results = self.results[:20]
        logger.info(f"Optimization complete — {len(self.results)} valid configs found")

        return {
            'total_tested': self.current_test,
            'valid_configs': len(self.results),
            'top_configs': top_results,
            'selected_coins': self.selected_coins,
            'days_tested': days,
        }
