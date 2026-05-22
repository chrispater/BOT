import os
import sys
import math
import pickle
import numpy as np
import pandas as pd
import ccxt
import time
import logging
import talib
from datetime import datetime
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

os.environ['LOKY_MAX_CPU_COUNT'] = '1'

_MODEL_DIR = os.environ.get('BOT_MODEL_DIR', '/tmp/bot_models')
os.makedirs(_MODEL_DIR, exist_ok=True)

# ── Optional high-performance models (LightGBM > XGBoost > SVM fallback) ──
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ── Optional signal engine (second-opinion ensemble) ──
try:
    _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from signal_engine import SignalEngine
    SIGNAL_ENGINE_AVAILABLE = True
except Exception:
    SIGNAL_ENGINE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

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

VALID_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']

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
    def __init__(self, user_id: int, api_key=None, api_secret=None, api_password=None,
                 starting_balance=10000, leverage=10, selected_coins=None,
                 risk_per_trade=0.02, stop_loss_pct=0.015, take_profit_pct=0.03,
                 trade_cooldown=300, min_confidence=0.65, timeframe='5m',
                 trailing_stop_pct=0.01, max_drawdown_pct=0.20,
                 retrain_every=50, profit_risk_multiplier=1.5,
                 adx_threshold=18,
                 on_trade=None, on_signal=None, on_performance=None):

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

        # Second-opinion signal engine
        self._signal_engine = SignalEngine() if SIGNAL_ENGINE_AVAILABLE else None

        self.on_trade = on_trade
        self.on_signal = on_signal
        self.on_performance = on_performance

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
        sl        = self.stop_loss_pct

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
        df.loc[(df['future_return'] > threshold) & (df['future_min_pct'] > -sl), 'signal'] = 1
        df.loc[(df['future_return'] < -threshold) & (df['future_max_pct'] < sl), 'signal'] = -1

        # Fallback: if clean labeling is too sparse, use plain threshold so the
        # model always has enough examples to train.
        labeled = int((df['signal'] != 0).sum())
        if labeled < 50:
            logger.warning(
                f"User {self.user_id}: Clean labels sparse ({labeled} rows) — "
                f"falling back to plain threshold labeling"
            )
            df['signal'] = 0
            df.loc[df['future_return'] > threshold, 'signal'] = 1
            df.loc[df['future_return'] < -threshold, 'signal'] = -1

        return df

    def prepare_features(self, df):
        feature_columns = [
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
        ]
        available = [c for c in feature_columns if c in df.columns]
        return df[available].copy(), available

    # ── Model persistence & incremental learning ────────────────────────────

    def _load_persisted_model(self):
        if not os.path.exists(self._model_file):
            return
        try:
            data = pickle.load(open(self._model_file, 'rb'))
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
                'is_tree': self._model_is_tree, 'trained_at': datetime.now().isoformat(),
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
        roi_file = self._buffer_file.replace('.pkl', '_roi.pkl')
        try:
            if os.path.exists(roi_file):
                with open(roi_file, 'rb') as f:
                    data = pickle.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            pass
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
            if len(self._feedback_buffer) > 500:
                self._feedback_buffer = self._feedback_buffer[-500:]
        except Exception:
            pass

    # ── Model ───────────────────────────────────────────────────────────────

    def _build_model(self):
        """Return best available classifier: LightGBM → XGBoost → SVM."""
        if LGBM_AVAILABLE:
            return lgb.LGBMClassifier(**LGBM_PARAMS)
        if XGB_AVAILABLE:
            return xgb.XGBClassifier(**XGB_PARAMS)
        return SVC(probability=True, C=10, gamma='scale', kernel='rbf', random_state=42)

    def train_model(self, df):
        logger.info(f"User {self.user_id}: Training ML model ({'LGBM' if LGBM_AVAILABLE else 'XGB' if XGB_AVAILABLE else 'SVM'})...")
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

        is_tree = LGBM_AVAILABLE or XGB_AVAILABLE
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
        baseline = self._drawdown_baseline or self.starting_balance
        dd = (baseline - self.balance) / max(baseline, 1)
        if dd <= 0.05:
            return 1.00
        if dd <= 0.10:
            return 0.75
        if dd <= 0.15:
            return 0.50
        return 0.25

    def _is_drawdown_exceeded(self):
        baseline = self._drawdown_baseline or self.starting_balance
        drawdown = (baseline - self.balance) / baseline
        if drawdown > self.max_drawdown_pct:
            logger.warning(
                f"User {self.user_id}: Drawdown {drawdown:.1%} > max {self.max_drawdown_pct:.1%} — halting entries"
            )
            return True
        return False

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

    def _calculate_margin(self, confidence: float = None, signal: int = None, df=None) -> float:
        """
        Profit-tier compounding using Kelly fraction, scaled by four independent multipliers:
          1. Confidence: 0.5× at threshold → 1.5× at 90%+
          2. Volatility (ATR-based): high vol → 0.5×, low vol → 1.5×
          3. Market regime: trade with the macro trend, not against it
          4. Drawdown recovery: shrink bets while clawing back losses
        Hard cap: min(max(risk_per_trade × 2, 10%), 75%) of current balance.
        """
        k = self._kelly_fraction()
        profit = max(0.0, self.balance - self.starting_balance)
        base_cap = min(self.balance, self.starting_balance)
        base_margin = base_cap * k
        profit_margin = profit * k * self.profit_risk_multiplier
        margin = base_margin + profit_margin

        # 1. Confidence scaling — aggressive on no-brainers, modest on borderlines
        conf_scale = 1.0
        if confidence is not None and not (confidence != confidence):  # guard NaN
            confidence = max(0.0, min(1.0, confidence))  # clamp to [0,1]
            conf_range = max(0.01, 1.0 - self.min_confidence)
            conf_scale = 0.5 + (confidence - self.min_confidence) / conf_range
            conf_scale = max(0.5, min(1.5, conf_scale))
            margin = margin * conf_scale

        # Cap respects the user's configured risk setting rather than a hardcoded 10%.
        risk_ceiling = min(max(self.risk_per_trade * 2.0, 0.10), 0.75)
        capped = min(margin, self.balance * risk_ceiling)

        # 2. Volatility multiplier: reduce size in choppy/high-ATR conditions
        vol_mult = self._get_volatility_multiplier(df) if df is not None else 1.0

        # 3. Regime multiplier: size with the macro trend, penalise counter-trend trades
        regime_mult = self._get_regime_multiplier(signal) if signal is not None else 1.0

        # 4. Drawdown recovery scale: shrink position as losses grow
        dd_scale = self._get_drawdown_recovery_scale()

        final = capped * vol_mult * regime_mult * dd_scale

        # Track peak balance for drawdown and compound projection accuracy
        self._peak_balance = max(self._peak_balance, self.balance)

        conf_str = f"{confidence:.2f}" if confidence is not None else "n/a"
        logger.info(
            f"User {self.user_id}: [MARGIN] base=${base_cap:.2f} k={k*100:.1f}% | "
            f"profit_tier=${profit:.2f}×{self.profit_risk_multiplier} | "
            f"conf={conf_str} scale={conf_scale:.2f} risk_ceil={risk_ceiling*100:.0f}% | "
            f"vol={vol_mult:.2f}× regime={regime_mult:.2f}× dd_scale={dd_scale:.2f}× → ${final:.2f}"
        )
        return max(1.0, final)

    def _get_volatility_multiplier(self, df) -> float:
        """Scale size inversely to ATR: high vol → 0.5×, low vol → 1.5×. Baseline 0.15% ATR/price."""
        if 'atr' not in df.columns or df['atr'].isna().all():
            return 1.0
        atr = df['atr'].iloc[-1]
        price = df['close'].iloc[-1]
        if pd.isna(atr) or pd.isna(price) or price <= 0 or atr <= 0:
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
        adx = 25.0 if pd.isna(adx_raw) else float(adx_raw)
        vol = 1.0  if pd.isna(vol_raw) else float(vol_raw)
        if adx < self.adx_threshold:
            logger.info(f"User {self.user_id}: Entry blocked — ADX {adx:.1f} < {self.adx_threshold} (weak trend)")
            return False
        if vol < 0.65:
            logger.info(f"User {self.user_id}: Entry blocked — volume_ratio {vol:.2f} < 0.65 (thin volume)")
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
        # Keep all-time peak in sync after every close
        self._peak_balance = max(self._peak_balance, self.balance + pnl)

    def _compound_projection(self):
        """
        Return compound metrics based on actual realized per-trade ROI.
        Uses last 20 closed trades to compute average ROI per trade, then
        projects trades/days to reach milestone targets from current balance.
        """
        recent = self._trade_roi_history[-20:] if self._trade_roi_history else []
        if not recent:
            return None

        avg_roi = sum(recent) / len(recent)  # average ROI per trade (signed fraction)
        if avg_roi <= 0:
            return None  # net negative — can't project toward $1MM

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
        projected_days_to_1m = compound['days_to_1m'] if compound else None

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
            'model_type': 'LGBM' if LGBM_AVAILABLE else 'XGB' if XGB_AVAILABLE else 'SVM',
            'signal_engine_active': SIGNAL_ENGINE_AVAILABLE,
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
            # Drawdown & recovery
            'peak_balance': float(self._peak_balance),
            'dd_recovery_scale': self._get_drawdown_recovery_scale(),
        }

    # ── Trade execution ─────────────────────────────────────────────────────

    def _exit_logic(self, position, price, signal, confidence):
        """Shared exit decision for both sim and live. Returns (should_exit, reason)."""
        entry = position['entry_price']
        pnl_pct = (
            (price - entry) / entry if position['side'] == 'long'
            else (entry - price) / entry
        )

        if pnl_pct <= -self.stop_loss_pct:
            return True, 'Stop Loss'
        if pnl_pct >= self.take_profit_pct:
            return True, 'Take Profit'

        # Trailing stop — only activates once in profit
        if pnl_pct > 0:
            if position['side'] == 'long':
                if price <= position['high_water_mark'] * (1 - self.trailing_stop_pct):
                    return True, 'Trailing Stop'
            else:
                if price >= position['low_water_mark'] * (1 + self.trailing_stop_pct):
                    return True, 'Trailing Stop'

        # Signal reversal exit — lower threshold than entry (80%) to avoid lock-in
        exit_thresh = self.min_confidence * 0.8
        if ((signal == -1 and position['side'] == 'long') or
                (signal == 1 and position['side'] == 'short')):
            if confidence >= exit_thresh:
                return True, 'Signal Reversal'

        return False, ''

    def simulate_trade(self, signal, price, confidence, symbol=None, df=None):
        symbol = symbol or self.get_current_symbol()
        current_time = time.time()
        if current_time - self.last_trade_times.get(symbol, 0) < self.trade_cooldown:
            return

        position = self.positions.get(symbol)

        if position is None and signal != 0 and confidence >= self.min_confidence:
            if self._is_drawdown_exceeded():
                return
            if self._check_performance_floor():
                return
            if not self._entry_filter(signal, df):
                return

            margin = self._calculate_margin(confidence, signal=signal, df=df)
            if margin <= 0:
                return

            dyn_lev = self._dynamic_leverage(confidence)
            notional = margin * dyn_lev
            size = notional / price
            side = 'long' if signal == 1 else 'short'
            conf_tier = 'NO-BRAINER' if confidence >= 0.85 else ('STRONG' if confidence >= 0.75 else 'MODERATE')

            self.positions[symbol] = {
                'side': side, 'size': size, 'entry_price': price,
                'symbol': symbol, 'margin': margin, 'leverage': dyn_lev,
                'high_water_mark': price, 'low_water_mark': price,
            }
            self.entry_price = price
            self.last_trade_times[symbol] = current_time

            trade = {
                'type': 'open', 'side': side, 'size': float(size), 'price': float(price),
                'confidence': float(confidence), 'symbol': symbol, 'margin': float(margin),
                'leverage': dyn_lev, 'conf_tier': conf_tier,
                'regime': self.market_regime, 'time': datetime.now().isoformat(),
            }
            self.trades_history.append(trade)
            if self.on_trade:
                self.on_trade(self.user_id, symbol, side, 'open', size, price, None, confidence, None)
            logger.info(
                f"User {self.user_id}: [SIM] Open {side.upper()} {symbol} @ ${price:.2f} | "
                f"Margin ${margin:.2f} · {dyn_lev}x lev [{conf_tier}]"
            )

        elif position is not None:
            if position['side'] == 'long':
                position['high_water_mark'] = max(position['high_water_mark'], price)
            else:
                position['low_water_mark'] = min(position['low_water_mark'], price)

            should_exit, exit_reason = self._exit_logic(position, price, signal, confidence)

            if should_exit:
                price_change = (price - position['entry_price']) if position['side'] == 'long' else (position['entry_price'] - price)
                pnl = price_change * position['size']
                self.total_pnl += pnl
                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
                self._record_trade_roi(pnl)
                self.balance += pnl

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
                                  position['size'], price, pnl, None, exit_reason)
                if self.on_performance:
                    self.on_performance(self.user_id, self.balance, self.total_pnl,
                                        self.total_trades, self.winning_trades)
                logger.info(
                    f"User {self.user_id}: [SIM] Close {symbol} — {exit_reason} "
                    f"PnL ${pnl:.2f} ({lev_pct:.1f}%) | Balance ${self.balance:.2f}"
                )
                del self.positions[symbol]

    def execute_live_trade(self, signal, price, confidence, symbol=None, df=None):
        symbol = symbol or self.get_current_symbol()
        current_time = time.time()
        cooldown_remaining = self.trade_cooldown - (current_time - self.last_trade_times.get(symbol, 0))
        if cooldown_remaining > 0:
            if confidence >= self.min_confidence:
                logger.info(f"User {self.user_id}: [{symbol}] Signal skipped (conf={confidence:.1%}) — cooldown {cooldown_remaining:.0f}s remaining")
            return

        try:
            position = self.positions.get(symbol)
            sig_label = 'LONG' if signal == 1 else ('SHORT' if signal == -1 else 'FLAT')
            logger.info(f"User {self.user_id}: [{symbol}] Cycle — signal={sig_label} conf={confidence:.1%} threshold={self.min_confidence:.1%} position={'open' if position else 'none'}")

            if position is None and signal != 0 and confidence >= self.min_confidence:
                if self._is_drawdown_exceeded():
                    logger.warning(f"User {self.user_id}: [{symbol}] Entry blocked — max drawdown limit reached")
                    return
                if self._check_performance_floor():
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

                margin = self._calculate_margin(confidence, signal=signal, df=df)
                margin = min(margin, avail * 0.95)

                dyn_lev = self._dynamic_leverage(confidence)
                notional = margin * dyn_lev
                size = notional / price
                side = 'buy' if signal == 1 else 'sell'
                pos_side = 'long' if signal == 1 else 'short'
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

                order = self.exchange.create_order(
                    symbol=symbol, type='market', side=side, amount=amount,
                    params={'posSide': pos_side},
                )
                self.positions[symbol] = {
                    'side': pos_side, 'size': size, 'entry_price': price,
                    'symbol': symbol, 'margin': margin, 'leverage': dyn_lev,
                    'order_id': order.get('id'),
                    'high_water_mark': price, 'low_water_mark': price,
                }
                self.entry_price = price
                self.last_trade_times[symbol] = current_time

                self.trades_history.append({
                    'type': 'open', 'side': pos_side, 'size': float(size), 'price': float(price),
                    'confidence': float(confidence), 'symbol': symbol, 'margin': float(margin),
                    'leverage': dyn_lev, 'conf_tier': conf_tier,
                    'regime': self.market_regime, 'time': datetime.now().isoformat(),
                })
                if self.on_trade:
                    self.on_trade(self.user_id, symbol, pos_side, 'open', size, price, None, confidence, None)
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

                if position['side'] == 'long':
                    position['high_water_mark'] = max(position['high_water_mark'], price)
                else:
                    position['low_water_mark'] = min(position['low_water_mark'], price)

                should_exit, exit_reason = self._exit_logic(position, price, signal, confidence)

                if should_exit:
                    pos_sym = position.get('symbol', symbol)
                    close_side = 'sell' if position['side'] == 'long' else 'buy'
                    market = self.exchange.market(pos_sym)
                    amount = position['size'] / (market.get('contractSize', 1) or 1)

                    self.exchange.create_order(
                        symbol=pos_sym, type='market', side=close_side, amount=amount,
                        params={'posSide': position['side'], 'reduceOnly': True},
                    )
                    price_change = (price - position['entry_price']) if position['side'] == 'long' else (position['entry_price'] - price)
                    pnl = price_change * position['size']
                    self.total_pnl += pnl
                    self.total_trades += 1
                    if pnl > 0:
                        self.winning_trades += 1
                    self._record_trade_roi(pnl)
                    self.balance += pnl

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
                                      position['size'], price, pnl, None, exit_reason)
                    if self.on_performance:
                        self.on_performance(self.user_id, self.balance, self.total_pnl,
                                            self.total_trades, self.winning_trades)
                    logger.info(
                        f"User {self.user_id}: [LIVE] Close {pos_sym} — {exit_reason} "
                        f"PnL ${pnl:.2f} ({lev_pct:.1f}%)"
                    )
                    del self.positions[symbol]
                    self._sync_live_balance()

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

            is_tree = LGBM_AVAILABLE or XGB_AVAILABLE
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
        cooldown_candles = max(1, math.ceil(self.trade_cooldown / 60 / minutes_per_candle))
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

                if i - last_trade_candle < cooldown_candles:
                    continue

                # Entry quality filter — same gate as live trading
                adx_raw = row.get('adx', 25); adx_f = 25.0 if pd.isna(adx_raw) else float(adx_raw)
                vol_raw = row.get('volume_ratio', 1.0); vol_f = 1.0 if pd.isna(vol_raw) else float(vol_raw)
                if position is None and sig_val != 0 and conf >= self.min_confidence and adx_f >= self.adx_threshold and vol_f >= 0.65:
                    # Margin formula mirrors live bot: confidence scaling + risk_per_trade ceiling.
                    # vol_mult/regime_mult omitted here — live bot applies them per-candle via _calculate_margin.
                    SLIPPAGE = 0.0005  # 5 bps round-trip slippage per leg (realistic for market orders)
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
                    entry_fee = notional * (TAKER_FEE + SLIPPAGE)
                    position = {
                        'side': 'long' if sig_val == 1 else 'short',
                        'size': size, 'margin': margin, 'entry_fee': entry_fee,
                    }
                    entry_price = price
                    hwm = lwm = price
                    last_trade_candle = i

                elif position is not None:
                    if position['side'] == 'long':
                        hwm = max(hwm, price)
                    else:
                        lwm = min(lwm, price)

                    pnl_pct = (
                        (price - entry_price) / entry_price if position['side'] == 'long'
                        else (entry_price - price) / entry_price
                    )

                    should_exit = False
                    exit_reason = ''
                    if pnl_pct <= -self.stop_loss_pct:
                        should_exit, exit_reason = True, 'Stop Loss'
                    elif pnl_pct >= self.take_profit_pct:
                        should_exit, exit_reason = True, 'Take Profit'
                    else:
                        if pnl_pct > 0:
                            if position['side'] == 'long' and price <= hwm * (1 - self.trailing_stop_pct):
                                should_exit, exit_reason = True, 'Trailing Stop'
                            elif position['side'] == 'short' and price >= lwm * (1 + self.trailing_stop_pct):
                                should_exit, exit_reason = True, 'Trailing Stop'
                        if not should_exit:
                            exit_conf = self.min_confidence * 0.8
                            if ((sig_val == -1 and position['side'] == 'long') or
                                    (sig_val == 1 and position['side'] == 'short')):
                                if conf >= exit_conf:
                                    should_exit, exit_reason = True, 'Signal Reversal'

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

    # ── Main cycle ───────────────────────────────────────────────────────────

    def run_cycle(self):
        if not self.running:
            return None

        self._cycle_count += 1

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

            if self.simulation_mode:
                self.simulate_trade(signal, price, confidence, symbol=symbol, df=df_ind)
            else:
                self.execute_live_trade(signal, price, confidence, symbol=symbol, df=df_ind)

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

    LEVERAGES = [5, 10, 15, 20, 25, 50]          # removed 2x (too conservative), added 15x, 50x
    RISK_PER_TRADE = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
    STOP_LOSS = [0.005, 0.008, 0.010, 0.012, 0.015, 0.020, 0.025]
    TAKE_PROFIT = [0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100]
    COOLDOWNS = [60, 180, 300, 600, 900]
    CONFIDENCES = [x / 100 for x in range(60, 90, 5)]  # raised floor from 55% to 60%
    TIMEFRAMES = ['5m', '15m', '30m', '1h', '2h', '4h']
    TRAILING_STOPS = [0.005, 0.008, 0.010, 0.015, 0.020]
    PROFIT_MULTIPLIERS = [1.0, 1.25, 1.5, 2.0, 2.5]
    ADX_THRESHOLDS = [8, 10, 13, 16, 18, 20, 23, 25]  # per-token optimized entry trend filter

    MIN_TRADES = 20                  # raised from 15 — more robust signal requirement
    SAMPLES_PER_TIMEFRAME = 120      # raised from 100

    def __init__(self, user_id: int, selected_coins: list, starting_balance: float = 10000,
                 api_key: str = None, api_secret: str = None, api_password: str = None):
        self.user_id = user_id
        self.selected_coins = selected_coins
        self.starting_balance = starting_balance
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_password = api_password
        self.ohlcv_cache = {}
        self.model_cache = {}
        self.progress = 0
        self.total_tests = 0
        self.current_test = 0
        self.results = []
        self.phase = 'idle'

    def _random_params(self):
        import random
        for _ in range(50):
            params = {
                'leverage': random.choice(self.LEVERAGES),
                'risk_per_trade': random.choice(self.RISK_PER_TRADE),
                'stop_loss_pct': random.choice(self.STOP_LOSS),
                'take_profit_pct': random.choice(self.TAKE_PROFIT),
                'trade_cooldown': random.choice(self.COOLDOWNS),
                'min_confidence': random.choice(self.CONFIDENCES),
                'trailing_stop_pct': random.choice(self.TRAILING_STOPS),
                'profit_risk_multiplier': random.choice(self.PROFIT_MULTIPLIERS),
                'adx_threshold': random.choice(self.ADX_THRESHOLDS),
            }
            # Enforce minimum 1.5:1 reward:risk — required for sustainable compounding
            if params['take_profit_pct'] / params['stop_loss_pct'] >= 1.5:
                return params
        return params  # fallback (extremely rare after 50 attempts)

    def _cache_ohlcv(self, symbol: str, timeframe: str, days: int = 30):
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
                df = bot.create_labels(df)
                df = df.dropna()
            self.ohlcv_cache[key] = df
        return self.ohlcv_cache.get(key)

    def _train_and_cache_model(self, symbol: str, timeframe: str, days: int = 30):
        key = (symbol, timeframe)
        if key in self.model_cache:
            return

        df = self._cache_ohlcv(symbol, timeframe, days)
        if df is None or len(df) < 100:
            self.model_cache[key] = None
            return

        train_df = df.iloc[:len(df) // 2]
        test_df = df.iloc[len(df) // 2:]

        bot = TradingService(
            user_id=self.user_id, starting_balance=self.starting_balance,
            selected_coins=[symbol], timeframe=timeframe,
        )
        X_train, _ = bot.prepare_features(train_df)
        y_train = train_df['signal']
        mask = y_train != 0
        X_train, y_train = X_train[mask], y_train[mask]

        if len(X_train) < 20:
            self.model_cache[key] = None
            return

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

            is_tree = LGBM_AVAILABLE or XGB_AVAILABLE
            model = bot._build_model()
            if is_tree:
                model.fit(X_res, _encode_y(y_res))
            else:
                tscv = TimeSeriesSplit(n_splits=3)
                grid = GridSearchCV(model, SVM_PARAMS, cv=tscv, scoring='f1_weighted', n_jobs=1)
                grid.fit(X_res, y_res)
                model = grid.best_estimator_

            X_test, _ = bot.prepare_features(test_df)
            self.model_cache[key] = {
                'model': model, 'imputer': imp, 'scaler': sc,
                'test_df': test_df, 'X_test': X_test,
                'temp_bot': bot, 'is_tree': is_tree,
            }
            logger.info(f"Optimizer: Trained model for {symbol} {timeframe}")
        except Exception as e:
            logger.warning(f"Optimizer: Failed to train {symbol} {timeframe}: {e}")
            self.model_cache[key] = None

    def _run_cached_backtest(self, timeframe: str, params: dict):
        TAKER_FEE = 0.0006
        all_trades = []
        total_balance = 0
        overall_max_dd = 0
        balance_per_coin = self.starting_balance / len(self.selected_coins)

        for symbol in self.selected_coins:
            cached = self.model_cache.get((symbol, timeframe))
            if cached is None:
                total_balance += balance_per_coin
                continue

            model = cached['model']
            imp = cached['imputer']
            sc = cached['scaler']
            test_df = cached['test_df']
            X_test = cached['X_test']
            bot = cached['temp_bot']
            is_tree = cached.get('is_tree', False)

            balance = balance_per_coin
            peak_balance = balance_per_coin
            position = None
            entry_price = 0
            hwm = lwm = 0
            trades = []
            last_trade_candle = -999
            minutes = bot._get_timeframe_minutes()
            cooldown_candles = max(1, math.ceil(params['trade_cooldown'] / 60 / minutes))

            for i in range(len(test_df)):
                try:
                    row = test_df.iloc[i]
                    price = row['close']
                    X_row = X_test.iloc[[i]]
                    X_imp = imp.transform(X_row)
                    X_sc = sc.transform(X_imp)
                    raw = model.predict(X_sc)[0]
                    sig_val = int(_decode_y([raw])[0]) if is_tree else int(raw)
                    conf = float(max(model.predict_proba(X_sc)[0]))

                    if i - last_trade_candle < cooldown_candles:
                        continue

                    # Entry filter must match live bot exactly (ADX + volume gate)
                    adx_o = row.get('adx', 25); adx_o = 25.0 if pd.isna(adx_o) else float(adx_o)
                    vol_o = row.get('volume_ratio', 1.0); vol_o = 1.0 if pd.isna(vol_o) else float(vol_o)
                    SLIP = 0.0005  # 5 bps slippage per leg
                    if position is None and sig_val != 0 and conf >= params['min_confidence'] and adx_o >= params['adx_threshold'] and vol_o >= 0.65:
                        # Confidence-scaled margin (mirrors live _calculate_margin)
                        conf_rng_o = max(0.01, 1.0 - params['min_confidence'])
                        c_scale_o = max(0.5, min(1.5, 0.5 + (conf - params['min_confidence']) / conf_rng_o))
                        risk_ceil_o = min(max(params['risk_per_trade'] * 2.0, 0.10), 0.75)
                        margin = min(balance * params['risk_per_trade'] * c_scale_o, balance * risk_ceil_o)
                        if margin <= 0 or margin > balance * 0.95:
                            continue
                        notional = margin * params['leverage']
                        size = notional / price
                        entry_fee = notional * (TAKER_FEE + SLIP)
                        position = {
                            'side': 'long' if sig_val == 1 else 'short',
                            'size': size, 'margin': margin, 'entry_fee': entry_fee,
                        }
                        entry_price = price
                        hwm = lwm = price
                        last_trade_candle = i

                    elif position is not None:
                        if position['side'] == 'long':
                            hwm = max(hwm, price)
                        else:
                            lwm = min(lwm, price)

                        pnl_pct = (
                            (price - entry_price) / entry_price if position['side'] == 'long'
                            else (entry_price - price) / entry_price
                        )
                        ts = params.get('trailing_stop_pct', 0.01)
                        should_exit = False

                        if pnl_pct <= -params['stop_loss_pct']:
                            should_exit = True
                        elif pnl_pct >= params['take_profit_pct']:
                            should_exit = True
                        else:
                            if pnl_pct > 0:
                                if position['side'] == 'long' and price <= hwm * (1 - ts):
                                    should_exit = True
                                elif position['side'] == 'short' and price >= lwm * (1 + ts):
                                    should_exit = True
                            if not should_exit:
                                exit_conf = params['min_confidence'] * 0.8
                                if sig_val != 0 and conf >= exit_conf:
                                    if (sig_val == -1 and position['side'] == 'long') or \
                                       (sig_val == 1 and position['side'] == 'short'):
                                        should_exit = True

                        if should_exit:
                            price_change = (
                                (price - entry_price) if position['side'] == 'long'
                                else (entry_price - price)
                            )
                            pnl_amount = price_change * position['size']
                            exit_fee = position['size'] * price * (TAKER_FEE + SLIP)
                            net_pnl = pnl_amount - position['entry_fee'] - exit_fee
                            balance += net_pnl

                            # Track drawdown per coin
                            peak_balance = max(peak_balance, balance)
                            dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
                            overall_max_dd = max(overall_max_dd, dd)

                            trades.append({'pnl': net_pnl})
                            position = None
                            last_trade_candle = i
                except Exception:
                    continue

            all_trades.extend(trades)
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

        return {
            'total_return': total_return_pct,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': winning,
            'win_rate': win_rate,
            'final_balance': total_balance,
            'max_drawdown': overall_max_dd,
            'sharpe_ratio': sharpe_ratio,
        }

    def _calculate_score(self, result: dict) -> float:
        if result['total_trades'] < self.MIN_TRADES:
            return -999
        if result['total_return'] < 0:
            return -999

        roi_score     = min(result['total_return'] / 100, 1.0)
        winrate_score = result['win_rate'] / 100
        trade_score   = min(result['total_trades'] / 100, 1.0)

        # Calmar ratio: return / max_drawdown — key metric for sustainable compounding
        max_dd = max(result.get('max_drawdown', 0.1), 0.1)
        calmar_score = min(result['total_return'] / max_dd, 5.0) / 5.0

        # Sharpe ratio: consistency and quality of returns (capped at Sharpe 3.0 → 1.0)
        sharpe_score = max(0.0, min(result.get('sharpe_ratio', 0) / 3.0, 1.0))

        # Penalise drawdowns above 15%
        dd_penalty = max(0.0, (result.get('max_drawdown', 0) - 15) / 100)

        return (
            0.30 * roi_score
            + 0.20 * winrate_score
            + 0.10 * trade_score
            + 0.25 * calmar_score
            + 0.15 * sharpe_score
            - dd_penalty
        )

    def optimize(self, days: int = 30, progress_callback=None):
        import random
        import time as _time
        print(f"[OPT] User {self.user_id}: optimize() started, days={days}", flush=True)
        random.seed(42)

        self.results = []
        self.total_tests = len(self.TIMEFRAMES) * self.SAMPLES_PER_TIMEFRAME
        self.current_test = 0

        logger.info(f"Starting optimization: {self.total_tests} tests across {len(self.TIMEFRAMES)} timeframes")

        # Phase 1: fetch data once per (coin, timeframe)
        total_fetches = len(self.TIMEFRAMES) * len(self.selected_coins)
        fetch_count = 0
        self.ohlcv_cache = {}
        self.phase = 'fetching'

        for tf_idx, timeframe in enumerate(self.TIMEFRAMES):
            logger.info(f"Fetching data: {timeframe} ({tf_idx + 1}/{len(self.TIMEFRAMES)})")
            for symbol in self.selected_coins:
                fetch_count += 1
                self.progress = (fetch_count / total_fetches) * 20
                if progress_callback:
                    progress_callback(self.progress)
                try:
                    self._cache_ohlcv(symbol, timeframe, days)
                    _time.sleep(0.2)
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol} {timeframe}: {e}")

        logger.info(f"Phase 1 complete: {len(self.ohlcv_cache)} datasets cached")

        # Phase 2: train once per (coin, timeframe), then sweep params
        self.model_cache = {}
        self.phase = 'testing'

        for tf_idx, timeframe in enumerate(self.TIMEFRAMES):
            logger.info(f"Optimizing timeframe: {timeframe} ({tf_idx + 1}/{len(self.TIMEFRAMES)})")
            for symbol in self.selected_coins:
                self._train_and_cache_model(symbol, timeframe, days)

            for sample_idx in range(self.SAMPLES_PER_TIMEFRAME):
                self.current_test += 1
                self.progress = 20 + (self.current_test / self.total_tests) * 80
                if progress_callback:
                    progress_callback(self.progress)
                if sample_idx % 50 == 0:
                    _time.sleep(0.01)

                params = self._random_params()
                try:
                    result = self._run_cached_backtest(timeframe, params)
                    score = self._calculate_score(result)
                    if score > -999:
                        self.results.append({
                            'timeframe': timeframe,
                            'leverage': params['leverage'],
                            'risk_per_trade': params['risk_per_trade'],
                            'stop_loss_pct': params['stop_loss_pct'],
                            'take_profit_pct': params['take_profit_pct'],
                            'trailing_stop_pct': params['trailing_stop_pct'],
                            'profit_risk_multiplier': params['profit_risk_multiplier'],
                            'trade_cooldown': params['trade_cooldown'],
                            'min_confidence': params['min_confidence'],
                            'total_return': round(result['total_return'], 2),
                            'win_rate': round(result['win_rate'], 2),
                            'total_trades': result['total_trades'],
                            'total_pnl': round(result['total_pnl'], 2),
                            'max_drawdown': round(result.get('max_drawdown', 0), 2),
                            'score': round(score, 4),
                        })
                except Exception as e:
                    logger.warning(f"Backtest failed: {e}")
                    continue

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
