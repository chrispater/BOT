import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

import math
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

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

SVM_PARAMS = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']}


class TradingService:
    def __init__(self, user_id: int, api_key=None, api_secret=None, api_password=None,
                 starting_balance=10000, leverage=10, selected_coins=None,
                 risk_per_trade=0.02, stop_loss_pct=0.015, take_profit_pct=0.03,
                 trade_cooldown=300, min_confidence=0.65, timeframe='5m',
                 trailing_stop_pct=0.01, max_drawdown_pct=0.20,
                 retrain_every=50, profit_risk_multiplier=1.5,
                 on_trade=None, on_signal=None, on_performance=None):
        self.user_id = user_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_password = api_password
        self.exchange = None
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.last_trade_times = {}      # symbol → timestamp (per-symbol cooldown)
        self.positions = {}             # symbol → position dict (multi-position support)
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
        self._cycle_count = 0           # Tracks cycles for periodic retraining

        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trade_cooldown = trade_cooldown
        self.min_confidence = min_confidence
        self.timeframe = timeframe if timeframe in VALID_TIMEFRAMES else '5m'
        self.trailing_stop_pct = trailing_stop_pct          # Distance below high-water-mark to trail
        self.max_drawdown_pct = max_drawdown_pct            # Circuit breaker: stop entries beyond this
        self.retrain_every = retrain_every                  # Retrain model every N cycles
        self.profit_risk_multiplier = profit_risk_multiplier  # Extra risk% applied to profits above base

        self.on_trade = on_trade
        self.on_signal = on_signal
        self.on_performance = on_performance

        self._initialize_exchange()

    def _initialize_exchange(self):
        try:
            self.exchange = ccxt.blofin({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_password,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap', 'adjustForTimeDifference': True}
            })

            if self.api_key and self.api_secret and self.api_password:
                self.exchange.load_markets()
                self.simulation_mode = False
                logger.info(f"User {self.user_id}: Exchange connected - LIVE MODE")
            else:
                logger.warning(f"User {self.user_id}: No API keys - SIMULATION MODE")
        except Exception as e:
            logger.error(f"User {self.user_id}: Exchange init failed: {e}")
            self.simulation_mode = True

    def get_current_symbol(self):
        if self.selected_coins:
            symbol = self.selected_coins[self.current_symbol_index % len(self.selected_coins)]
            return symbol
        return 'BTC/USDT:USDT'

    def fetch_ohlcv(self, symbol=None, limit=LOOKBACK_PERIODS):
        symbol = symbol or self.get_current_symbol()
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except:
            return self._generate_simulated_data(limit)

    def _get_timeframe_minutes(self):
        tf_map = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '1d': 1440
        }
        return tf_map.get(self.timeframe, 5)

    def _get_pandas_freq(self):
        freq_map = {
            '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1h', '2h': '2h', '4h': '4h', '1d': '1D'
        }
        return freq_map.get(self.timeframe, '5min')

    def _generate_simulated_data(self, limit):
        np.random.seed(int(time.time()) % 1000)
        base_price = 95000
        freq = self._get_pandas_freq()
        timestamps = pd.date_range(end=datetime.now(), periods=limit, freq=freq)
        prices = [base_price]
        for i in range(1, limit):
            change = np.random.randn() * 50 + np.sin(i/20) * 30
            prices.append(max(prices[-1] + change, base_price * 0.9))
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.randn()) * 0.002) for p in prices],
            'low': [p * (1 - abs(np.random.randn()) * 0.002) for p in prices],
            'close': [p + np.random.randn() * 20 for p in prices],
            'volume': [np.random.uniform(100, 1000) for _ in prices]
        }, index=timestamps)
        return df

    def calculate_indicators(self, df):
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        open_price = df['open'].values.astype(float)
        volume = df['volume'].values.astype(float)

        # ============ TRADITIONAL MOMENTUM INDICATORS ============
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

        # ============ VWAP & VOLUME-WEIGHTED ANALYSIS ============
        # FIX: rolling 20-period VWAP instead of cumsum (which drifts across the dataset)
        typical_price = (high + low + close) / 3
        tp_series = pd.Series(typical_price, index=df.index)
        vol_series = pd.Series(volume, index=df.index)
        df['vwap'] = (tp_series * vol_series).rolling(window=20).sum() / vol_series.rolling(window=20).sum()

        df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
        df['vwap_slope'] = df['vwap'].pct_change(periods=5) * 100

        # ============ ON-BALANCE VOLUME (OBV) ============
        df['obv'] = talib.OBV(close, volume)
        df['obv_sma'] = df['obv'].rolling(window=20).mean()
        df['obv_slope'] = df['obv'].pct_change(periods=5) * 100

        # ============ ACCUMULATION/DISTRIBUTION ============
        df['ad'] = talib.AD(high, low, close, volume)
        df['ad_slope'] = df['ad'].pct_change(periods=5) * 100

        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        mfv = pd.Series(mfm * volume, index=df.index)
        df['cmf'] = mfv.rolling(window=20).sum() / vol_series.rolling(window=20).sum()

        # ============ VOLUME-PRICE DIVERGENCE DETECTION ============
        price_change_5 = df['close'].pct_change(periods=5)
        volume_change_5 = df['volume'].pct_change(periods=5)
        df['volume_price_confirm'] = np.sign(price_change_5) * np.sign(volume_change_5)
        expected_vol_change = abs(price_change_5) * df['volume_sma']
        actual_vol_change = abs(df['volume'] - df['volume_sma'])
        df['volume_divergence'] = (actual_vol_change - expected_vol_change) / (df['volume_sma'] + 1e-10)

        # ============ BREAKOUT/BREAKDOWN DETECTION ============
        df['donchian_high'] = df['high'].rolling(window=20).max()
        df['donchian_low'] = df['low'].rolling(window=20).min()
        df['donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2
        donchian_range = df['donchian_high'] - df['donchian_low']
        df['breakout_proximity'] = (df['close'] - df['donchian_mid']) / (donchian_range / 2 + 1e-10)
        df['is_new_high'] = (df['close'] >= df['donchian_high'].shift(1)).astype(int)
        df['is_new_low'] = (df['close'] <= df['donchian_low'].shift(1)).astype(int)
        df['breakout_quality'] = (
            df['is_new_high'] * (1 + df['volume_ratio'].clip(0, 2) - 1) -
            df['is_new_low'] * (1 + df['volume_ratio'].clip(0, 2) - 1)
        )

        # ============ PRICE MOMENTUM WITH VOLUME WEIGHT ============
        df['vol_weighted_mom'] = (df['returns'] * df['volume_ratio']).rolling(window=10).sum()
        df['vol_weighted_roc'] = df['vol_weighted_mom'].pct_change(periods=5) * 100

        # ============ SQUEEZE DETECTION (Pre-breakout compression) ============
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_squeeze'] = df['bb_width'] / df['bb_width'].rolling(window=50).mean()
        df['keltner_upper'] = df['ema_12'] + 2 * df['atr']
        df['keltner_lower'] = df['ema_12'] - 2 * df['atr']
        df['in_squeeze'] = ((df['bb_lower'] > df['keltner_lower']) &
                           (df['bb_upper'] < df['keltner_upper'])).astype(int)

        # ============ TREND STRENGTH WITH VOLUME ============
        df['vol_adj_adx'] = df['adx'] * df['volume_ratio'].clip(0.5, 2)
        up_volume = pd.Series(np.where(close > open_price, volume, 0), index=df.index)
        down_volume = pd.Series(np.where(close < open_price, volume, 0), index=df.index)
        df['directional_volume'] = (up_volume.rolling(window=10).sum() -
                                   down_volume.rolling(window=10).sum()) / \
                                   (df['volume'].rolling(window=10).sum() + 1e-10)

        return df

    def create_labels(self, df, forward_periods=5, threshold=0.005):
        df['future_return'] = df['close'].shift(-forward_periods) / df['close'] - 1
        df['signal'] = 0
        df.loc[df['future_return'] > threshold, 'signal'] = 1
        df.loc[df['future_return'] < -threshold, 'signal'] = -1
        return df

    def prepare_features(self, df):
        feature_columns = [
            # Traditional momentum indicators
            'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position',
            'atr', 'adx', 'stoch_k', 'stoch_d', 'cci', 'mfi',
            'roc', 'mom', 'trend_sma', 'volatility', 'volume_ratio',
            # VWAP & Volume-Weighted Analysis
            'vwap_distance', 'vwap_slope',
            # On-Balance Volume
            'obv_slope',
            # Accumulation/Distribution
            'ad_slope', 'cmf',
            # Volume-Price Divergence
            'volume_price_confirm', 'volume_divergence',
            # Breakout/Breakdown Detection
            'breakout_proximity', 'breakout_quality',
            # Volume-Weighted Momentum
            'vol_weighted_mom', 'vol_weighted_roc',
            # Squeeze Detection
            'bb_squeeze', 'in_squeeze',
            # Trend Strength with Volume
            'vol_adj_adx', 'directional_volume',
        ]
        available = [col for col in feature_columns if col in df.columns]
        return df[available].copy(), available

    def train_model(self, df):
        logger.info(f"User {self.user_id}: Training ML model...")
        df = self.calculate_indicators(df.copy())
        df = self.create_labels(df)
        df = df.dropna()

        if len(df) < 50:
            return False

        X, _ = self.prepare_features(df)
        y = df['signal']
        mask = y != 0
        X, y = X[mask], y[mask]

        if len(X) < 30:
            return False

        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        try:
            smote = SMOTE(random_state=42, k_neighbors=min(3, len(y[y==1])-1, len(y[y==-1])-1))
            X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        except:
            X_resampled, y_resampled = X_scaled, y

        tscv = TimeSeriesSplit(n_splits=3)
        svm = SVC(probability=True, random_state=42)
        grid = GridSearchCV(svm, SVM_PARAMS, cv=tscv, scoring='f1_weighted', n_jobs=1)
        grid.fit(X_resampled, y_resampled)
        self.model = grid.best_estimator_

        accuracy = accuracy_score(y, self.model.predict(X_scaled))
        logger.info(f"User {self.user_id}: Model trained - Accuracy: {accuracy:.2%}")
        return True

    def predict_signal(self, df):
        if self.model is None:
            return 0, 0.5
        df = self.calculate_indicators(df.copy())
        X, _ = self.prepare_features(df)
        if X.empty:
            return 0, 0.5
        X_latest = X.iloc[[-1]]
        X_imputed = self.imputer.transform(X_latest)
        X_scaled = self.scaler.transform(X_imputed)
        pred = self.model.predict(X_scaled)[0]
        conf = max(self.model.predict_proba(X_scaled)[0])
        return pred, conf

    # ============ NEW: COMPOUNDING & RISK HELPERS ============

    def _is_drawdown_exceeded(self):
        """Circuit breaker: block new entries once drawdown exceeds max_drawdown_pct."""
        drawdown = (self.starting_balance - self.balance) / self.starting_balance
        if drawdown > self.max_drawdown_pct:
            logger.warning(
                f"User {self.user_id}: Drawdown {drawdown:.1%} > max {self.max_drawdown_pct:.1%} - halting new entries"
            )
            return True
        return False

    def _calculate_margin(self):
        """
        Profit-tier compounding:
          - Base capital (up to starting_balance) risks at risk_per_trade.
          - Profit above starting_balance risks at risk_per_trade × profit_risk_multiplier.
        Hard cap: never commit more than 10% of current balance as margin on one trade.
        """
        profit = max(0.0, self.balance - self.starting_balance)
        base_cap = min(self.balance, self.starting_balance)
        margin = (base_cap * self.risk_per_trade) + (profit * self.risk_per_trade * self.profit_risk_multiplier)
        return min(margin, self.balance * 0.10)

    def _get_volatility_multiplier(self, df):
        """
        ATR-based sizing: scale position size inversely to current volatility.
        High volatility → smaller size (0.5×); low volatility → larger size (1.5×).
        Baseline: 0.15% ATR-to-price, typical for 5m BTC.
        """
        if 'atr' not in df.columns or df['atr'].isna().all():
            return 1.0
        atr = df['atr'].iloc[-1]
        price = df['close'].iloc[-1]
        if price <= 0 or atr <= 0:
            return 1.0
        atr_pct = atr / price
        baseline_atr_pct = 0.0015
        multiplier = baseline_atr_pct / (atr_pct + 1e-10)
        return max(0.5, min(1.5, multiplier))

    def _sync_live_balance(self):
        """Pull actual USDT balance from Blofin and update self.balance."""
        if self.simulation_mode or self.exchange is None:
            return
        try:
            account = self.exchange.fetch_balance()
            usdt = account.get('USDT', {})
            total = float(usdt.get('total', 0))
            if total > 0:
                old = self.balance
                self.balance = total
                logger.info(f"User {self.user_id}: Balance synced ${old:.2f} → ${total:.2f}")
        except Exception as e:
            logger.warning(f"User {self.user_id}: Balance sync failed: {e}")

    # ============ STATUS ============

    def get_status(self):
        coin_signals = {}
        for signal in self.signals_history:
            symbol = signal.get('symbol', '')
            coin = symbol.split('/')[0] if symbol else 'Unknown'
            coin_signals[coin] = signal

        open_positions = list(self.positions.values())

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
            'total_pnl': float(self.total_pnl),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'positions': open_positions,            # Multi-position list
            'position': open_positions[0] if open_positions else None,  # Backwards compat
            'last_signals': self.signals_history[-10:] if self.signals_history else [],
            'coin_signals': coin_signals,
            'recent_trades': self.trades_history[-10:] if self.trades_history else []
        }

    # ============ TRADE EXECUTION ============

    def simulate_trade(self, signal, price, confidence, symbol=None, df=None):
        symbol = symbol or self.get_current_symbol()
        current_time = time.time()
        last_trade = self.last_trade_times.get(symbol, 0)
        if current_time - last_trade < self.trade_cooldown:
            return

        position = self.positions.get(symbol)

        if position is None and signal != 0 and confidence >= self.min_confidence:
            if self._is_drawdown_exceeded():
                return

            vol_mult = self._get_volatility_multiplier(df) if df is not None else 1.0
            margin = self._calculate_margin() * vol_mult
            if margin <= 0:
                return
            notional = margin * self.leverage
            size = notional / price

            side = 'long' if signal == 1 else 'short'
            self.positions[symbol] = {
                'side': side, 'size': size, 'entry_price': price,
                'symbol': symbol, 'margin': margin,
                'high_water_mark': price,   # Trailing stop reference for longs
                'low_water_mark': price,    # Trailing stop reference for shorts
            }
            self.entry_price = price
            self.last_trade_times[symbol] = current_time

            trade = {
                'type': 'open', 'side': side, 'size': float(size), 'price': float(price),
                'confidence': float(confidence), 'symbol': symbol, 'margin': float(margin),
                'time': datetime.now().isoformat()
            }
            self.trades_history.append(trade)

            if self.on_trade:
                self.on_trade(self.user_id, symbol, side, 'open', size, price, None, confidence, None)

            logger.info(
                f"User {self.user_id}: [SIM] Opened {side.upper()} {symbol} @ ${price:.2f} | "
                f"Margin: ${margin:.2f} | Size: {size:.4f} | VolMult: {vol_mult:.2f}"
            )

        elif position is not None:
            entry = position['entry_price']

            # Update trailing water marks each cycle
            if position['side'] == 'long':
                position['high_water_mark'] = max(position['high_water_mark'], price)
            else:
                position['low_water_mark'] = min(position['low_water_mark'], price)

            pnl_pct = (
                (price - entry) / entry if position['side'] == 'long'
                else (entry - price) / entry
            )

            should_exit = False
            exit_reason = ""

            if pnl_pct <= -self.stop_loss_pct:
                should_exit, exit_reason = True, "Stop Loss"
            elif pnl_pct >= self.take_profit_pct:
                should_exit, exit_reason = True, "Take Profit"
            else:
                # Trailing stop: only activates once position is in profit
                if position['side'] == 'long':
                    trail_price = position['high_water_mark'] * (1 - self.trailing_stop_pct)
                    if price <= trail_price and pnl_pct > 0:
                        should_exit, exit_reason = True, "Trailing Stop"
                else:
                    trail_price = position['low_water_mark'] * (1 + self.trailing_stop_pct)
                    if price >= trail_price and pnl_pct > 0:
                        should_exit, exit_reason = True, "Trailing Stop"

                # FIX: exit confidence bar is lower than entry (80% of min_confidence)
                # Prevents being locked in a trade just because model uncertainty grew
                if not should_exit:
                    exit_conf_threshold = self.min_confidence * 0.8
                    if ((signal == -1 and position['side'] == 'long') or
                            (signal == 1 and position['side'] == 'short')):
                        if confidence >= exit_conf_threshold:
                            should_exit, exit_reason = True, "Signal Reversal"

            if should_exit:
                price_change = (
                    (price - entry) if position['side'] == 'long' else (entry - price)
                )
                pnl_amount = price_change * position['size']

                self.total_pnl += pnl_amount
                self.total_trades += 1
                if pnl_amount > 0:
                    self.winning_trades += 1
                self.balance += pnl_amount

                margin_used = position.get('margin', self.balance * self.risk_per_trade)
                leveraged_return_pct = (pnl_amount / margin_used) * 100 if margin_used > 0 else 0

                trade = {
                    'type': 'close', 'side': position['side'], 'price': float(price),
                    'pnl': float(pnl_amount), 'pnl_pct': round(leveraged_return_pct, 2),
                    'reason': exit_reason, 'symbol': symbol, 'time': datetime.now().isoformat()
                }
                self.trades_history.append(trade)

                if self.on_trade:
                    self.on_trade(self.user_id, symbol, position['side'], 'close',
                                 position['size'], price, pnl_amount, None, exit_reason)

                if self.on_performance:
                    self.on_performance(self.user_id, self.balance, self.total_pnl,
                                       self.total_trades, self.winning_trades)

                logger.info(
                    f"User {self.user_id}: [SIM] Closed {symbol} - {exit_reason} - "
                    f"PnL: ${pnl_amount:.2f} ({leveraged_return_pct:.1f}%) | Balance: ${self.balance:.2f}"
                )
                del self.positions[symbol]
                self.last_trade_times[symbol] = current_time

    def execute_live_trade(self, signal, price, confidence, symbol=None, df=None):
        symbol = symbol or self.get_current_symbol()
        current_time = time.time()
        last_trade = self.last_trade_times.get(symbol, 0)
        if current_time - last_trade < self.trade_cooldown:
            return

        try:
            position = self.positions.get(symbol)

            if position is None and signal != 0 and confidence >= self.min_confidence:
                if self._is_drawdown_exceeded():
                    return

                try:
                    account_balance = self.exchange.fetch_balance()
                    available_margin = float(account_balance.get('USDT', {}).get('free', 0))
                    if available_margin <= 0 and 'free' in account_balance:
                        available_margin = float(account_balance['free'].get('USDT', 0))
                    logger.info(f"User {self.user_id}: Available margin: ${available_margin:.2f}")
                except Exception as e:
                    logger.warning(f"User {self.user_id}: Could not fetch balance: {e}, using tracked balance")
                    available_margin = self.balance

                vol_mult = self._get_volatility_multiplier(df) if df is not None else 1.0
                margin = self._calculate_margin() * vol_mult

                if margin > available_margin * 0.95:
                    margin = available_margin * 0.95
                    logger.warning(f"User {self.user_id}: Reduced margin to fit available: ${margin:.2f}")

                notional = margin * self.leverage
                size = notional / price

                side = 'buy' if signal == 1 else 'sell'
                position_side = 'long' if signal == 1 else 'short'

                market = self.exchange.market(symbol)
                contract_size = market.get('contractSize', 1) or 1
                amount = size / contract_size

                try:
                    self.exchange.set_leverage(self.leverage, symbol)
                except Exception as e:
                    logger.warning(f"User {self.user_id}: Could not set leverage: {e}")

                order = self.exchange.create_order(
                    symbol=symbol, type='market', side=side, amount=amount,
                    params={'posSide': position_side}
                )

                self.positions[symbol] = {
                    'side': position_side, 'size': size, 'entry_price': price,
                    'symbol': symbol, 'margin': margin, 'order_id': order.get('id'),
                    'high_water_mark': price, 'low_water_mark': price,
                }
                self.entry_price = price
                self.last_trade_times[symbol] = current_time

                trade = {
                    'type': 'open', 'side': position_side, 'size': float(size), 'price': float(price),
                    'confidence': float(confidence), 'symbol': symbol, 'margin': float(margin),
                    'time': datetime.now().isoformat()
                }
                self.trades_history.append(trade)

                if self.on_trade:
                    self.on_trade(self.user_id, symbol, position_side, 'open', size, price, None, confidence, None)

                logger.info(
                    f"User {self.user_id}: [LIVE] Opened {position_side.upper()} {symbol} @ ${price:.2f} | "
                    f"Margin: ${margin:.2f} | VolMult: {vol_mult:.2f}"
                )

            elif position is not None:
                entry = position['entry_price']

                if position['side'] == 'long':
                    position['high_water_mark'] = max(position['high_water_mark'], price)
                else:
                    position['low_water_mark'] = min(position['low_water_mark'], price)

                pnl_pct = (
                    (price - entry) / entry if position['side'] == 'long'
                    else (entry - price) / entry
                )

                should_exit = False
                exit_reason = ""

                if pnl_pct <= -self.stop_loss_pct:
                    should_exit, exit_reason = True, "Stop Loss"
                elif pnl_pct >= self.take_profit_pct:
                    should_exit, exit_reason = True, "Take Profit"
                else:
                    if position['side'] == 'long':
                        trail_price = position['high_water_mark'] * (1 - self.trailing_stop_pct)
                        if price <= trail_price and pnl_pct > 0:
                            should_exit, exit_reason = True, "Trailing Stop"
                    else:
                        trail_price = position['low_water_mark'] * (1 + self.trailing_stop_pct)
                        if price >= trail_price and pnl_pct > 0:
                            should_exit, exit_reason = True, "Trailing Stop"

                    if not should_exit:
                        exit_conf_threshold = self.min_confidence * 0.8
                        if ((signal == -1 and position['side'] == 'long') or
                                (signal == 1 and position['side'] == 'short')):
                            if confidence >= exit_conf_threshold:
                                should_exit, exit_reason = True, "Signal Reversal"

                if should_exit:
                    pos_symbol = position.get('symbol', symbol)
                    close_side = 'sell' if position['side'] == 'long' else 'buy'

                    market = self.exchange.market(pos_symbol)
                    contract_size = market.get('contractSize', 1) or 1
                    amount = position['size'] / contract_size

                    order = self.exchange.create_order(
                        symbol=pos_symbol, type='market', side=close_side, amount=amount,
                        params={'posSide': position['side'], 'reduceOnly': True}
                    )

                    price_change = (
                        (price - entry) if position['side'] == 'long' else (entry - price)
                    )
                    pnl_amount = price_change * position['size']

                    self.total_pnl += pnl_amount
                    self.total_trades += 1
                    if pnl_amount > 0:
                        self.winning_trades += 1
                    self.balance += pnl_amount

                    margin_used = position.get('margin', self.balance * self.risk_per_trade)
                    leveraged_return_pct = (pnl_amount / margin_used) * 100 if margin_used > 0 else 0

                    trade = {
                        'type': 'close', 'side': position['side'], 'price': float(price),
                        'pnl': float(pnl_amount), 'pnl_pct': round(leveraged_return_pct, 2),
                        'reason': exit_reason, 'symbol': pos_symbol, 'time': datetime.now().isoformat()
                    }
                    self.trades_history.append(trade)

                    if self.on_trade:
                        self.on_trade(self.user_id, pos_symbol, position['side'], 'close',
                                     position['size'], price, pnl_amount, None, exit_reason)

                    if self.on_performance:
                        self.on_performance(self.user_id, self.balance, self.total_pnl,
                                           self.total_trades, self.winning_trades)

                    logger.info(
                        f"User {self.user_id}: [LIVE] Closed {pos_symbol} - {exit_reason} - "
                        f"PnL: ${pnl_amount:.2f} ({leveraged_return_pct:.1f}%)"
                    )
                    del self.positions[symbol]
                    self.last_trade_times[symbol] = current_time

                    # Sync actual exchange balance after every close
                    self._sync_live_balance()

        except Exception as e:
            logger.error(f"User {self.user_id}: Live trade execution failed: {e}")

    # ============ BACKTEST ============

    def run_backtest_single_coin(self, symbol, days=30):
        """
        Backtest aligned with live trading logic.
        Includes: trailing stop, profit-tier compounding, ATR sizing, lower exit confidence.
        """
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
            imputer = SimpleImputer(strategy='mean')
            scaler = StandardScaler()
            X_imputed = imputer.fit_transform(X_train)
            X_scaled = scaler.fit_transform(X_imputed)

            try:
                smote = SMOTE(random_state=42, k_neighbors=min(3, len(y_train[y_train==1])-1, len(y_train[y_train==-1])-1))
                X_resampled, y_resampled = smote.fit_resample(X_scaled, y_train)
            except:
                X_resampled, y_resampled = X_scaled, y_train

            tscv = TimeSeriesSplit(n_splits=3)
            svm = SVC(probability=True, random_state=42)
            grid = GridSearchCV(svm, SVM_PARAMS, cv=tscv, scoring='f1_weighted', n_jobs=1)
            grid.fit(X_resampled, y_resampled)
            model = grid.best_estimator_
        except Exception as e:
            return {'symbol': symbol, 'error': f'Model training failed: {str(e)}', 'total_return': 0, 'win_rate': 0, 'total_trades': 0}

        balance_per_coin = self.starting_balance / len(self.selected_coins)
        balance = balance_per_coin
        position = None
        entry_price = 0
        high_water_mark = 0
        low_water_mark = 0
        trades = []
        last_trade_candle = -999

        cooldown_candles = max(1, math.ceil(self.trade_cooldown / 60 / minutes_per_candle))
        X_test, _ = self.prepare_features(test_df)

        for i in range(len(test_df)):
            try:
                row = test_df.iloc[i]
                price = row['close']
                X_row = X_test.iloc[[i]]
                X_imp = imputer.transform(X_row)
                X_sc = scaler.transform(X_imp)
                signal = model.predict(X_sc)[0]
                conf = max(model.predict_proba(X_sc)[0])

                if i - last_trade_candle < cooldown_candles:
                    continue

                if position is None and signal != 0 and conf >= self.min_confidence:
                    # Profit-tier compounding for backtest
                    profit = max(0, balance - balance_per_coin)
                    margin = (balance_per_coin * self.risk_per_trade +
                              profit * self.risk_per_trade * self.profit_risk_multiplier)
                    margin = min(margin, balance * 0.10)
                    if margin <= 0:
                        continue

                    notional = margin * self.leverage
                    size = notional / price
                    entry_fee = notional * TAKER_FEE

                    position = {
                        'side': 'long' if signal == 1 else 'short',
                        'size': size, 'margin': margin, 'entry_fee': entry_fee
                    }
                    entry_price = price
                    high_water_mark = price
                    low_water_mark = price
                    last_trade_candle = i

                elif position is not None:
                    if position['side'] == 'long':
                        high_water_mark = max(high_water_mark, price)
                    else:
                        low_water_mark = min(low_water_mark, price)

                    pnl_pct = (
                        (price - entry_price) / entry_price if position['side'] == 'long'
                        else (entry_price - price) / entry_price
                    )

                    should_exit = False
                    exit_reason = ""

                    if pnl_pct <= -self.stop_loss_pct:
                        should_exit, exit_reason = True, "Stop Loss"
                    elif pnl_pct >= self.take_profit_pct:
                        should_exit, exit_reason = True, "Take Profit"
                    else:
                        if position['side'] == 'long':
                            trail_price = high_water_mark * (1 - self.trailing_stop_pct)
                            if price <= trail_price and pnl_pct > 0:
                                should_exit, exit_reason = True, "Trailing Stop"
                        else:
                            trail_price = low_water_mark * (1 + self.trailing_stop_pct)
                            if price >= trail_price and pnl_pct > 0:
                                should_exit, exit_reason = True, "Trailing Stop"

                        if not should_exit:
                            exit_conf = self.min_confidence * 0.8
                            if ((signal == -1 and position['side'] == 'long') or
                                    (signal == 1 and position['side'] == 'short')):
                                if conf >= exit_conf:
                                    should_exit, exit_reason = True, "Signal Reversal"

                    if should_exit:
                        price_change = (
                            (price - entry_price) if position['side'] == 'long'
                            else (entry_price - price)
                        )
                        pnl_amount = price_change * position['size']
                        exit_notional = position['size'] * price
                        exit_fee = exit_notional * TAKER_FEE
                        total_fees = position.get('entry_fee', 0) + exit_fee
                        net_pnl = pnl_amount - total_fees
                        balance += net_pnl

                        margin_used = position.get('margin', balance * self.risk_per_trade)
                        leveraged_pnl_pct = (net_pnl / margin_used) * 100 if margin_used > 0 else 0

                        trades.append({
                            'symbol': symbol,
                            'coin': symbol.split('/')[0],
                            'side': position['side'],
                            'entry': round(entry_price, 2),
                            'exit': round(price, 2),
                            'pnl': round(net_pnl, 2),
                            'pnl_pct': round(leveraged_pnl_pct, 2),
                            'fees': round(total_fees, 4),
                            'reason': exit_reason
                        })
                        position = None
                        last_trade_candle = i
            except:
                continue

        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        total_pnl = sum(t['pnl'] for t in trades)
        total_fees = sum(t.get('fees', 0) for t in trades)
        total_return_pct = ((balance - balance_per_coin) / balance_per_coin) * 100

        max_drawdown = 0
        peak = balance_per_coin
        running_balance = balance_per_coin
        for t in trades:
            running_balance += t['pnl']
            peak = max(peak, running_balance)
            drawdown = (peak - running_balance) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

        return {
            'symbol': symbol,
            'coin': symbol.split('/')[0],
            'starting_balance': round(balance_per_coin, 2),
            'final_balance': round(balance, 2),
            'total_return': round(total_return_pct, 2),
            'total_pnl': round(total_pnl, 2),
            'total_fees': round(total_fees, 4),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': round((winning_trades / total_trades * 100) if total_trades > 0 else 0, 2),
            'max_drawdown': round(max_drawdown, 2),
            'trades': trades
        }

    def run_backtest(self, days=30):
        coin_results = []
        all_trades = []

        for symbol in self.selected_coins:
            result = self.run_backtest_single_coin(symbol, days)
            coin_results.append(result)
            if 'trades' in result:
                all_trades.extend(result.get('trades', []))

        total_starting = self.starting_balance
        total_final = sum(r.get('final_balance', r.get('starting_balance', 0)) for r in coin_results)
        total_pnl = total_final - total_starting
        total_trades_count = sum(r.get('total_trades', 0) for r in coin_results)
        winning_trades = sum(r.get('winning_trades', 0) for r in coin_results)
        max_drawdown = max((r.get('max_drawdown', 0) for r in coin_results), default=0)
        total_fees = sum(r.get('total_fees', 0) for r in coin_results)

        total_return_pct = ((total_final - total_starting) / total_starting) * 100 if total_starting > 0 else 0

        return {
            'period_days': days,
            'starting_balance': total_starting,
            'final_balance': round(total_final, 2),
            'total_return': round(total_return_pct, 2),
            'total_pnl': round(total_pnl, 2),
            'total_fees': round(total_fees, 4),
            'total_trades': total_trades_count,
            'winning_trades': winning_trades,
            'win_rate': round((winning_trades / total_trades_count * 100) if total_trades_count > 0 else 0, 2),
            'max_drawdown': round(max_drawdown, 2),
            'leverage': self.leverage,
            'risk_per_trade': self.risk_per_trade,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'trailing_stop_pct': self.trailing_stop_pct,
            'min_confidence': self.min_confidence,
            'timeframe': self.timeframe,
            'selected_coins': self.selected_coins,
            'coin_results': coin_results,
            'all_trades': all_trades
        }

    # ============ MAIN CYCLE ============

    def run_cycle(self):
        if not self.running:
            return None

        self._cycle_count += 1
        results = []

        for symbol in self.selected_coins:
            df = self.fetch_ohlcv(symbol=symbol)
            if df is None or len(df) < 50:
                continue

            # Periodic retraining: keeps model adapted to current market regime
            should_train = self.model is None or (self._cycle_count % self.retrain_every == 0)
            if should_train:
                logger.info(
                    f"User {self.user_id}: {'Initial' if self.model is None else 'Periodic'} "
                    f"model training (cycle {self._cycle_count})"
                )
                self.train_model(df)

            df = self.calculate_indicators(df)
            signal, confidence = self.predict_signal(df)
            latest = df.iloc[-1]
            price = latest['close']

            signal_data = {
                'signal': int(signal),
                'confidence': float(confidence),
                'price': float(price),
                'symbol': symbol,
                'rsi': float(latest.get('rsi', 0)) if pd.notna(latest.get('rsi')) else 0,
                'macd': float(latest.get('macd', 0)) if pd.notna(latest.get('macd')) else 0,
                'adx': float(latest.get('adx', 0)) if pd.notna(latest.get('adx')) else 0,
                'time': datetime.now().isoformat()
            }
            self.signals_history.append(signal_data)
            if len(self.signals_history) > 100:
                self.signals_history = self.signals_history[-100:]

            if self.on_signal:
                self.on_signal(self.user_id, signal, confidence, price,
                              signal_data['rsi'], signal_data['macd'], signal_data['adx'])

            if self.simulation_mode:
                self.simulate_trade(signal, price, confidence, symbol=symbol, df=df)
            else:
                self.execute_live_trade(signal, price, confidence, symbol=symbol, df=df)

            results.append(signal_data)

        self.current_symbol_index += 1
        return results[-1] if results else None


class ParameterOptimizer:
    """
    Smart parameter optimization using stratified random search.
    Finds optimal trading parameters across timeframes.
    Now also optimises trailing_stop_pct.
    """

    LEVERAGES = [2, 5, 10, 20, 25]
    RISK_PER_TRADE = [0.01, 0.02, 0.03, 0.04, 0.05]
    STOP_LOSS = [x/1000 for x in range(5, 35, 5)]
    TAKE_PROFIT = [x/1000 for x in range(10, 110, 10)]
    COOLDOWNS = [60, 300, 600, 900]
    CONFIDENCES = [x/100 for x in range(55, 95, 5)]
    TIMEFRAMES = ['5m', '15m', '30m', '1h', '2h', '4h']
    TRAILING_STOPS = [0.005, 0.01, 0.015, 0.02]    # Also optimise trailing stop

    MIN_TRADES = 15
    SAMPLES_PER_TIMEFRAME = 100

    def __init__(self, user_id: int, selected_coins: list, starting_balance: float = 10000,
                 api_key: str = None, api_secret: str = None, api_password: str = None):
        self.user_id = user_id
        self.selected_coins = selected_coins
        self.starting_balance = starting_balance
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_password = api_password
        self.ohlcv_cache = {}
        self.model_cache = {}           # FIX: was never initialized, caused AttributeError
        self.progress = 0
        self.total_tests = 0
        self.current_test = 0
        self.results = []
        self.phase = "idle"

    def _random_params(self):
        import random
        return {
            'leverage': random.choice(self.LEVERAGES),
            'risk_per_trade': random.choice(self.RISK_PER_TRADE),
            'stop_loss_pct': random.choice(self.STOP_LOSS),
            'take_profit_pct': random.choice(self.TAKE_PROFIT),
            'trade_cooldown': random.choice(self.COOLDOWNS),
            'min_confidence': random.choice(self.CONFIDENCES),
            'trailing_stop_pct': random.choice(self.TRAILING_STOPS),
        }

    def _cache_ohlcv(self, symbol: str, timeframe: str, days: int = 30):
        cache_key = (symbol, timeframe)
        if cache_key not in self.ohlcv_cache:
            temp_bot = TradingService(
                user_id=self.user_id,
                starting_balance=self.starting_balance,
                selected_coins=[symbol],
                timeframe=timeframe,
                api_key=self.api_key,
                api_secret=self.api_secret,
                api_password=self.api_password
            )
            minutes_per_candle = temp_bot._get_timeframe_minutes()
            periods = int(days * 24 * 60 / minutes_per_candle)
            df = temp_bot.fetch_ohlcv(symbol=symbol, limit=min(periods, LOOKBACK_PERIODS * 3))
            if df is not None:
                df = temp_bot.calculate_indicators(df)
                df = temp_bot.create_labels(df)
                df = df.dropna()
            self.ohlcv_cache[cache_key] = df
        return self.ohlcv_cache.get(cache_key)

    def _run_single_backtest(self, timeframe: str, params: dict, days: int = 30):
        TAKER_FEE = 0.0006

        all_trades = []
        total_balance = 0
        balance_per_coin = self.starting_balance / len(self.selected_coins)

        for symbol in self.selected_coins:
            df = self._cache_ohlcv(symbol, timeframe, days)
            if df is None or len(df) < 100:
                total_balance += balance_per_coin
                continue

            train_size = len(df) // 2
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]

            temp_bot = TradingService(
                user_id=self.user_id,
                starting_balance=balance_per_coin,
                selected_coins=[symbol],
                timeframe=timeframe
            )
            X_train, _ = temp_bot.prepare_features(train_df)
            y_train = train_df['signal']
            mask = y_train != 0
            X_train, y_train = X_train[mask], y_train[mask]

            if len(X_train) < 20:
                total_balance += balance_per_coin
                continue

            try:
                imputer = SimpleImputer(strategy='mean')
                scaler = StandardScaler()
                X_imputed = imputer.fit_transform(X_train)
                X_scaled = scaler.fit_transform(X_imputed)

                try:
                    smote = SMOTE(random_state=42, k_neighbors=min(3, len(y_train[y_train==1])-1, len(y_train[y_train==-1])-1))
                    X_resampled, y_resampled = smote.fit_resample(X_scaled, y_train)
                except:
                    X_resampled, y_resampled = X_scaled, y_train

                tscv = TimeSeriesSplit(n_splits=3)
                svm = SVC(probability=True, random_state=42)
                grid = GridSearchCV(svm, SVM_PARAMS, cv=tscv, scoring='f1_weighted', n_jobs=1)
                grid.fit(X_resampled, y_resampled)
                model = grid.best_estimator_
            except:
                total_balance += balance_per_coin
                continue

            balance = balance_per_coin
            position = None
            entry_price = 0
            high_water_mark = 0
            low_water_mark = 0
            trades = []
            last_trade_candle = -999

            minutes_per_candle = temp_bot._get_timeframe_minutes()
            cooldown_candles = max(1, math.ceil(params['trade_cooldown'] / 60 / minutes_per_candle))
            X_test, _ = temp_bot.prepare_features(test_df)

            for i in range(len(test_df)):
                try:
                    row = test_df.iloc[i]
                    price = row['close']
                    X_row = X_test.iloc[[i]]
                    X_imp = imputer.transform(X_row)
                    X_sc = scaler.transform(X_imp)
                    signal = model.predict(X_sc)[0]
                    conf = max(model.predict_proba(X_sc)[0])

                    if i - last_trade_candle < cooldown_candles:
                        continue

                    if position is None and signal != 0 and conf >= params['min_confidence']:
                        margin = balance * params['risk_per_trade']
                        if margin <= 0 or margin > balance * 0.95:
                            continue

                        notional = margin * params['leverage']
                        size = notional / price
                        entry_fee = notional * TAKER_FEE

                        position = {
                            'side': 'long' if signal == 1 else 'short',
                            'size': size, 'margin': margin, 'entry_fee': entry_fee
                        }
                        entry_price = price
                        high_water_mark = price
                        low_water_mark = price
                        last_trade_candle = i

                    elif position is not None:
                        if position['side'] == 'long':
                            high_water_mark = max(high_water_mark, price)
                        else:
                            low_water_mark = min(low_water_mark, price)

                        pnl_pct = (
                            (price - entry_price) / entry_price if position['side'] == 'long'
                            else (entry_price - price) / entry_price
                        )

                        should_exit = False
                        ts = params.get('trailing_stop_pct', 0.01)

                        if pnl_pct <= -params['stop_loss_pct']:
                            should_exit = True
                        elif pnl_pct >= params['take_profit_pct']:
                            should_exit = True
                        else:
                            if position['side'] == 'long':
                                if price <= high_water_mark * (1 - ts) and pnl_pct > 0:
                                    should_exit = True
                            else:
                                if price >= low_water_mark * (1 + ts) and pnl_pct > 0:
                                    should_exit = True

                            if not should_exit:
                                exit_conf = params['min_confidence'] * 0.8
                                if ((signal == -1 and position['side'] == 'long') or
                                        (signal == 1 and position['side'] == 'short')):
                                    if conf >= exit_conf:
                                        should_exit = True

                        if should_exit:
                            price_change = (
                                (price - entry_price) if position['side'] == 'long'
                                else (entry_price - price)
                            )
                            pnl_amount = price_change * position['size']
                            exit_notional = position['size'] * price
                            exit_fee = exit_notional * TAKER_FEE
                            total_fees = position.get('entry_fee', 0) + exit_fee
                            net_pnl = pnl_amount - total_fees
                            balance += net_pnl

                            trades.append({'pnl': net_pnl})
                            position = None
                            last_trade_candle = i
                except:
                    continue

            all_trades.extend(trades)
            total_balance += balance

        total_trades = len(all_trades)
        winning_trades = len([t for t in all_trades if t['pnl'] > 0])
        total_pnl = sum(t['pnl'] for t in all_trades)
        total_return_pct = ((total_balance - self.starting_balance) / self.starting_balance) * 100
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        return {
            'total_return': total_return_pct,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'final_balance': total_balance
        }

    def _train_and_cache_model(self, symbol: str, timeframe: str, days: int = 30):
        cache_key = (symbol, timeframe)
        if cache_key in self.model_cache:
            return

        df = self._cache_ohlcv(symbol, timeframe, days)
        if df is None or len(df) < 100:
            self.model_cache[cache_key] = None
            return

        train_size = len(df) // 2
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        temp_bot = TradingService(
            user_id=self.user_id,
            starting_balance=self.starting_balance,
            selected_coins=[symbol],
            timeframe=timeframe
        )
        X_train, _ = temp_bot.prepare_features(train_df)
        y_train = train_df['signal']
        mask = y_train != 0
        X_train, y_train = X_train[mask], y_train[mask]

        if len(X_train) < 20:
            self.model_cache[cache_key] = None
            return

        try:
            imputer = SimpleImputer(strategy='mean')
            scaler = StandardScaler()
            X_imputed = imputer.fit_transform(X_train)
            X_scaled = scaler.fit_transform(X_imputed)

            try:
                smote = SMOTE(random_state=42, k_neighbors=min(3, len(y_train[y_train==1])-1, len(y_train[y_train==-1])-1))
                X_resampled, y_resampled = smote.fit_resample(X_scaled, y_train)
            except:
                X_resampled, y_resampled = X_scaled, y_train

            tscv = TimeSeriesSplit(n_splits=3)
            svm = SVC(probability=True, random_state=42)
            grid = GridSearchCV(svm, SVM_PARAMS, cv=tscv, scoring='f1_weighted', n_jobs=1)
            grid.fit(X_resampled, y_resampled)
            model = grid.best_estimator_

            X_test, _ = temp_bot.prepare_features(test_df)

            self.model_cache[cache_key] = {
                'model': model,
                'imputer': imputer,
                'scaler': scaler,
                'test_df': test_df,
                'X_test': X_test,
                'temp_bot': temp_bot
            }
            logger.info(f"Trained model for {symbol} {timeframe}")
        except Exception as e:
            logger.warning(f"Failed to train model for {symbol} {timeframe}: {e}")
            self.model_cache[cache_key] = None

    def _run_cached_backtest(self, timeframe: str, params: dict):
        TAKER_FEE = 0.0006

        all_trades = []
        total_balance = 0
        balance_per_coin = self.starting_balance / len(self.selected_coins)

        for symbol in self.selected_coins:
            cache_key = (symbol, timeframe)
            cached = self.model_cache.get(cache_key)

            if cached is None:
                total_balance += balance_per_coin
                continue

            model = cached['model']
            imputer = cached['imputer']
            scaler = cached['scaler']
            test_df = cached['test_df']
            X_test = cached['X_test']
            temp_bot = cached['temp_bot']

            balance = balance_per_coin
            position = None
            entry_price = 0
            high_water_mark = 0
            low_water_mark = 0
            trades = []
            last_trade_candle = -999

            minutes_per_candle = temp_bot._get_timeframe_minutes()
            cooldown_candles = max(1, math.ceil(params['trade_cooldown'] / 60 / minutes_per_candle))

            for i in range(len(test_df)):
                try:
                    row = test_df.iloc[i]
                    price = row['close']
                    X_row = X_test.iloc[[i]]
                    X_imp = imputer.transform(X_row)
                    X_sc = scaler.transform(X_imp)
                    signal = model.predict(X_sc)[0]
                    conf = max(model.predict_proba(X_sc)[0])

                    if i - last_trade_candle < cooldown_candles:
                        continue

                    if position is None and signal != 0 and conf >= params['min_confidence']:
                        margin = balance * params['risk_per_trade']
                        if margin <= 0 or margin > balance * 0.95:
                            continue

                        notional = margin * params['leverage']
                        size = notional / price
                        entry_fee = notional * TAKER_FEE

                        position = {
                            'side': 'long' if signal == 1 else 'short',
                            'size': size, 'margin': margin, 'entry_fee': entry_fee
                        }
                        entry_price = price
                        high_water_mark = price
                        low_water_mark = price
                        last_trade_candle = i

                    elif position is not None:
                        if position['side'] == 'long':
                            high_water_mark = max(high_water_mark, price)
                        else:
                            low_water_mark = min(low_water_mark, price)

                        pnl_pct = (
                            (price - entry_price) / entry_price if position['side'] == 'long'
                            else (entry_price - price) / entry_price
                        )

                        should_exit = False
                        ts = params.get('trailing_stop_pct', 0.01)

                        if pnl_pct <= -params['stop_loss_pct']:
                            should_exit = True
                        elif pnl_pct >= params['take_profit_pct']:
                            should_exit = True
                        else:
                            if position['side'] == 'long':
                                if price <= high_water_mark * (1 - ts) and pnl_pct > 0:
                                    should_exit = True
                            else:
                                if price >= low_water_mark * (1 + ts) and pnl_pct > 0:
                                    should_exit = True

                            if not should_exit:
                                exit_conf = params['min_confidence'] * 0.8
                                if signal != 0 and conf >= exit_conf:
                                    expected_side = 'long' if signal == 1 else 'short'
                                    if expected_side != position['side']:
                                        should_exit = True

                        if should_exit:
                            notional = position['size'] * price
                            exit_fee = notional * TAKER_FEE
                            raw_pnl = pnl_pct * position['margin'] * params['leverage']
                            net_pnl = raw_pnl - position['entry_fee'] - exit_fee
                            balance += net_pnl
                            trades.append({'pnl': net_pnl})
                            position = None
                            last_trade_candle = i
                except:
                    continue

            all_trades.extend(trades)
            total_balance += balance

        total_trades = len(all_trades)
        winning_trades = len([t for t in all_trades if t['pnl'] > 0])
        total_pnl = sum(t['pnl'] for t in all_trades)
        total_return_pct = ((total_balance - self.starting_balance) / self.starting_balance) * 100
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        return {
            'total_return': total_return_pct,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'final_balance': total_balance
        }

    def _calculate_score(self, result: dict) -> float:
        if result['total_trades'] < self.MIN_TRADES:
            return -999
        if result['total_return'] < 0:
            return -999

        roi_score = min(result['total_return'] / 100, 1.0)
        winrate_score = result['win_rate'] / 100
        trade_score = min(result['total_trades'] / 100, 1.0)

        score = (0.5 * roi_score) + (0.3 * winrate_score) + (0.2 * trade_score)
        return score

    def optimize(self, days: int = 30, progress_callback=None):
        import random
        import time
        print(f"[OPT] User {self.user_id}: optimize() called, days={days}", flush=True)

        random.seed(42)

        self.results = []
        self.total_tests = len(self.TIMEFRAMES) * self.SAMPLES_PER_TIMEFRAME
        self.current_test = 0

        print(f"[OPT] User {self.user_id}: Starting {self.total_tests} tests across {len(self.TIMEFRAMES)} timeframes", flush=True)
        logger.info(f"Starting optimization: {self.total_tests} tests across {len(self.TIMEFRAMES)} timeframes")

        # PHASE 1: Fetch ALL data upfront (one API call per coin/timeframe combo)
        total_fetches = len(self.TIMEFRAMES) * len(self.selected_coins)
        fetch_count = 0
        logger.info(f"Phase 1: Fetching data for {len(self.selected_coins)} coins × {len(self.TIMEFRAMES)} timeframes ({total_fetches} total)")
        self.ohlcv_cache = {}
        self.phase = "fetching"

        for tf_idx, timeframe in enumerate(self.TIMEFRAMES):
            logger.info(f"Fetching data for timeframe: {timeframe} ({tf_idx+1}/{len(self.TIMEFRAMES)})")
            for symbol in self.selected_coins:
                fetch_count += 1
                self.progress = (fetch_count / total_fetches) * 20
                if progress_callback:
                    progress_callback(self.progress)
                try:
                    self._cache_ohlcv(symbol, timeframe, days)
                    time.sleep(0.2)
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol} {timeframe}: {e}")

        logger.info(f"Phase 1 complete: Cached {len(self.ohlcv_cache)} datasets")

        # PHASE 2: Train models ONCE per timeframe, then test parameters
        logger.info("Phase 2: Training models and testing parameters")
        self.phase = "testing"
        self.model_cache = {}

        for tf_idx, timeframe in enumerate(self.TIMEFRAMES):
            logger.info(f"Optimizing timeframe: {timeframe} ({tf_idx+1}/{len(self.TIMEFRAMES)})")

            for symbol in self.selected_coins:
                self._train_and_cache_model(symbol, timeframe, days)

            for sample_idx in range(self.SAMPLES_PER_TIMEFRAME):
                self.current_test += 1
                self.progress = 20 + (self.current_test / self.total_tests) * 80

                if progress_callback:
                    progress_callback(self.progress)

                if sample_idx % 50 == 0:
                    time.sleep(0.01)

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
                            'trade_cooldown': params['trade_cooldown'],
                            'min_confidence': params['min_confidence'],
                            'total_return': round(result['total_return'], 2),
                            'win_rate': round(result['win_rate'], 2),
                            'total_trades': result['total_trades'],
                            'total_pnl': round(result['total_pnl'], 2),
                            'score': round(score, 4)
                        })
                except Exception as e:
                    logger.warning(f"Backtest failed: {e}")
                    continue

        self.results.sort(key=lambda x: x['score'], reverse=True)
        top_results = self.results[:20]

        logger.info(f"Optimization complete. Found {len(self.results)} valid configurations.")

        return {
            'total_tested': self.current_test,
            'valid_configs': len(self.results),
            'top_configs': top_results,
            'selected_coins': self.selected_coins,
            'days_tested': days
        }
