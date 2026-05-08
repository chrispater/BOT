"""
Signal Engine Module
Calculates technical indicators and detects momentum signals.
Uses pandas_ta for indicators and custom logic for signal generation.
Enhanced with backtesting for confidence score adjustment.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from backtesting_engine import BacktestingEngine
from ml_predictor import MLPredictor


@dataclass
class Signal:
    """Represents a trading signal with metadata."""
    symbol: str
    signal_type: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0-100 score (adjusted with backtesting and ML)
    indicators: Dict[str, float]
    reason: str
    timestamp: pd.Timestamp
    original_confidence: Optional[float] = None  # Pre-backtest confidence
    indicators_present: Optional[List[str]] = None  # Which indicators triggered
    ml_prediction: Optional[str] = None  # ML predicted direction (BREAKOUT/BREAKDOWN/NEUTRAL)
    ml_confidence: Optional[float] = None  # ML prediction confidence (0-100)
    ml_enabled: Optional[bool] = False  # Whether ML contributed to this signal


class SignalEngine:
    """Calculate indicators and generate trading signals with backtesting support."""
    
    def __init__(self, backtesting_engine: Optional[BacktestingEngine] = None):
        """
        Initialize the signal engine with default parameters.
        
        Args:
            backtesting_engine: Optional backtesting engine for confidence adjustment
        """
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.volume_spike_threshold = 2.5  # Volume must be 2.5x average
        self.bb_width_threshold = 0.02  # Bollinger Band width for squeeze detection
        self.backtesting_engine = backtesting_engine
        self.ml_predictor = MLPredictor()  # ML-based breakout/breakdown prediction
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicator columns
        """
        if df is None or df.empty or len(df) < 50:
            return df
        
        df = df.copy()
        
        # RSI (Relative Strength Index)
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
        
        # Stochastic Oscillator
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None:
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
        
        # Bollinger Bands
        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None and isinstance(bbands, pd.DataFrame) and not bbands.empty:
            # pandas_ta returns column names like 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0'
            # Get the actual column names
            bb_cols = bbands.columns.tolist()
            
            # Find the upper, middle, and lower band columns
            bb_upper_col = [col for col in bb_cols if 'BBU' in col]
            bb_middle_col = [col for col in bb_cols if 'BBM' in col]
            bb_lower_col = [col for col in bb_cols if 'BBL' in col]
            
            if bb_upper_col and bb_middle_col and bb_lower_col:
                df['bb_upper'] = bbands[bb_upper_col[0]]
                df['bb_middle'] = bbands[bb_middle_col[0]]
                df['bb_lower'] = bbands[bb_lower_col[0]]
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price change and volatility
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['close'].rolling(window=20).std()
        df['volatility_norm'] = df['volatility'] / df['close']
        
        # Average True Range (ATR) for volatility
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Moving averages
        df['ema_9'] = ta.ema(df['close'], length=9)
        df['ema_21'] = ta.ema(df['close'], length=21)
        df['ema_50'] = ta.ema(df['close'], length=50)
        
        return df
    
    def detect_volume_spike(self, df: pd.DataFrame) -> bool:
        """Check if current volume is significantly above average."""
        if 'volume_ratio' not in df.columns or df.empty:
            return False
        
        latest_ratio = df['volume_ratio'].iloc[-1]
        return latest_ratio > self.volume_spike_threshold
    
    def detect_rsi_extreme(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Detect RSI extremes (oversold/overbought).
        
        Returns:
            (is_extreme, direction)
        """
        if 'rsi' not in df.columns or df.empty:
            return False, 'neutral'
        
        latest_rsi = df['rsi'].iloc[-1]
        
        if pd.isna(latest_rsi):
            return False, 'neutral'
        
        if latest_rsi < self.rsi_oversold:
            return True, 'oversold'
        elif latest_rsi > self.rsi_overbought:
            return True, 'overbought'
        
        return False, 'neutral'
    
    def detect_macd_crossover(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Detect MACD signal line crossovers.
        
        Returns:
            (is_crossover, direction)
        """
        if 'macd' not in df.columns or 'macd_signal' not in df.columns or len(df) < 2:
            return False, 'neutral'
        
        # Current and previous values
        macd_curr = df['macd'].iloc[-1]
        signal_curr = df['macd_signal'].iloc[-1]
        macd_prev = df['macd'].iloc[-2]
        signal_prev = df['macd_signal'].iloc[-2]
        
        if pd.isna(macd_curr) or pd.isna(signal_curr):
            return False, 'neutral'
        
        # Bullish crossover: MACD crosses above signal
        if macd_prev <= signal_prev and macd_curr > signal_curr:
            return True, 'bullish'
        
        # Bearish crossover: MACD crosses below signal
        if macd_prev >= signal_prev and macd_curr < signal_curr:
            return True, 'bearish'
        
        return False, 'neutral'
    
    def detect_bollinger_breakout(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Detect Bollinger Band breakouts or squeezes.
        
        Returns:
            (is_breakout, direction)
        """
        if 'bb_percent' not in df.columns or df.empty:
            return False, 'neutral'
        
        bb_pct = df['bb_percent'].iloc[-1]
        bb_width = df['bb_width'].iloc[-1]
        
        if pd.isna(bb_pct) or pd.isna(bb_width):
            return False, 'neutral'
        
        # Squeeze detection (low volatility, potential breakout coming)
        if bb_width < self.bb_width_threshold:
            return True, 'squeeze'
        
        # Upper band breakout
        if bb_pct > 1.0:
            return True, 'upper_breakout'
        
        # Lower band breakout
        if bb_pct < 0.0:
            return True, 'lower_breakout'
        
        return False, 'neutral'
    
    def calculate_z_score(self, df: pd.DataFrame, column: str = 'close', window: int = 20) -> float:
        """
        Calculate z-score for anomaly detection.
        
        Args:
            df: DataFrame with price data
            column: Column to calculate z-score for
            window: Rolling window size
            
        Returns:
            Latest z-score value
        """
        if column not in df.columns or len(df) < window:
            return 0.0
        
        rolling_mean = df[column].rolling(window=window).mean()
        rolling_std = df[column].rolling(window=window).std()
        
        latest_value = df[column].iloc[-1]
        latest_mean = rolling_mean.iloc[-1]
        latest_std = rolling_std.iloc[-1]
        
        if latest_std == 0 or pd.isna(latest_std):
            return 0.0
        
        z_score = (latest_value - latest_mean) / latest_std
        return z_score
    
    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """
        Generate a trading signal based on multiple indicators.
        
        Args:
            symbol: Trading pair symbol
            df: DataFrame with calculated indicators
            
        Returns:
            Signal object or None if no strong signal
        """
        if df is None or df.empty:
            return None
        
        # Calculate indicators if not already present
        if 'rsi' not in df.columns:
            df = self.calculate_indicators(df)
        
        # Collect current indicator values
        latest = df.iloc[-1]
        
        indicators = {
            'price': latest['close'],
            'rsi': latest.get('rsi', np.nan),
            'macd': latest.get('macd', np.nan),
            'volume_ratio': latest.get('volume_ratio', np.nan),
            'bb_percent': latest.get('bb_percent', np.nan),
            'stoch_k': latest.get('stoch_k', np.nan)
        }
        
        # Signal scoring system
        bullish_score = 0
        bearish_score = 0
        reasons = []
        indicators_triggered = []  # Track which indicators fired
        
        # RSI signals
        is_rsi_extreme, rsi_dir = self.detect_rsi_extreme(df)
        if is_rsi_extreme:
            if rsi_dir == 'oversold':
                bullish_score += 25
                reasons.append(f"RSI oversold ({indicators['rsi']:.1f})")
                indicators_triggered.append('RSI')
            elif rsi_dir == 'overbought':
                bearish_score += 25
                reasons.append(f"RSI overbought ({indicators['rsi']:.1f})")
                indicators_triggered.append('RSI')
        
        # MACD crossover
        is_macd_cross, macd_dir = self.detect_macd_crossover(df)
        if is_macd_cross:
            if macd_dir == 'bullish':
                bullish_score += 30
                reasons.append("MACD bullish crossover")
                indicators_triggered.append('MACD')
            elif macd_dir == 'bearish':
                bearish_score += 30
                reasons.append("MACD bearish crossover")
                indicators_triggered.append('MACD')
        
        # Volume spike confirmation
        volume_spike = self.detect_volume_spike(df)
        if volume_spike:
            bullish_score += 15
            bearish_score += 15  # Volume confirms both directions
            reasons.append(f"Volume spike ({indicators['volume_ratio']:.1f}x)")
            indicators_triggered.append('Volume')
        
        # Bollinger Band breakout
        is_bb_breakout, bb_dir = self.detect_bollinger_breakout(df)
        if is_bb_breakout:
            if bb_dir == 'lower_breakout':
                bullish_score += 20
                reasons.append("BB lower band bounce")
                indicators_triggered.append('BollingerBands')
            elif bb_dir == 'upper_breakout':
                bearish_score += 20
                reasons.append("BB upper band rejection")
                indicators_triggered.append('BollingerBands')
            elif bb_dir == 'squeeze':
                reasons.append("BB squeeze (low volatility)")
                indicators_triggered.append('BollingerBands')
        
        # Stochastic confirmation
        if not pd.isna(indicators['stoch_k']):
            if indicators['stoch_k'] < 20:
                bullish_score += 10
                reasons.append("Stochastic oversold")
                indicators_triggered.append('Stochastic')
            elif indicators['stoch_k'] > 80:
                bearish_score += 10
                reasons.append("Stochastic overbought")
                indicators_triggered.append('Stochastic')
        
        # Price z-score anomaly
        z_score = self.calculate_z_score(df, 'close', 20)
        if abs(z_score) > 2:
            reasons.append(f"Price anomaly (z={z_score:.2f})")
            indicators_triggered.append('PriceAnomaly')
            if z_score < -2:
                bullish_score += 15
            else:
                bearish_score += 15
        
        # Determine signal type and confidence
        if bullish_score > bearish_score and bullish_score >= 40:
            signal_type = 'bullish'
            original_confidence = min(bullish_score, 100)
        elif bearish_score > bullish_score and bearish_score >= 40:
            signal_type = 'bearish'
            original_confidence = min(bearish_score, 100)
        else:
            return None  # No strong signal
        
        reason = " | ".join(reasons)
        
        # Apply backtesting adjustment if available
        adjusted_confidence = original_confidence
        if self.backtesting_engine:
            adjusted_confidence = self.backtesting_engine.get_adjusted_confidence(
                original_confidence, 
                indicators_triggered
            )
            
            # Record this signal for future backtesting
            self.backtesting_engine.record_signal(
                symbol=symbol,
                signal_type=signal_type,
                entry_price=latest['close'],
                confidence=adjusted_confidence,
                reason=reason
            )
        
        # Apply ML prediction to further enhance confidence
        ml_prediction = None
        ml_confidence = 0.0
        ml_enabled = False
        
        try:
            # Prepare features for ML
            price_1h = ((df['close'].iloc[-1] - df['close'].iloc[-12]) / df['close'].iloc[-12] * 100) if len(df) >= 12 else 0
            price_24h = ((df['close'].iloc[-1] - df['close'].iloc[-288]) / df['close'].iloc[-288] * 100) if len(df) >= 288 else 0
            price_7d = ((df['close'].iloc[-1] - df['close'].iloc[-2016]) / df['close'].iloc[-2016] * 100) if len(df) >= 2016 else 0
            
            avg_volume = df['volume'].rolling(20).mean().iloc[-1] if 'volume' in df.columns else 1
            current_volume = df['volume'].iloc[-1] if 'volume' in df.columns else 1
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            volatility = df['close'].pct_change().std() * np.sqrt(len(df)) if len(df) > 1 else 0
            
            ml_features = {
                'rsi': indicators.get('rsi', 50),
                'macd': indicators.get('macd', 0),
                'bb_position': indicators.get('bb_percent', 0.5) / 100 if not pd.isna(indicators.get('bb_percent')) else 0.5,
                'volume_ratio': volume_ratio,
                'price_change_1h': price_1h,
                'price_change_24h': price_24h,
                'price_change_7d': price_7d,
                'volatility': volatility,
                'trend_strength': bullish_score / 100 if signal_type == 'bullish' else -bearish_score / 100,
                'volume_trend': volume_ratio - 1,
            }
            
            # Get ML prediction
            ml_result = self.ml_predictor.predict_breakout_probability(ml_features)
            ml_prediction = ml_result['prediction']
            ml_confidence = ml_result['confidence']
            ml_enabled = ml_result['ml_enabled']
            
            # Boost confidence if ML agrees with signal
            if ml_enabled:
                if (signal_type == 'bullish' and ml_prediction == 'BREAKOUT') or \
                   (signal_type == 'bearish' and ml_prediction == 'BREAKDOWN'):
                    # ML agrees - boost confidence
                    boost = ml_confidence * 0.2  # Up to +20 points
                    adjusted_confidence = min(100, adjusted_confidence + boost)
                    reasons.append(f"ML confirms {ml_prediction} ({ml_confidence:.0f}% confident)")
                elif ml_prediction != 'NEUTRAL':
                    # ML disagrees - reduce confidence
                    penalty = ml_confidence * 0.15  # Up to -15 points
                    adjusted_confidence = max(0, adjusted_confidence - penalty)
                    reasons.append(f"ML predicts {ml_prediction} (conflict)")
                
                reason = " | ".join(reasons)
        
        except Exception as e:
            print(f"ML prediction error in signal generation: {e}")
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=adjusted_confidence,
            indicators=indicators,
            reason=reason,
            timestamp=latest.name,
            original_confidence=original_confidence,
            indicators_present=indicators_triggered,
            ml_prediction=ml_prediction,
            ml_confidence=ml_confidence,
            ml_enabled=ml_enabled
        )
    
    def scan_multiple(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Scan multiple symbols and generate signals.
        
        Args:
            data: Dictionary mapping symbols to DataFrames
            
        Returns:
            List of Signal objects, sorted by confidence
        """
        signals = []
        
        for symbol, df in data.items():
            signal = self.generate_signal(symbol, df)
            if signal:
                signals.append(signal)
        
        # Sort by confidence (highest first)
        signals.sort(key=lambda s: s.confidence, reverse=True)
        
        return signals
