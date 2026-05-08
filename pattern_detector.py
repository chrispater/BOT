"""
Advanced Candlestick and Price Action Pattern Detection
Finds the next big pump or crash using bulletproof technical patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import time


@dataclass
class PatternSignal:
    """Signal from pattern detection"""
    pattern_name: str
    signal_type: str  # "BULLISH_PUMP", "BEARISH_CRASH", "NEUTRAL"
    confidence: float  # 0-100
    description: str
    timeframe: str  # "1h", "4h", "24h"
    strength: str  # "STRONG", "MODERATE", "WEAK"


class PatternDetector:
    """
    Advanced pattern detection for finding pumps and crashes.
    Combines candlestick patterns + price action + volume analysis.
    """
    
    def __init__(self):
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        self.last_request_time = 0
        self.min_request_interval = 1.5  # 1.5 seconds between requests to avoid rate limits
    
    def analyze_token_patterns(self, coingecko_id: str) -> List[PatternSignal]:
        """
        Analyze token for all patterns across multiple timeframes.
        
        Args:
            coingecko_id: CoinGecko token ID
            
        Returns:
            List of detected patterns with signals
        """
        patterns = []
        
        # Get OHLC data for multiple timeframes
        # days=1 gives 30-min candles (48 candles for 24h)
        ohlc_short = self._fetch_ohlc_data(coingecko_id, days=1)  
        # days=7 gives 4-hour candles (42 candles for 7 days)
        ohlc_medium = self._fetch_ohlc_data(coingecko_id, days=7)  
        
        if ohlc_short is not None and len(ohlc_short) > 0:
            # Short-term patterns (30-min candles, recent action)
            patterns.extend(self._detect_candlestick_patterns(ohlc_short, "30m"))
            patterns.extend(self._detect_price_action_patterns(ohlc_short, "30m"))
            patterns.extend(self._detect_volume_patterns(ohlc_short, "30m"))
        
        if ohlc_medium is not None and len(ohlc_medium) > 0:
            # Medium-term patterns (4h candles, broader trend)
            patterns.extend(self._detect_candlestick_patterns(ohlc_medium, "4h"))
            patterns.extend(self._detect_price_action_patterns(ohlc_medium, "4h"))
        
        # Sort by confidence (highest first)
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        return patterns
    
    def _fetch_ohlc_data(self, coingecko_id: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch OHLC candlestick data from CoinGecko with rate limiting and retries"""
        cache_key = f"ohlc_{coingecko_id}_{days}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_timeout:
                return cached_data
        
        # Rate limiting: wait if needed
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        # Retry logic with exponential backoff
        max_retries = 2
        for attempt in range(max_retries):
            try:
                url = f"{self.coingecko_base}/coins/{coingecko_id}/ohlc"
                params = {'vs_currency': 'usd', 'days': days}
                
                self.last_request_time = time.time()
                response = requests.get(url, params=params, timeout=8)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if not data or len(data) == 0:
                        return None
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Calculate volume proxy (high-low range * close as approximation)
                    df['volume'] = (df['high'] - df['low']) * df['close']
                    
                    self.cache[cache_key] = (time.time(), df)
                    return df
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"Rate limited, waiting {wait_time}s before retry {attempt+1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    return None
                
            except requests.Timeout:
                print(f"Timeout fetching OHLC for {coingecko_id} (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(1)
                continue
            except Exception as e:
                print(f"Error fetching OHLC for {coingecko_id}: {e}")
                return None
        
        return None
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame, timeframe: str) -> List[PatternSignal]:
        """Detect classic candlestick patterns"""
        patterns = []
        
        if len(df) < 5:
            return patterns
        
        # Get last 5 candles for pattern detection
        recent = df.tail(5).reset_index(drop=True)
        last = recent.iloc[-1]
        prev = recent.iloc[-2] if len(recent) > 1 else last
        
        # Calculate candle properties
        body = abs(last['close'] - last['open'])
        upper_wick = last['high'] - max(last['open'], last['close'])
        lower_wick = min(last['open'], last['close']) - last['low']
        total_range = last['high'] - last['low']
        
        # Avoid division by zero
        if total_range == 0:
            return patterns
        
        body_pct = body / total_range
        upper_wick_pct = upper_wick / total_range
        lower_wick_pct = lower_wick / total_range
        
        # HAMMER (bullish reversal - signals bottom, next pump coming)
        # More lenient: lower wick > 50%, small body < 40%
        if lower_wick_pct > 0.5 and body_pct < 0.4 and last['close'] > last['open']:
            confidence = 60 + (lower_wick_pct * 30)  # 60-90%
            patterns.append(PatternSignal(
                pattern_name="HAMMER",
                signal_type="BULLISH_PUMP",
                confidence=min(90, confidence),
                description="Hammer pattern - Strong rejection of lows, buyers stepping in. PUMP likely!",
                timeframe=timeframe,
                strength="STRONG" if lower_wick_pct > 0.65 else "MODERATE"
            ))
        
        # SHOOTING STAR (bearish reversal - signals top, crash coming)
        # More lenient: upper wick > 50%, small body < 40%
        if upper_wick_pct > 0.5 and body_pct < 0.4 and last['close'] < last['open']:
            confidence = 60 + (upper_wick_pct * 30)  # 60-90%
            patterns.append(PatternSignal(
                pattern_name="SHOOTING_STAR",
                signal_type="BEARISH_CRASH",
                confidence=min(90, confidence),
                description="Shooting Star - Strong rejection at highs, sellers taking control. CRASH likely!",
                timeframe=timeframe,
                strength="STRONG" if upper_wick_pct > 0.65 else "MODERATE"
            ))
        
        # DOJI (indecision - trend reversal possible)
        if body_pct < 0.1:
            trend = self._determine_trend(recent)
            if trend == "UPTREND":
                patterns.append(PatternSignal(
                    pattern_name="DOJI_TOP",
                    signal_type="BEARISH_CRASH",
                    confidence=60.0,
                    description="Doji at top - Indecision after rally, potential reversal. Watch for CRASH!",
                    timeframe=timeframe,
                    strength="MODERATE"
                ))
            elif trend == "DOWNTREND":
                patterns.append(PatternSignal(
                    pattern_name="DOJI_BOTTOM",
                    signal_type="BULLISH_PUMP",
                    confidence=60.0,
                    description="Doji at bottom - Indecision after selloff, potential reversal. Watch for PUMP!",
                    timeframe=timeframe,
                    strength="MODERATE"
                ))
        
        # BULLISH ENGULFING (strong pump signal)
        if len(recent) >= 2:
            if (prev['close'] < prev['open'] and  # Previous red candle
                last['close'] > last['open'] and  # Current green candle
                last['open'] <= prev['close'] and  # Opens at or below prev close
                last['close'] >= prev['open']):    # Closes above prev open
                
                patterns.append(PatternSignal(
                    pattern_name="BULLISH_ENGULFING",
                    signal_type="BULLISH_PUMP",
                    confidence=85.0,
                    description="Bullish Engulfing - Bulls overtaking bears completely. Strong PUMP signal!",
                    timeframe=timeframe,
                    strength="STRONG"
                ))
        
        # BEARISH ENGULFING (strong crash signal)
        if len(recent) >= 2:
            if (prev['close'] > prev['open'] and  # Previous green candle
                last['close'] < last['open'] and  # Current red candle
                last['open'] >= prev['close'] and  # Opens at or above prev close
                last['close'] <= prev['open']):    # Closes below prev open
                
                patterns.append(PatternSignal(
                    pattern_name="BEARISH_ENGULFING",
                    signal_type="BEARISH_CRASH",
                    confidence=85.0,
                    description="Bearish Engulfing - Bears overtaking bulls completely. Strong CRASH signal!",
                    timeframe=timeframe,
                    strength="STRONG"
                ))
        
        # MORNING STAR (bullish reversal - bottom found, pump incoming)
        if len(recent) >= 3:
            first = recent.iloc[-3]
            middle = recent.iloc[-2]
            last = recent.iloc[-1]
            
            first_body = abs(first['close'] - first['open'])
            middle_body = abs(middle['close'] - middle['open'])
            last_body = abs(last['close'] - last['open'])
            
            if (first['close'] < first['open'] and  # First candle bearish
                middle_body < first_body * 0.3 and  # Middle small body (indecision)
                last['close'] > last['open'] and    # Last candle bullish
                last['close'] > first['open']):      # Closes above first open
                
                patterns.append(PatternSignal(
                    pattern_name="MORNING_STAR",
                    signal_type="BULLISH_PUMP",
                    confidence=90.0,
                    description="Morning Star - Classic bottom reversal pattern. Major PUMP incoming!",
                    timeframe=timeframe,
                    strength="STRONG"
                ))
        
        # EVENING STAR (bearish reversal - top found, crash incoming)
        if len(recent) >= 3:
            first = recent.iloc[-3]
            middle = recent.iloc[-2]
            last = recent.iloc[-1]
            
            first_body = abs(first['close'] - first['open'])
            middle_body = abs(middle['close'] - middle['open'])
            last_body = abs(last['close'] - last['open'])
            
            if (first['close'] > first['open'] and  # First candle bullish
                middle_body < first_body * 0.3 and  # Middle small body
                last['close'] < last['open'] and    # Last candle bearish
                last['close'] < first['open']):      # Closes below first open
                
                patterns.append(PatternSignal(
                    pattern_name="EVENING_STAR",
                    signal_type="BEARISH_CRASH",
                    confidence=90.0,
                    description="Evening Star - Classic top reversal pattern. Major CRASH incoming!",
                    timeframe=timeframe,
                    strength="STRONG"
                ))
        
        return patterns
    
    def _detect_price_action_patterns(self, df: pd.DataFrame, timeframe: str) -> List[PatternSignal]:
        """Detect price action patterns (flags, triangles, breakouts)"""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        recent = df.tail(20).reset_index(drop=True)
        
        # BREAKOUT detection (pump starting)
        resistance = recent['high'].rolling(window=10).max().iloc[-1]
        current_price = recent['close'].iloc[-1]
        prev_price = recent['close'].iloc[-2]
        
        if prev_price < resistance * 0.99 and current_price > resistance:
            patterns.append(PatternSignal(
                pattern_name="BREAKOUT",
                signal_type="BULLISH_PUMP",
                confidence=80.0,
                description="Resistance breakout! Price breaking above recent highs. PUMP starting!",
                timeframe=timeframe,
                strength="STRONG"
            ))
        
        # BREAKDOWN detection (crash starting)
        support = recent['low'].rolling(window=10).min().iloc[-1]
        
        if prev_price > support * 1.01 and current_price < support:
            patterns.append(PatternSignal(
                pattern_name="BREAKDOWN",
                signal_type="BEARISH_CRASH",
                confidence=80.0,
                description="Support breakdown! Price breaking below recent lows. CRASH starting!",
                timeframe=timeframe,
                strength="STRONG"
            ))
        
        # BULL FLAG (continuation pattern - pump continues)
        first_half = recent.head(10)
        second_half = recent.tail(10)
        
        first_half_trend = (first_half['close'].iloc[-1] - first_half['close'].iloc[0]) / first_half['close'].iloc[0]
        second_half_range = (second_half['high'].max() - second_half['low'].min()) / second_half['close'].mean()
        
        if first_half_trend > 0.1 and second_half_range < 0.05:  # Strong rally then consolidation
            patterns.append(PatternSignal(
                pattern_name="BULL_FLAG",
                signal_type="BULLISH_PUMP",
                confidence=70.0,
                description="Bull Flag forming - Healthy consolidation after rally. PUMP continuation likely!",
                timeframe=timeframe,
                strength="MODERATE"
            ))
        
        # STRONG DOWNTREND (consecutive red candles)
        recent_5 = recent.tail(5)
        consecutive_down = sum(1 for i in range(len(recent_5)-1) if recent_5.iloc[i+1]['close'] < recent_5.iloc[i]['close'])
        
        if consecutive_down >= 3:  # 3+ consecutive down candles
            total_drop = (recent_5.iloc[0]['close'] - recent_5.iloc[-1]['close']) / recent_5.iloc[0]['close']
            if total_drop > 0.02:  # >2% drop
                patterns.append(PatternSignal(
                    pattern_name="DOWNTREND",
                    signal_type="BEARISH_CRASH",
                    confidence=70.0 + (consecutive_down * 5),
                    description=f"Strong downtrend - {consecutive_down} consecutive down candles, {total_drop*100:.1f}% drop. CRASH continuing!",
                    timeframe=timeframe,
                    strength="STRONG" if consecutive_down >= 4 else "MODERATE"
                ))
        
        # STRONG UPTREND (consecutive green candles)
        consecutive_up = sum(1 for i in range(len(recent_5)-1) if recent_5.iloc[i+1]['close'] > recent_5.iloc[i]['close'])
        
        if consecutive_up >= 3:  # 3+ consecutive up candles
            total_gain = (recent_5.iloc[-1]['close'] - recent_5.iloc[0]['close']) / recent_5.iloc[0]['close']
            if total_gain > 0.02:  # >2% gain
                patterns.append(PatternSignal(
                    pattern_name="UPTREND",
                    signal_type="BULLISH_PUMP",
                    confidence=70.0 + (consecutive_up * 5),
                    description=f"Strong uptrend - {consecutive_up} consecutive up candles, {total_gain*100:.1f}% gain. PUMP continuing!",
                    timeframe=timeframe,
                    strength="STRONG" if consecutive_up >= 4 else "MODERATE"
                ))
        
        # PARABOLIC MOVE (crash warning - too extended)
        price_changes = recent['close'].pct_change()
        consecutive_up_extended = 0
        for change in price_changes.tail(10):
            if change > 0:
                consecutive_up_extended += 1
            else:
                break
        
        total_move = (recent['close'].iloc[-1] - recent['close'].iloc[-11]) / recent['close'].iloc[-11]
        
        if consecutive_up_extended >= 6 and total_move > 0.3:  # 6+ consecutive up candles, >30% gain (more lenient)
            patterns.append(PatternSignal(
                pattern_name="PARABOLIC_EXHAUSTION",
                signal_type="BEARISH_CRASH",
                confidence=75.0 + min(15, consecutive_up_extended * 2),
                description=f"Parabolic exhaustion - {consecutive_up_extended} consecutive ups, {total_move*100:.0f}% gain. CRASH imminent!",
                timeframe=timeframe,
                strength="STRONG"
            ))
        
        return patterns
    
    def _detect_volume_patterns(self, df: pd.DataFrame, timeframe: str) -> List[PatternSignal]:
        """Detect volume-based patterns"""
        patterns = []
        
        if len(df) < 20 or 'volume' not in df.columns:
            return patterns
        
        recent = df.tail(20).reset_index(drop=True)
        
        # VOLUME SPIKE (unusual activity - pump/crash starting)
        avg_volume = recent['volume'].iloc[:-1].mean()
        current_volume = recent['volume'].iloc[-1]
        price_change = (recent['close'].iloc[-1] - recent['close'].iloc[-2]) / recent['close'].iloc[-2]
        
        if current_volume > avg_volume * 3:  # 3x average volume
            if price_change > 0.05:  # +5% with volume
                patterns.append(PatternSignal(
                    pattern_name="VOLUME_BREAKOUT",
                    signal_type="BULLISH_PUMP",
                    confidence=75.0,
                    description="Massive volume spike with price surge - Big money entering. PUMP confirmed!",
                    timeframe=timeframe,
                    strength="STRONG"
                ))
            elif price_change < -0.05:  # -5% with volume
                patterns.append(PatternSignal(
                    pattern_name="VOLUME_BREAKDOWN",
                    signal_type="BEARISH_CRASH",
                    confidence=75.0,
                    description="Massive volume spike with price drop - Panic selling. CRASH confirmed!",
                    timeframe=timeframe,
                    strength="STRONG"
                ))
        
        # VOLUME DIVERGENCE (warning signal)
        last_5_prices = recent['close'].tail(5)
        last_5_volumes = recent['volume'].tail(5)
        
        price_trending_up = all(last_5_prices.iloc[i] <= last_5_prices.iloc[i+1] 
                               for i in range(len(last_5_prices)-1))
        volume_declining = last_5_volumes.iloc[-1] < last_5_volumes.iloc[0] * 0.5
        
        if price_trending_up and volume_declining:
            patterns.append(PatternSignal(
                pattern_name="BEARISH_DIVERGENCE",
                signal_type="BEARISH_CRASH",
                confidence=65.0,
                description="Bearish divergence - Price rising but volume declining. Weak rally, CRASH likely!",
                timeframe=timeframe,
                strength="MODERATE"
            ))
        
        return patterns
    
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Determine overall trend direction"""
        if len(df) < 5:
            return "NEUTRAL"
        
        first_close = df['close'].iloc[0]
        last_close = df['close'].iloc[-1]
        change = (last_close - first_close) / first_close
        
        if change > 0.1:
            return "UPTREND"
        elif change < -0.1:
            return "DOWNTREND"
        else:
            return "NEUTRAL"
    
    def get_pattern_summary(self, patterns: List[PatternSignal]) -> Dict:
        """
        Get summary of detected patterns.
        
        Returns:
            Dict with pump/crash signals and overall verdict
        """
        if not patterns:
            return {
                'verdict': 'NEUTRAL',
                'confidence': 0,
                'top_pattern': None,
                'pump_signals': [],
                'crash_signals': []
            }
        
        pump_signals = [p for p in patterns if p.signal_type == "BULLISH_PUMP"]
        crash_signals = [p for p in patterns if p.signal_type == "BEARISH_CRASH"]
        
        # Calculate weighted verdict (more lenient - single strong signal counts)
        pump_score = sum(p.confidence for p in pump_signals)
        crash_score = sum(p.confidence for p in crash_signals)
        
        # If we have strong signals (confidence > 70), that's enough
        if pump_score > crash_score and pump_score > 60:  # Lowered from 100 to 60
            verdict = "PUMP_INCOMING"
            confidence = min(100, pump_score / len(pump_signals) if pump_signals else 0)
        elif crash_score > pump_score and crash_score > 60:  # Lowered from 100 to 60
            verdict = "CRASH_INCOMING"
            confidence = min(100, crash_score / len(crash_signals) if crash_signals else 0)
        else:
            verdict = "NEUTRAL"
            confidence = 50
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'top_pattern': patterns[0] if patterns else None,
            'pump_signals': pump_signals,
            'crash_signals': crash_signals
        }
