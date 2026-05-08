"""
Phantom Wallet Trending Token Analyzer

Integrates with CoinGecko API (same data source as Phantom wallet) to:
- Get trending Solana tokens from Phantom's data feed
- Analyze 24-hour and 1-hour charts
- Identify tokens poised to explode (explosive patterns)
- Detect already-pumped tokens on the downside
- Classify momentum: EXPLOSIVE, CONSOLIDATING, or DECLINING
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pattern_detector import PatternDetector, PatternSignal
from ml_predictor import MLPredictor


@dataclass
class PhantomToken:
    """Represents a trending token from Phantom wallet with momentum analysis"""
    symbol: str
    name: str
    coingecko_id: str
    price_usd: float
    market_cap: float
    volume_24h: float
    price_change_1h: float
    price_change_24h: float
    price_change_7d: float
    
    # Momentum classification
    momentum_signal: str  # EXPLOSIVE, CONSOLIDATING, DECLINING
    momentum_score: float  # 0-100
    
    # Chart analysis
    volume_trend: str  # INCREASING, STABLE, DECREASING
    price_pattern: str  # BREAKOUT, PARABOLIC_TOP, CONSOLIDATION, DOWNTREND
    
    # Explosive vs Pumped indicators
    is_explosive: bool  # Ready to pump
    is_already_pumped: bool  # Already topped out
    
    # Detailed metrics
    volume_acceleration: float  # 1h volume vs 24h average
    rsi_1h: Optional[float] = None
    rsi_24h: Optional[float] = None
    
    # Candlestick & Price Action Patterns
    candlestick_patterns: List[PatternSignal] = field(default_factory=list)
    pattern_verdict: str = ""  # PUMP_INCOMING, CRASH_INCOMING, NEUTRAL
    pattern_confidence: float = 0.0
    
    # Reasoning
    reason: str = ""
    chart_url: str = ""


class PhantomTrendingAnalyzer:
    """
    Analyzes trending tokens from Phantom wallet (via CoinGecko API).
    Identifies explosive opportunities vs already-pumped tokens.
    """
    
    def __init__(self):
        # CoinGecko API (free tier: 10-30 calls/min)
        self.coingecko_api_key = os.getenv("COINGECKO_API_KEY")
        
        if self.coingecko_api_key:
            self.base_url = "https://pro-api.coingecko.com/api/v3"
        else:
            self.base_url = "https://api.coingecko.com/api/v3"
        
        # Advanced pattern detector
        self.pattern_detector = PatternDetector()
        
        # ML predictor for breakout/breakdown detection
        self.ml_predictor = MLPredictor()
        
        # Initialize ML model with synthetic data if not already trained
        if not self.ml_predictor.is_trained:
            print("🤖 Initializing ML model with synthetic training data...")
            self.ml_predictor.create_synthetic_training_data()
        
        # Cache
        self.cache = {}
        self.cache_timeout = 60  # 60 seconds
    
    def get_phantom_trending_tokens(self, limit: int = 20) -> List[PhantomToken]:
        """
        Get trending Solana tokens (same as Phantom wallet displays).
        
        Args:
            limit: Number of trending tokens to analyze
            
        Returns:
            List of PhantomToken objects with momentum analysis
        """
        cache_key = f"phantom_trending_{limit}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_timeout:
                return cached_data
        
        try:
            # Get trending Solana ecosystem tokens from CoinGecko
            trending_tokens = self._fetch_trending_solana_tokens(limit * 2)
            
            if not trending_tokens:
                return []
            
            phantom_tokens = []
            
            for token_data in trending_tokens[:limit]:
                try:
                    # Analyze charts and momentum
                    momentum_analysis = self._analyze_token_momentum(token_data)
                    
                    if not momentum_analysis:
                        continue
                    
                    phantom_token = PhantomToken(
                        symbol=token_data['symbol'].upper(),
                        name=token_data['name'],
                        coingecko_id=token_data['id'],
                        price_usd=token_data['current_price'],
                        market_cap=token_data.get('market_cap', 0),
                        volume_24h=token_data.get('total_volume', 0),
                        price_change_1h=token_data.get('price_change_percentage_1h_in_currency', 0),
                        price_change_24h=token_data.get('price_change_percentage_24h', 0),
                        price_change_7d=token_data.get('price_change_percentage_7d_in_currency', 0),
                        **momentum_analysis
                    )
                    
                    phantom_tokens.append(phantom_token)
                    
                except Exception as e:
                    print(f"Error analyzing token {token_data.get('symbol', 'UNKNOWN')}: {e}")
                    continue
            
            # Sort by momentum score (highest first)
            phantom_tokens.sort(key=lambda x: x.momentum_score, reverse=True)
            
            # Cache results
            self.cache[cache_key] = (time.time(), phantom_tokens)
            
            return phantom_tokens
            
        except Exception as e:
            print(f"Error fetching Phantom trending tokens: {e}")
            return []
    
    def _fetch_trending_solana_tokens(self, limit: int) -> List[Dict]:
        """Fetch trending Solana ecosystem tokens from CoinGecko"""
        try:
            params = {
                'vs_currency': 'usd',
                'category': 'solana-ecosystem',
                'order': 'volume_desc',
                'per_page': limit,
                'page': 1,
                'sparkline': False,
                'price_change_percentage': '1h,24h,7d'
            }
            
            if self.coingecko_api_key:
                params['x_cg_pro_api_key'] = self.coingecko_api_key
            
            response = requests.get(
                f"{self.base_url}/coins/markets",
                params=params,
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"CoinGecko API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching from CoinGecko: {e}")
            return []
    
    def _analyze_token_momentum(self, token_data: Dict) -> Optional[Dict]:
        """
        Analyze token momentum using 1h and 24h price changes + advanced candlestick patterns.
        
        Returns:
            Dict with momentum analysis or None if data insufficient
        """
        try:
            # Extract price changes
            price_1h = token_data.get('price_change_percentage_1h_in_currency', 0) or 0
            price_24h = token_data.get('price_change_percentage_24h', 0) or 0
            price_7d = token_data.get('price_change_percentage_7d_in_currency', 0) or 0
            
            volume_24h = token_data.get('total_volume', 0)
            market_cap = token_data.get('market_cap', 1)
            coingecko_id = token_data.get('id', '')
            
            # Calculate volume-to-market-cap ratio
            vol_to_mcap = (volume_24h / market_cap) if market_cap > 0 else 0
            
            # Analyze momentum pattern (basic)
            momentum_signal, momentum_score, is_explosive, is_pumped = self._classify_momentum(
                price_1h, price_24h, price_7d, vol_to_mcap
            )
            
            # Determine volume trend
            volume_trend = self._analyze_volume_trend(vol_to_mcap)
            
            # Advanced candlestick pattern detection
            candlestick_patterns = []
            pattern_verdict = "NEUTRAL"
            pattern_confidence = 0.0
            price_pattern = "UNCLEAR"
            
            try:
                # Detect advanced patterns
                print(f"Analyzing patterns for {coingecko_id}...")
                detected_patterns = self.pattern_detector.analyze_token_patterns(coingecko_id)
                candlestick_patterns = detected_patterns
                
                if detected_patterns:
                    print(f"✓ Found {len(detected_patterns)} patterns for {coingecko_id}")
                    # Get pattern summary
                    pattern_summary = self.pattern_detector.get_pattern_summary(detected_patterns)
                    pattern_verdict = pattern_summary['verdict']
                    pattern_confidence = pattern_summary['confidence']
                    
                    print(f"  Verdict: {pattern_verdict} ({pattern_confidence:.0f}% confidence)")
                    
                    # Use top pattern for price_pattern
                    if pattern_summary['top_pattern']:
                        price_pattern = pattern_summary['top_pattern'].pattern_name
                        print(f"  Top pattern: {price_pattern}")
                    
                    # Adjust momentum score based on candlestick patterns
                    if pattern_verdict == "PUMP_INCOMING":
                        momentum_score = min(100, momentum_score + pattern_confidence * 0.3)
                        is_explosive = True
                    elif pattern_verdict == "CRASH_INCOMING":
                        momentum_score = max(0, momentum_score - pattern_confidence * 0.3)
                        is_pumped = True
                else:
                    print(f"⚠ No patterns detected for {coingecko_id} - using basic pattern")
                    price_pattern = self._identify_price_pattern(price_1h, price_24h, price_7d)
            
            except Exception as pattern_error:
                # If pattern detection fails, fall back to basic pattern
                print(f"❌ Pattern detection error for {coingecko_id}: {pattern_error}")
                price_pattern = self._identify_price_pattern(price_1h, price_24h, price_7d)
            
            # Apply ML prediction for additional intelligence
            try:
                ml_features = {
                    'rsi': 50.0,  # Placeholder - could calculate from OHLC data
                    'macd': 0.0,
                    'bb_position': 0.5,
                    'volume_ratio': vol_to_mcap * 10,  # Normalize to reasonable range
                    'price_change_1h': price_1h,
                    'price_change_24h': price_24h,
                    'price_change_7d': price_7d,
                    'volatility': abs(price_1h - price_24h/24) / 100 if price_24h != 0 else 0,
                    'trend_strength': momentum_score / 100,
                    'volume_trend': vol_to_mcap,
                }
                
                ml_result = self.ml_predictor.predict_breakout_probability(ml_features)
                
                if ml_result['ml_enabled']:
                    print(f"🤖 ML predicts: {ml_result['prediction']} ({ml_result['confidence']:.0f}%)")
                    
                    # Adjust momentum score based on ML prediction
                    if ml_result['prediction'] == 'BREAKOUT':
                        boost = ml_result['confidence'] * 0.25  # Up to +25 points
                        momentum_score = min(100, momentum_score + boost)
                        is_explosive = True
                        print(f"  → Boosting score by {boost:.1f} points (ML confirms BREAKOUT)")
                    elif ml_result['prediction'] == 'BREAKDOWN':
                        penalty = ml_result['confidence'] * 0.25  # Up to -25 points
                        momentum_score = max(0, momentum_score - penalty)
                        is_pumped = True
                        print(f"  → Reducing score by {penalty:.1f} points (ML predicts BREAKDOWN)")
            
            except Exception as ml_error:
                print(f"ML prediction error: {ml_error}")
            
            # Reclassify signal based on adjusted score
            if momentum_score >= 70:
                momentum_signal = "EXPLOSIVE"
            elif momentum_score >= 40:
                momentum_signal = "CONSOLIDATING"
            else:
                momentum_signal = "DECLINING"
            
            # Generate reasoning
            reason = self._generate_momentum_reasoning(
                momentum_signal, price_1h, price_24h, price_7d, vol_to_mcap, price_pattern,
                candlestick_patterns
            )
            
            # Chart URL
            chart_url = f"https://www.coingecko.com/en/coins/{token_data['id']}"
            
            return {
                'momentum_signal': momentum_signal,
                'momentum_score': momentum_score,
                'volume_trend': volume_trend,
                'price_pattern': price_pattern,
                'is_explosive': is_explosive,
                'is_already_pumped': is_pumped,
                'volume_acceleration': vol_to_mcap * 100,  # As percentage
                'candlestick_patterns': candlestick_patterns,
                'pattern_verdict': pattern_verdict,
                'pattern_confidence': pattern_confidence,
                'reason': reason,
                'chart_url': chart_url
            }
            
        except Exception as e:
            print(f"Error in momentum analysis: {e}")
            return None
    
    def _classify_momentum(
        self, 
        price_1h: float, 
        price_24h: float, 
        price_7d: float,
        vol_to_mcap: float
    ) -> Tuple[str, float, bool, bool]:
        """
        Classify token momentum as EXPLOSIVE, CONSOLIDATING, or DECLINING.
        
        Returns:
            Tuple of (signal, score, is_explosive, is_already_pumped)
        """
        score = 50.0  # Base score
        is_explosive = False
        is_pumped = False
        
        # EXPLOSIVE PATTERN DETECTION (tokens poised to explode)
        # Criteria: Early momentum + increasing volume + not overextended
        
        # Strong 1h momentum (early stage pump starting)
        if price_1h > 5 and price_24h < 50:  # Up >5% in 1h but <50% in 24h
            score += 15
            is_explosive = True
        
        # Accelerating momentum (1h gain > 24h average)
        if price_1h > 0 and price_24h > 0:
            hourly_avg_24h = price_24h / 24
            if price_1h > hourly_avg_24h * 2:  # 1h gain > 2x average
                score += 20
                is_explosive = True
        
        # High volume but not parabolic price yet
        if vol_to_mcap > 0.3 and price_24h < 100:  # High volume, room to run
            score += 15
            is_explosive = True
        
        # Fresh breakout (positive across timeframes)
        if price_1h > 0 and price_24h > 0 and price_7d > 0:
            score += 10
        
        # ALREADY PUMPED DETECTION (tokens on the downside)
        # Criteria: Parabolic gains + declining momentum + volume exhaustion
        
        # Parabolic 24h move (likely topped)
        if price_24h > 100:  # >100% in 24h = parabolic
            score -= 20
            is_pumped = True
        
        # Negative 1h divergence (topping signal)
        if price_1h < 0 and price_24h > 50:  # Falling short-term after big run
            score -= 25
            is_pumped = True
        
        # Extreme 7d run without follow-through
        if price_7d > 200 and price_1h < 5:  # Massive run but momentum fading
            score -= 15
            is_pumped = True
        
        # Volume declining (exhaustion)
        if vol_to_mcap < 0.1:
            score -= 10
        
        # CONSOLIDATION PATTERNS (neutral/watching)
        if abs(price_1h) < 2 and abs(price_24h) < 10:
            score = 50  # Neutral
        
        # Downtrend
        if price_1h < -5 and price_24h < -10:
            score = max(10, score - 30)
        
        # Classify signal
        if score >= 70:
            signal = "EXPLOSIVE"
            is_explosive = True
        elif score >= 40:
            signal = "CONSOLIDATING"
        else:
            signal = "DECLINING"
        
        return signal, min(100, max(0, score)), is_explosive, is_pumped
    
    def _analyze_volume_trend(self, vol_to_mcap: float) -> str:
        """Analyze volume trend"""
        if vol_to_mcap > 0.5:
            return "INCREASING"
        elif vol_to_mcap > 0.2:
            return "STABLE"
        else:
            return "DECREASING"
    
    def _identify_price_pattern(self, price_1h: float, price_24h: float, price_7d: float) -> str:
        """Identify chart pattern"""
        # Breakout pattern
        if price_1h > 3 and price_24h > 10 and price_24h < 50:
            return "BREAKOUT"
        
        # Parabolic top (already pumped)
        if price_24h > 100 or (price_7d > 200 and price_1h < 0):
            return "PARABOLIC_TOP"
        
        # Downtrend
        if price_1h < -5 and price_24h < -10:
            return "DOWNTREND"
        
        # Consolidation
        if abs(price_1h) < 2 and abs(price_24h) < 10:
            return "CONSOLIDATION"
        
        return "UNCLEAR"
    
    def _generate_momentum_reasoning(
        self,
        signal: str,
        price_1h: float,
        price_24h: float,
        price_7d: float,
        vol_to_mcap: float,
        pattern: str,
        candlestick_patterns: List[PatternSignal] = None
    ) -> str:
        """Generate human-readable reasoning for momentum classification"""
        reasons = []
        
        # Add candlestick pattern insights first (highest priority)
        if candlestick_patterns:
            pump_patterns = [p for p in candlestick_patterns if p.signal_type == "BULLISH_PUMP" and p.strength == "STRONG"]
            crash_patterns = [p for p in candlestick_patterns if p.signal_type == "BEARISH_CRASH" and p.strength == "STRONG"]
            
            if pump_patterns:
                top_pump = pump_patterns[0]
                reasons.append(f"🚀 {top_pump.pattern_name}: {top_pump.description.split('.')[0]}")
            
            if crash_patterns:
                top_crash = crash_patterns[0]
                reasons.append(f"📉 {top_crash.pattern_name}: {top_crash.description.split('.')[0]}")
        
        if signal == "EXPLOSIVE":
            if price_1h > 5:
                reasons.append(f"Strong 1h momentum: +{price_1h:.1f}%")
            if vol_to_mcap > 0.3:
                reasons.append(f"High volume: {vol_to_mcap*100:.0f}% of market cap")
            if pattern == "BREAKOUT" or pattern == "BULLISH_ENGULFING":
                reasons.append(f"{pattern.replace('_', ' ').title()} pattern detected")
            if price_24h > 0 and price_24h < 50:
                reasons.append("Early-stage move with room to run")
        
        elif signal == "DECLINING":
            if price_24h > 100:
                reasons.append(f"Parabolic run: +{price_24h:.0f}% in 24h (likely topped)")
            if price_1h < 0 and price_24h > 50:
                reasons.append(f"Negative divergence: -{abs(price_1h):.1f}% in 1h after big run")
            if pattern == "PARABOLIC_TOP" or pattern == "EVENING_STAR":
                reasons.append(f"{pattern.replace('_', ' ').title()} - already pumped")
            if vol_to_mcap < 0.1:
                reasons.append("Volume declining - momentum exhausted")
        
        else:  # CONSOLIDATING
            if abs(price_1h) < 2:
                reasons.append(f"Low 1h volatility: {price_1h:+.1f}%")
            if pattern == "CONSOLIDATION":
                reasons.append("Consolidating sideways - watching for breakout")
            elif pattern and pattern != "UNCLEAR":
                reasons.append(f"{pattern.replace('_', ' ').title()} pattern forming")
        
        return " | ".join(reasons) if reasons else "Analyzing momentum..."


def get_phantom_analyzer() -> PhantomTrendingAnalyzer:
    """Get singleton instance of Phantom trending analyzer"""
    return PhantomTrendingAnalyzer()
