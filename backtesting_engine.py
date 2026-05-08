"""
Backtesting Engine Module
Tracks historical signal accuracy and adjusts confidence scores based on performance.
Stores signal outcomes and calculates performance metrics for individual indicators
and indicator combinations.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
from ml_predictor import MLPredictor


@dataclass
class SignalOutcome:
    """Represents the outcome of a historical signal."""
    symbol: str
    signal_type: str  # 'bullish' or 'bearish'
    entry_price: float
    entry_time: datetime
    indicators_present: List[str]  # Which indicators triggered
    original_confidence: float
    
    # Outcome data (filled after waiting period)
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    was_successful: Optional[bool] = None
    price_change_pct: Optional[float] = None
    max_favorable_move: Optional[float] = None  # Best move in predicted direction
    max_adverse_move: Optional[float] = None    # Worst move against prediction


@dataclass
class IndicatorPerformance:
    """Performance metrics for an indicator or indicator combination."""
    indicator_name: str
    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    success_rate: float = 0.0
    avg_gain_on_success: float = 0.0
    avg_loss_on_failure: float = 0.0
    avg_time_to_target: float = 0.0  # Hours
    confidence_multiplier: float = 1.0  # Adjustment factor based on performance
    last_updated: Optional[datetime] = None


class BacktestingEngine:
    """
    Backtesting engine that tracks signal performance and adjusts confidence scores.
    """
    
    def __init__(self, 
                 data_file: str = "backtest_data.json",
                 success_threshold_pct: float = 2.0,
                 evaluation_period_hours: int = 24,
                 min_samples_for_adjustment: int = 10):
        """
        Initialize the backtesting engine.
        
        Args:
            data_file: Path to JSON file for storing historical data
            success_threshold_pct: Price move % to consider signal successful
            evaluation_period_hours: How long to track each signal
            min_samples_for_adjustment: Minimum signals needed before adjusting confidence
        """
        self.data_file = Path(data_file)
        self.success_threshold_pct = success_threshold_pct
        self.evaluation_period_hours = evaluation_period_hours
        self.min_samples_for_adjustment = min_samples_for_adjustment
        
        # In-memory storage
        self.pending_outcomes: List[SignalOutcome] = []
        self.completed_outcomes: List[SignalOutcome] = []
        self.indicator_performance: Dict[str, IndicatorPerformance] = {}
        
        # ML predictor for continuous learning
        self.ml_predictor = MLPredictor()
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load historical backtesting data from file."""
        if not self.data_file.exists():
            return
        
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            
            # Load completed outcomes
            for outcome_data in data.get('completed_outcomes', []):
                outcome_data['entry_time'] = datetime.fromisoformat(outcome_data['entry_time'])
                if outcome_data.get('exit_time'):
                    outcome_data['exit_time'] = datetime.fromisoformat(outcome_data['exit_time'])
                self.completed_outcomes.append(SignalOutcome(**outcome_data))
            
            # Load pending outcomes
            for outcome_data in data.get('pending_outcomes', []):
                outcome_data['entry_time'] = datetime.fromisoformat(outcome_data['entry_time'])
                if outcome_data.get('exit_time'):
                    outcome_data['exit_time'] = datetime.fromisoformat(outcome_data['exit_time'])
                self.pending_outcomes.append(SignalOutcome(**outcome_data))
            
            # Load indicator performance
            for ind_name, perf_data in data.get('indicator_performance', {}).items():
                if perf_data.get('last_updated'):
                    perf_data['last_updated'] = datetime.fromisoformat(perf_data['last_updated'])
                self.indicator_performance[ind_name] = IndicatorPerformance(**perf_data)
                
        except Exception as e:
            print(f"Warning: Could not load backtesting data: {e}")
    
    def _save_data(self):
        """Save backtesting data to file."""
        try:
            data = {
                'completed_outcomes': [
                    {**asdict(outcome), 
                     'entry_time': outcome.entry_time.isoformat(),
                     'exit_time': outcome.exit_time.isoformat() if outcome.exit_time else None}
                    for outcome in self.completed_outcomes
                ],
                'pending_outcomes': [
                    {**asdict(outcome),
                     'entry_time': outcome.entry_time.isoformat(),
                     'exit_time': outcome.exit_time.isoformat() if outcome.exit_time else None}
                    for outcome in self.pending_outcomes
                ],
                'indicator_performance': {
                    name: {**asdict(perf),
                           'last_updated': perf.last_updated.isoformat() if perf.last_updated else None}
                    for name, perf in self.indicator_performance.items()
                }
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save backtesting data: {e}")
    
    def _extract_indicators_from_reason(self, reason: str) -> List[str]:
        """
        Extract which indicators triggered from the signal reason string.
        
        Args:
            reason: Signal reason string like "RSI oversold | MACD bullish crossover"
            
        Returns:
            List of indicator names
        """
        indicators = []
        reason_lower = reason.lower()
        
        # Map keywords to indicator names
        indicator_keywords = {
            'rsi': 'RSI',
            'macd': 'MACD',
            'volume spike': 'Volume',
            'bollinger': 'BollingerBands',
            'bb': 'BollingerBands',
            'stochastic': 'Stochastic',
            'price anomaly': 'PriceAnomaly',
            'z=': 'PriceAnomaly'
        }
        
        for keyword, indicator_name in indicator_keywords.items():
            if keyword in reason_lower:
                if indicator_name not in indicators:
                    indicators.append(indicator_name)
        
        return indicators
    
    def record_signal(self, symbol: str, signal_type: str, entry_price: float, 
                     confidence: float, reason: str):
        """
        Record a new signal for future evaluation.
        
        Args:
            symbol: Trading pair symbol
            signal_type: 'bullish' or 'bearish'
            entry_price: Current price when signal generated
            confidence: Original confidence score
            reason: Signal reason string
        """
        indicators = self._extract_indicators_from_reason(reason)
        
        outcome = SignalOutcome(
            symbol=symbol,
            signal_type=signal_type,
            entry_price=entry_price,
            entry_time=datetime.now(),
            indicators_present=indicators,
            original_confidence=confidence
        )
        
        self.pending_outcomes.append(outcome)
        self._save_data()
    
    def update_outcomes(self, current_prices: Dict[str, float]):
        """
        Update pending outcomes with current prices and evaluate them.
        
        Args:
            current_prices: Dict mapping symbols to current prices
        """
        now = datetime.now()
        newly_completed = []
        
        for outcome in self.pending_outcomes[:]:  # Copy to allow modification
            # Check if evaluation period has passed
            time_elapsed = now - outcome.entry_time
            hours_elapsed = time_elapsed.total_seconds() / 3600
            
            if hours_elapsed >= self.evaluation_period_hours:
                # Get current price
                current_price = current_prices.get(outcome.symbol)
                
                if current_price is not None:
                    # Calculate outcome
                    price_change_pct = ((current_price - outcome.entry_price) / outcome.entry_price) * 100
                    
                    # Determine success based on signal direction
                    if outcome.signal_type == 'bullish':
                        was_successful = price_change_pct >= self.success_threshold_pct
                    else:  # bearish
                        was_successful = price_change_pct <= -self.success_threshold_pct
                    
                    # Update outcome
                    outcome.exit_price = current_price
                    outcome.exit_time = now
                    outcome.was_successful = was_successful
                    outcome.price_change_pct = price_change_pct
                    
                    # Move to completed
                    self.pending_outcomes.remove(outcome)
                    self.completed_outcomes.append(outcome)
                    newly_completed.append(outcome)
                    
                    # ML CONTINUOUS LEARNING: Feed outcome to ML model
                    try:
                        # Determine ML outcome classification
                        if was_successful:
                            if outcome.signal_type == 'bullish':
                                ml_outcome = 'BREAKOUT'
                            else:
                                ml_outcome = 'BREAKDOWN'
                        else:
                            ml_outcome = 'NEUTRAL'
                        
                        # Extract features from the signal (approximate values)
                        ml_features = {
                            'rsi': 50.0,  # Could be extracted from indicators_present
                            'macd': 0.0,
                            'bb_position': 0.5,
                            'volume_ratio': 1.0,
                            'price_change_1h': price_change_pct / 24 if hours_elapsed >= 24 else price_change_pct,
                            'price_change_24h': price_change_pct if hours_elapsed >= 24 else price_change_pct * 2,
                            'price_change_7d': 0.0,
                            'volatility': abs(price_change_pct) / 100,
                            'trend_strength': 1.0 if was_successful else -1.0,
                            'volume_trend': 0.5,
                        }
                        
                        # Learn from this outcome
                        self.ml_predictor.learn_from_outcome(ml_features, ml_outcome)
                        print(f"📚 ML learned from {outcome.symbol} signal: {ml_outcome} (price changed {price_change_pct:.2f}%)")
                    
                    except Exception as ml_error:
                        print(f"ML learning error: {ml_error}")
        
        # Recalculate performance metrics if we have new completions
        if newly_completed:
            self._recalculate_performance_metrics()
            self._save_data()
    
    def _recalculate_performance_metrics(self):
        """Recalculate performance metrics for all indicators and combinations."""
        # Reset metrics
        self.indicator_performance.clear()
        
        # Group outcomes by indicator
        indicator_outcomes: Dict[str, List[SignalOutcome]] = {}
        
        for outcome in self.completed_outcomes:
            if outcome.was_successful is None:
                continue
            
            # Track individual indicators
            for indicator in outcome.indicators_present:
                if indicator not in indicator_outcomes:
                    indicator_outcomes[indicator] = []
                indicator_outcomes[indicator].append(outcome)
            
            # Track indicator combinations (only pairs to avoid explosion)
            if len(outcome.indicators_present) >= 2:
                # Sort to ensure consistent combination names
                sorted_indicators = sorted(outcome.indicators_present)
                for i in range(len(sorted_indicators)):
                    for j in range(i + 1, len(sorted_indicators)):
                        combo_name = f"{sorted_indicators[i]}+{sorted_indicators[j]}"
                        if combo_name not in indicator_outcomes:
                            indicator_outcomes[combo_name] = []
                        indicator_outcomes[combo_name].append(outcome)
        
        # Calculate metrics for each indicator/combination
        for indicator_name, outcomes in indicator_outcomes.items():
            perf = IndicatorPerformance(indicator_name=indicator_name)
            
            perf.total_signals = len(outcomes)
            perf.successful_signals = sum(1 for o in outcomes if o.was_successful)
            perf.failed_signals = perf.total_signals - perf.successful_signals
            
            if perf.total_signals > 0:
                perf.success_rate = (perf.successful_signals / perf.total_signals) * 100
            
            # Calculate average gains/losses
            successful = [o for o in outcomes if o.was_successful and o.price_change_pct is not None]
            failed = [o for o in outcomes if not o.was_successful and o.price_change_pct is not None]
            
            if successful:
                perf.avg_gain_on_success = float(np.mean([abs(o.price_change_pct) for o in successful]))
            
            if failed:
                perf.avg_loss_on_failure = float(np.mean([abs(o.price_change_pct) for o in failed]))
            
            # Calculate confidence multiplier based on performance
            # Only apply if we have enough samples
            if perf.total_signals >= self.min_samples_for_adjustment:
                # Base multiplier on success rate
                if perf.success_rate >= 70:
                    perf.confidence_multiplier = 1.3  # Boost confidence by 30%
                elif perf.success_rate >= 60:
                    perf.confidence_multiplier = 1.15
                elif perf.success_rate >= 50:
                    perf.confidence_multiplier = 1.0  # No change
                elif perf.success_rate >= 40:
                    perf.confidence_multiplier = 0.85
                else:
                    perf.confidence_multiplier = 0.7  # Reduce confidence by 30%
            else:
                perf.confidence_multiplier = 1.0  # Not enough data yet
            
            perf.last_updated = datetime.now()
            self.indicator_performance[indicator_name] = perf
    
    def get_adjusted_confidence(self, original_confidence: float, 
                               indicators_present: List[str]) -> float:
        """
        Calculate adjusted confidence based on historical performance.
        
        Args:
            original_confidence: Original confidence score
            indicators_present: List of indicators that triggered
            
        Returns:
            Adjusted confidence score
        """
        if not indicators_present:
            return original_confidence
        
        # Collect multipliers for all relevant indicators
        multipliers = []
        
        # Check individual indicators
        for indicator in indicators_present:
            if indicator in self.indicator_performance:
                perf = self.indicator_performance[indicator]
                if perf.total_signals >= self.min_samples_for_adjustment:
                    multipliers.append(perf.confidence_multiplier)
        
        # Check indicator combinations (pairs)
        if len(indicators_present) >= 2:
            sorted_indicators = sorted(indicators_present)
            for i in range(len(sorted_indicators)):
                for j in range(i + 1, len(sorted_indicators)):
                    combo_name = f"{sorted_indicators[i]}+{sorted_indicators[j]}"
                    if combo_name in self.indicator_performance:
                        perf = self.indicator_performance[combo_name]
                        if perf.total_signals >= self.min_samples_for_adjustment:
                            # Weight combination performance more heavily
                            multipliers.append(perf.confidence_multiplier * 1.2)
        
        # If no historical data, return original
        if not multipliers:
            return original_confidence
        
        # Use weighted average of multipliers (prefer combination data)
        avg_multiplier = np.mean(multipliers)
        
        # Apply multiplier with bounds
        adjusted = original_confidence * avg_multiplier
        adjusted = float(max(0, min(100, adjusted)))  # Keep in 0-100 range
        
        return adjusted
    
    def get_performance_summary(self) -> Dict:
        """
        Get a summary of backtesting performance.
        
        Returns:
            Dictionary with performance statistics
        """
        total_completed = len(self.completed_outcomes)
        total_pending = len(self.pending_outcomes)
        
        if total_completed == 0:
            return {
                'total_completed': 0,
                'total_pending': total_pending,
                'overall_success_rate': 0,
                'total_indicators_tracked': 0,
                'best_performing_indicator': None,
                'worst_performing_indicator': None,
                'evaluation_period_hours': self.evaluation_period_hours,
                'success_threshold_pct': self.success_threshold_pct
            }
        
        successful = sum(1 for o in self.completed_outcomes if o.was_successful)
        overall_success_rate = (successful / total_completed) * 100
        
        # Find best and worst performing indicators (with enough samples)
        qualified_indicators = {
            name: perf for name, perf in self.indicator_performance.items()
            if perf.total_signals >= self.min_samples_for_adjustment and '+' not in name
        }
        
        best_indicator = None
        worst_indicator = None
        
        if qualified_indicators:
            best_indicator = max(qualified_indicators.items(), 
                               key=lambda x: x[1].success_rate)
            worst_indicator = min(qualified_indicators.items(), 
                                key=lambda x: x[1].success_rate)
        
        return {
            'total_completed': total_completed,
            'total_pending': total_pending,
            'overall_success_rate': overall_success_rate,
            'total_indicators_tracked': len(self.indicator_performance),
            'best_performing_indicator': best_indicator,
            'worst_performing_indicator': worst_indicator,
            'evaluation_period_hours': self.evaluation_period_hours,
            'success_threshold_pct': self.success_threshold_pct
        }
    
    def get_indicator_stats(self, min_samples: int = 5) -> pd.DataFrame:
        """
        Get performance statistics for all indicators as a DataFrame.
        
        Args:
            min_samples: Minimum samples to include in results
            
        Returns:
            DataFrame with indicator performance metrics
        """
        data = []
        
        for name, perf in self.indicator_performance.items():
            if perf.total_signals >= min_samples:
                data.append({
                    'Indicator': name,
                    'Total Signals': perf.total_signals,
                    'Success Rate %': round(perf.success_rate, 1),
                    'Avg Gain %': round(perf.avg_gain_on_success, 2),
                    'Avg Loss %': round(perf.avg_loss_on_failure, 2),
                    'Confidence Multiplier': round(perf.confidence_multiplier, 2)
                })
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df = df.sort_values('Success Rate %', ascending=False)
        return df
