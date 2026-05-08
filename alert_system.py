"""
Alert System Module
Manages signal alerts with thresholds and ranking.
"""

import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from signal_engine import Signal
from dataclasses import asdict


class AlertSystem:
    """Manage and filter trading signals based on thresholds."""
    
    def __init__(self, min_confidence: float = 50.0):
        """
        Initialize alert system.
        
        Args:
            min_confidence: Minimum confidence score to trigger an alert (0-100)
        """
        self.min_confidence = min_confidence
        self.alert_history: List[Signal] = []
        
    def filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Filter signals based on confidence threshold.
        
        Args:
            signals: List of Signal objects
            
        Returns:
            Filtered list of signals above threshold
        """
        return [s for s in signals if s.confidence >= self.min_confidence]
    
    def rank_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Rank signals by confidence and other factors.
        
        Args:
            signals: List of Signal objects
            
        Returns:
            Sorted list of signals
        """
        # Already sorted by confidence in signal_engine, but we can add more logic
        return sorted(signals, key=lambda s: s.confidence, reverse=True)
    
    def add_to_history(self, signals: List[Signal]):
        """Add signals to alert history."""
        self.alert_history.extend(signals)
        
        # Keep only last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
    
    def get_alerts(self, signals: List[Signal]) -> List[Signal]:
        """
        Process signals and return alerts above threshold.
        
        Args:
            signals: List of Signal objects
            
        Returns:
            Filtered and ranked alert signals
        """
        filtered = self.filter_signals(signals)
        ranked = self.rank_signals(filtered)
        self.add_to_history(ranked)
        return ranked
    
    def format_alert(self, signal: Signal) -> str:
        """
        Format a signal into a readable alert message.
        
        Args:
            signal: Signal object
            
        Returns:
            Formatted alert string
        """
        direction = "🚀 BULLISH" if signal.signal_type == 'bullish' else "🔻 BEARISH"
        
        alert = f"""
{direction} | {signal.symbol}
Confidence: {signal.confidence:.1f}%
Price: ${signal.indicators.get('price', 0):.6f}
RSI: {signal.indicators.get('rsi', 0):.1f}
Reason: {signal.reason}
"""
        return alert.strip()
    
    def print_alerts(self, signals: List[Signal]):
        """Print all alerts to console."""
        alerts = self.get_alerts(signals)
        
        if not alerts:
            print("No signals above threshold.")
            return
        
        print(f"\n{'='*60}")
        print(f"CRYPTO MOMENTUM ALERTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        for i, signal in enumerate(alerts, 1):
            print(f"[{i}] {self.format_alert(signal)}\n")
    
    def to_dataframe(self, signals: List[Signal]) -> pd.DataFrame:
        """
        Convert signals to a pandas DataFrame for display.
        
        Args:
            signals: List of Signal objects
            
        Returns:
            DataFrame with signal data
        """
        if not signals:
            return pd.DataFrame()
        
        data = []
        for signal in signals:
            row = {
                'Symbol': signal.symbol.replace('/USDT', ''),
                'Signal': signal.signal_type.upper(),
                'Confidence': f"{signal.confidence:.1f}%",
                'Price': signal.indicators.get('price', 0),
                'RSI': signal.indicators.get('rsi', 0),
                'Volume Ratio': signal.indicators.get('volume_ratio', 0),
                'Reason': signal.reason
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def get_summary_stats(self, signals: List[Signal]) -> Dict:
        """
        Get summary statistics for a list of signals.
        
        Args:
            signals: List of Signal objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not signals:
            return {
                'total_signals': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'avg_confidence': 0,
                'high_confidence_count': 0
            }
        
        bullish = [s for s in signals if s.signal_type == 'bullish']
        bearish = [s for s in signals if s.signal_type == 'bearish']
        high_conf = [s for s in signals if s.confidence >= 70]
        
        avg_conf = sum(s.confidence for s in signals) / len(signals)
        
        return {
            'total_signals': len(signals),
            'bullish_count': len(bullish),
            'bearish_count': len(bearish),
            'avg_confidence': avg_conf,
            'high_confidence_count': len(high_conf)
        }
