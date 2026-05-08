"""
Data Feed Module
Handles fetching live OHLCV data from multiple exchanges using ccxt.
Supports auto-fallback and demo mode.
Lightweight caching with Pandas DataFrames.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time


class CryptoDataFeed:
    """Fetch and cache live crypto market data from multiple exchanges with demo mode fallback."""
    
    def __init__(self, max_tokens: int = 100, exchange_name: str = 'auto', demo_mode: bool = False):
        """
        Initialize the data feed.
        
        Args:
            max_tokens: Maximum number of tokens to track (top by volume)
            exchange_name: Exchange to use ('auto', 'binance', 'kucoin', 'coinbase', 'demo')
            demo_mode: Use simulated data instead of live APIs
        """
        self.max_tokens = max_tokens
        self.cache: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}
        self.demo_mode = demo_mode
        self.exchange: Any = None
        self.selected_mode = exchange_name  # Store user's original selection
        self.exchange_name = exchange_name  # Will be updated to actual connected exchange
        
        if demo_mode or exchange_name == 'demo':
            self.demo_mode = True
            self.exchange_name = 'demo'
            self.selected_mode = 'demo'
        else:
            self._init_exchange(exchange_name)
    
    def _init_exchange(self, exchange_name: str = 'auto'):
        """Initialize exchange with automatic fallback."""
        exchanges_to_try = []
        
        if exchange_name == 'auto':
            # Try exchanges in order of reliability for global access
            exchanges_to_try = ['kucoin', 'kraken', 'coinbase', 'binance']
        else:
            exchanges_to_try = [exchange_name]
        
        for exch_name in exchanges_to_try:
            try:
                if exch_name == 'binance':
                    self.exchange = ccxt.binance({
                        'enableRateLimit': True,
                        'options': {'defaultType': 'spot'}
                    })
                elif exch_name == 'kucoin':
                    self.exchange = ccxt.kucoin({
                        'enableRateLimit': True
                    })
                elif exch_name == 'kraken':
                    self.exchange = ccxt.kraken({
                        'enableRateLimit': True
                    })
                elif exch_name == 'coinbase':
                    self.exchange = ccxt.coinbase({
                        'enableRateLimit': True
                    })
                else:
                    continue
                
                # Test the exchange
                self.exchange.load_markets()
                self.exchange_name = exch_name
                print(f"✅ Successfully connected to {exch_name}")
                return
                
            except Exception as e:
                print(f"❌ Failed to connect to {exch_name}: {e}")
                continue
        
        # If all fail, fallback to demo mode
        print("⚠️ All exchanges failed. Switching to demo mode.")
        self.demo_mode = True
        self.exchange_name = 'demo'
    
    def _generate_demo_symbols(self, quote_currency: str = 'USDT') -> List[str]:
        """Generate demo symbol list for testing."""
        base_symbols = [
            'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'MATIC', 
            'DOT', 'AVAX', 'LINK', 'UNI', 'ATOM', 'LTC', 'ETC',
            'FIL', 'APT', 'NEAR', 'ARB', 'OP', 'PEPE', 'SHIB',
            'IMX', 'RNDR', 'INJ', 'TIA', 'SUI', 'SEI', 'ORDI', 'SATS'
        ]
        return [f"{symbol}/{quote_currency}" for symbol in base_symbols[:self.max_tokens]]
    
    def _generate_demo_ohlcv(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Generate realistic demo OHLCV data."""
        # Base price depends on symbol
        if 'BTC' in symbol:
            base_price = 43000 + np.random.uniform(-1000, 1000)
        elif 'ETH' in symbol:
            base_price = 2300 + np.random.uniform(-100, 100)
        elif 'SOL' in symbol:
            base_price = 105 + np.random.uniform(-5, 5)
        else:
            base_price = np.random.uniform(0.1, 50)
        
        # Generate price walk
        timestamps = pd.date_range(end=datetime.now(), periods=limit, freq='5min')
        prices = []
        current_price = base_price
        
        # Create trending or ranging patterns
        trend = np.random.choice(['up', 'down', 'sideways'])
        trend_strength = np.random.uniform(0.0005, 0.003)
        
        for i in range(limit):
            # Add trend
            if trend == 'up':
                current_price *= (1 + trend_strength + np.random.normal(0, 0.01))
            elif trend == 'down':
                current_price *= (1 - trend_strength + np.random.normal(0, 0.01))
            else:
                current_price *= (1 + np.random.normal(0, 0.008))
            
            # Occasional spikes for momentum signals
            if np.random.random() < 0.05:
                spike = np.random.choice([1.02, 0.98])
                current_price *= spike
            
            prices.append(current_price)
        
        # Generate OHLC from prices
        data = []
        for i, ts in enumerate(timestamps):
            price = prices[i]
            volatility = price * 0.01
            
            open_price = price + np.random.normal(0, volatility * 0.5)
            high_price = max(price, open_price) + abs(np.random.normal(0, volatility))
            low_price = min(price, open_price) - abs(np.random.normal(0, volatility))
            close_price = price
            volume = np.random.uniform(1000000, 10000000) * (1 + abs(close_price - open_price) / open_price)
            
            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
        
    def get_top_tokens(self, quote_currency: str = 'USDT') -> List[str]:
        """
        Get top tokens by 24h volume.
        
        Args:
            quote_currency: Quote currency (USDT, BTC, etc.)
            
        Returns:
            List of trading pair symbols
        """
        if self.demo_mode:
            return self._generate_demo_symbols(quote_currency)
        
        try:
            tickers = self.exchange.fetch_tickers()
            
            # Filter for quote currency pairs and sort by volume
            pairs_with_volume = []
            for symbol, ticker in tickers.items():
                if quote_currency in symbol and ticker.get('quoteVolume'):
                    pairs_with_volume.append((symbol, float(ticker['quoteVolume'])))
            
            # Sort by 24h quote volume (descending)
            sorted_pairs = sorted(
                pairs_with_volume, 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Return top N symbols
            top_symbols = [pair[0] for pair in sorted_pairs[:self.max_tokens]]
            return top_symbols
            
        except Exception as e:
            print(f"Error fetching top tokens: {e}")
            print("Switching to demo mode...")
            self.demo_mode = True
            self.exchange_name = 'demo'
            return self._generate_demo_symbols(quote_currency)
    
    def fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = '5m', 
        limit: int = 100,
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol with caching.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, etc.)
            limit: Number of candles to fetch
            force_refresh: Skip cache and fetch fresh data
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{timeframe}"
        
        # Check cache freshness (refresh every minute for live, longer for demo)
        cache_ttl = 300 if self.demo_mode else 60
        if not force_refresh and cache_key in self.cache:
            last_update = self.last_update.get(cache_key)
            if last_update and (datetime.now() - last_update).seconds < cache_ttl:
                return self.cache[cache_key]
        
        # Demo mode
        if self.demo_mode:
            df = self._generate_demo_ohlcv(symbol, limit)
            self.cache[cache_key] = df
            self.last_update[cache_key] = datetime.now()
            return df
        
        try:
            # Fetch from exchange
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Cache it
            self.cache[cache_key] = df
            self.last_update[cache_key] = datetime.now()
            
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def fetch_batch(
        self, 
        symbols: List[str], 
        timeframe: str = '5m', 
        limit: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.
        
        Args:
            symbols: List of trading pairs
            timeframe: Candle timeframe
            limit: Number of candles per symbol
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, timeframe, limit)
            if df is not None and not df.empty:
                results[symbol] = df
            # Small delay to respect rate limits
            time.sleep(0.1)
        
        return results
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the most recent price for a symbol."""
        if self.demo_mode:
            # Return last close from cache if available
            for key in self.cache:
                if symbol in key:
                    return float(self.cache[key]['close'].iloc[-1])
            # Generate random price if not cached
            return float(np.random.uniform(0.1, 50000))
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker.get('last', 0))
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        self.last_update.clear()
