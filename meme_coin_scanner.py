"""
Solana Meme Coin Scanner
Discovers trending meme coins with 100x+ potential using internet trends and social signals.
"""

import requests
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time
from social_sentiment import SocialSentimentAnalyzer, SocialMetrics


@dataclass
class MemeCoin:
    """Represents a trending meme coin with potential metrics."""
    symbol: str
    name: str
    address: str
    price_usd: float
    market_cap: float
    volume_24h: float
    liquidity_usd: float
    price_change_24h: float
    created_timestamp: Optional[int]
    holder_count: Optional[int]
    social_score: float
    potential_score: float
    reason: str
    dex_url: str
    
    # Detailed social metrics
    social_metrics: Optional[SocialMetrics] = None


class MemeCoinScanner:
    """
    Scans for trending Solana meme coins with high growth potential.
    Combines on-chain metrics with social signals to identify 100x opportunities.
    """
    
    def __init__(self):
        self.dexscreener_base = "https://api.dexscreener.com/latest/dex"
        self.cache = {}
        self.cache_timeout = 60
        self.social_analyzer = SocialSentimentAnalyzer()
        
    def get_trending_solana_tokens(self, limit: int = 50) -> List[MemeCoin]:
        """
        Get trending Solana tokens from DexScreener.
        
        Args:
            limit: Maximum number of tokens to return
            
        Returns:
            List of MemeCoin objects sorted by potential score
        """
        try:
            cache_key = f"trending_{limit}"
            if cache_key in self.cache:
                cached_time, cached_data = self.cache[cache_key]
                if time.time() - cached_time < self.cache_timeout:
                    return cached_data
            
            url = f"{self.dexscreener_base}/search/?q=solana"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            pairs = data.get('pairs', []) if data else []
            if not pairs:
                return []
            
            meme_coins = []
            seen_addresses = set()
            
            for pair in pairs[:limit * 3]:
                try:
                    if pair['chainId'] != 'solana':
                        continue
                    
                    base_token = pair.get('baseToken', {})
                    token_address = base_token.get('address', '')
                    
                    if not token_address or token_address in seen_addresses:
                        continue
                    
                    seen_addresses.add(token_address)
                    
                    price_usd = float(pair.get('priceUsd', 0))
                    if price_usd == 0:
                        continue
                    
                    fdv = pair.get('fdv', 0)
                    volume_24h = float(pair.get('volume', {}).get('h24', 0))
                    liquidity = float(pair.get('liquidity', {}).get('usd', 0))
                    price_change_24h = float(pair.get('priceChange', {}).get('h24', 0))
                    
                    txns_24h = pair.get('txns', {}).get('h24', {})
                    buys_24h = txns_24h.get('buys', 0)
                    sells_24h = txns_24h.get('sells', 0)
                    
                    pair_created_at = pair.get('pairCreatedAt')
                    
                    symbol = base_token.get('symbol', 'UNKNOWN')
                    name = base_token.get('name', 'Unknown')
                    
                    # Fetch social metrics for the token
                    social_metrics = self.social_analyzer.get_social_metrics(
                        token_address=token_address,
                        symbol=symbol,
                        name=name
                    )
                    
                    # Calculate base technical potential score
                    technical_score, technical_reason = self._calculate_potential_score(
                        fdv=fdv,
                        volume_24h=volume_24h,
                        liquidity=liquidity,
                        price_change_24h=price_change_24h,
                        buys_24h=buys_24h,
                        sells_24h=sells_24h,
                        pair_age_ms=pair_created_at
                    )
                    
                    # Combine technical + social scores
                    # If social data available (score > 0), use 70% technical + 30% social
                    # Otherwise, fall back to 100% technical scoring
                    if social_metrics.social_score > 0:
                        combined_score = (technical_score * 0.7) + (social_metrics.social_score * 0.3)
                    else:
                        combined_score = technical_score
                    
                    # Enhanced reasoning
                    reason_parts = [technical_reason]
                    if social_metrics.social_score > 30:
                        reason_parts.append(f"Social buzz: {social_metrics.confidence_level}")
                    
                    combined_reason = " | ".join(reason_parts)
                    
                    if combined_score < 30:
                        continue
                    
                    meme_coin = MemeCoin(
                        symbol=symbol,
                        name=name,
                        address=token_address,
                        price_usd=price_usd,
                        market_cap=fdv if fdv else 0,
                        volume_24h=volume_24h,
                        liquidity_usd=liquidity,
                        price_change_24h=price_change_24h,
                        created_timestamp=pair_created_at,
                        holder_count=None,
                        social_score=social_metrics.social_score,
                        potential_score=combined_score,
                        reason=combined_reason,
                        dex_url=pair.get('url', ''),
                        social_metrics=social_metrics
                    )
                    
                    meme_coins.append(meme_coin)
                    
                except (KeyError, ValueError, TypeError) as e:
                    continue
            
            meme_coins.sort(key=lambda x: x.potential_score, reverse=True)
            result = meme_coins[:limit]
            
            self.cache[cache_key] = (time.time(), result)
            
            return result
            
        except Exception as e:
            print(f"Error fetching trending tokens: {e}")
            return []
    
    def search_new_launches(self, hours_old: int = 24, min_liquidity: float = 5000) -> List[MemeCoin]:
        """
        Search for newly launched tokens within specified hours.
        
        Args:
            hours_old: Maximum age in hours
            min_liquidity: Minimum liquidity in USD
            
        Returns:
            List of newly launched MemeCoin objects
        """
        all_tokens = self.get_trending_solana_tokens(limit=100)
        
        cutoff_time = int((datetime.now() - timedelta(hours=hours_old)).timestamp() * 1000)
        
        new_launches = [
            token for token in all_tokens
            if token.created_timestamp 
            and token.created_timestamp > cutoff_time
            and token.liquidity_usd >= min_liquidity
        ]
        
        return new_launches
    
    def _calculate_potential_score(
        self,
        fdv: float,
        volume_24h: float,
        liquidity: float,
        price_change_24h: float,
        buys_24h: int,
        sells_24h: int,
        pair_age_ms: Optional[int]
    ) -> tuple[float, str]:
        """
        Calculate 100x potential score (0-100) based on multiple factors.
        
        Returns:
            Tuple of (score, reasoning)
        """
        score = 0
        reasons = []
        
        if fdv == 0 or liquidity == 0:
            return 0, "Insufficient data"
        
        volume_to_liquidity = volume_24h / liquidity if liquidity > 0 else 0
        
        if 10000 < fdv < 1000000:
            score += 25
            reasons.append("Micro-cap (<$1M)")
        elif 1000000 <= fdv < 10000000:
            score += 15
            reasons.append("Small-cap ($1-10M)")
        elif fdv < 10000:
            score += 5
            reasons.append("Ultra-micro cap")
        
        if volume_to_liquidity > 3:
            score += 20
            reasons.append(f"High volume/liquidity ({volume_to_liquidity:.1f}x)")
        elif volume_to_liquidity > 1.5:
            score += 10
            reasons.append(f"Good volume/liquidity ({volume_to_liquidity:.1f}x)")
        
        if price_change_24h > 100:
            score += 20
            reasons.append(f"Strong momentum (+{price_change_24h:.0f}%)")
        elif price_change_24h > 50:
            score += 15
            reasons.append(f"Good momentum (+{price_change_24h:.0f}%)")
        elif price_change_24h > 20:
            score += 8
            reasons.append(f"Positive trend (+{price_change_24h:.0f}%)")
        elif price_change_24h < -30:
            score -= 10
            reasons.append(f"Negative trend ({price_change_24h:.0f}%)")
        
        total_txns = buys_24h + sells_24h
        buy_ratio = buys_24h / total_txns if total_txns > 0 else 0.5
        
        if buy_ratio > 0.65 and total_txns > 100:
            score += 15
            reasons.append(f"Strong buying pressure ({buy_ratio*100:.0f}% buys)")
        elif buy_ratio > 0.55 and total_txns > 50:
            score += 8
            reasons.append(f"Good buy/sell ratio ({buy_ratio*100:.0f}%)")
        
        if pair_age_ms:
            age_hours = (time.time() * 1000 - pair_age_ms) / (1000 * 60 * 60)
            
            if age_hours < 6:
                score += 15
                reasons.append(f"Fresh launch ({age_hours:.1f}h old)")
            elif age_hours < 24:
                score += 10
                reasons.append(f"New token ({age_hours:.1f}h old)")
            elif age_hours < 72:
                score += 5
                reasons.append(f"Recent launch ({age_hours/24:.1f}d old)")
        
        if liquidity > 50000:
            score += 10
            reasons.append(f"Strong liquidity (${liquidity/1000:.0f}K)")
        elif liquidity > 20000:
            score += 5
            reasons.append(f"Adequate liquidity (${liquidity/1000:.0f}K)")
        
        score = min(100, max(0, score))
        
        reason_text = "; ".join(reasons) if reasons else "Low potential indicators"
        
        return score, reason_text
    
    def get_token_details(self, token_address: str) -> Optional[MemeCoin]:
        """
        Get detailed information about a specific token.
        
        Args:
            token_address: Solana token address
            
        Returns:
            MemeCoin object or None if not found
        """
        try:
            url = f"{self.dexscreener_base}/tokens/{token_address}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if not data or 'pairs' not in data or not data['pairs']:
                return None
            
            pair = data['pairs'][0]
            base_token = pair.get('baseToken', {})
            
            price_usd = float(pair.get('priceUsd', 0))
            fdv = pair.get('fdv', 0)
            volume_24h = float(pair.get('volume', {}).get('h24', 0))
            liquidity = float(pair.get('liquidity', {}).get('usd', 0))
            price_change_24h = float(pair.get('priceChange', {}).get('h24', 0))
            
            txns_24h = pair.get('txns', {}).get('h24', {})
            buys_24h = txns_24h.get('buys', 0)
            sells_24h = txns_24h.get('sells', 0)
            pair_created_at = pair.get('pairCreatedAt')
            
            potential_score, reason = self._calculate_potential_score(
                fdv=fdv,
                volume_24h=volume_24h,
                liquidity=liquidity,
                price_change_24h=price_change_24h,
                buys_24h=buys_24h,
                sells_24h=sells_24h,
                pair_age_ms=pair_created_at
            )
            
            return MemeCoin(
                symbol=base_token.get('symbol', 'UNKNOWN'),
                name=base_token.get('name', 'Unknown'),
                address=token_address,
                price_usd=price_usd,
                market_cap=fdv if fdv else 0,
                volume_24h=volume_24h,
                liquidity_usd=liquidity,
                price_change_24h=price_change_24h,
                created_timestamp=pair_created_at,
                holder_count=None,
                social_score=0,
                potential_score=potential_score,
                reason=reason,
                dex_url=pair.get('url', '')
            )
            
        except Exception as e:
            print(f"Error fetching token details: {e}")
            return None
