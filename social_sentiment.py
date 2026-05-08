"""
Social Sentiment Analysis Module

Aggregates social signals from multiple sources to enhance meme coin potential scoring:
- Twitter/X: Trend detection, mention tracking, engagement metrics
- LunarCrush: Crypto-specific social sentiment scores
- Reddit: Community buzz from r/CryptoMoonShots and related subreddits
- Discord: Community activity and signal tracking
- Helius: Solana whale wallet tracking
- Pump.fun: New Solana token launch detection

All social metrics are combined into a unified social score (0-100) that enhances
the base technical scoring algorithm.
"""

import os
import time
import requests
import requests.auth
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json


@dataclass
class SocialMetrics:
    """Social sentiment metrics for a token"""
    twitter_mentions: int = 0
    twitter_sentiment: float = 0.0  # -1 to 1
    twitter_engagement: int = 0

    lunarcrush_galaxy_score: float = 0.0  # 0-100
    lunarcrush_alt_rank: int = 0

    reddit_mentions: int = 0
    reddit_sentiment: float = 0.0  # -1 to 1
    reddit_upvotes: int = 0

    discord_messages: int = 0
    discord_active_members: int = 0

    whale_buys_24h: int = 0
    whale_sells_24h: int = 0
    whale_net_flow: float = 0.0  # USD

    is_pump_fun_launch: bool = False
    pump_fun_graduation: bool = False

    social_score: float = 0.0  # 0-100 aggregated score
    confidence_level: str = "LOW"  # LOW, MEDIUM, HIGH, VERY_HIGH


class SocialSentimentAnalyzer:
    """
    Aggregates social sentiment from multiple sources for meme coin analysis.
    """

    def __init__(self):
        # API Keys (loaded from environment)
        self.twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        self.lunarcrush_api_key = os.getenv("LUNARCRUSH_API_KEY")
        self.reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.helius_api_key = os.getenv("HELIUS_API_KEY")

        # API Endpoints
        self.twitter_api_base = "https://api.twitter.com/2"
        self.lunarcrush_api_base = "https://lunarcrush.com/api4/public"
        self.reddit_api_base = "https://oauth.reddit.com"
        self.helius_api_base = "https://api.helius.xyz/v0"
        self.pumpfun_api_base = "https://frontend-api.pump.fun"

        # Cache for social data (60 second TTL)
        self.cache = {}
        self.cache_timeout = 60

        # Reddit OAuth token
        self.reddit_token = None
        self.reddit_token_expiry = None

    def get_social_metrics(self, token_address: str, symbol: str, name: str) -> SocialMetrics:
        """
        Fetch and aggregate all social metrics for a token.

        Args:
            token_address: Solana token address
            symbol: Token symbol (e.g., "BONK")
            name: Token name

        Returns:
            SocialMetrics object with aggregated data
        """
        cache_key = f"social_{token_address}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_timeout:
                return cached_data

        metrics = SocialMetrics()

        # Fetch from each source (failures are gracefully handled)
        twitter_data = self._fetch_twitter_metrics(symbol, name)
        lunarcrush_data = self._fetch_lunarcrush_metrics(symbol)
        reddit_data = self._fetch_reddit_metrics(symbol, name)
        whale_data = self._fetch_whale_activity(token_address)
        pumpfun_data = self._fetch_pumpfun_status(token_address)

        # Populate metrics
        if twitter_data:
            metrics.twitter_mentions = twitter_data.get('mentions', 0)
            metrics.twitter_sentiment = twitter_data.get('sentiment', 0.0)
            metrics.twitter_engagement = twitter_data.get('engagement', 0)

        if lunarcrush_data:
            metrics.lunarcrush_galaxy_score = lunarcrush_data.get('galaxy_score', 0.0)
            metrics.lunarcrush_alt_rank = lunarcrush_data.get('alt_rank', 0)

        if reddit_data:
            metrics.reddit_mentions = reddit_data.get('mentions', 0)
            metrics.reddit_sentiment = reddit_data.get('sentiment', 0.0)
            metrics.reddit_upvotes = reddit_data.get('upvotes', 0)

        if whale_data:
            metrics.whale_buys_24h = whale_data.get('buys', 0)
            metrics.whale_sells_24h = whale_data.get('sells', 0)
            metrics.whale_net_flow = whale_data.get('net_flow', 0.0)

        if pumpfun_data:
            metrics.is_pump_fun_launch = pumpfun_data.get('is_launch', False)
            metrics.pump_fun_graduation = pumpfun_data.get('graduated', False)

        # Calculate aggregated social score
        metrics.social_score, metrics.confidence_level = self._calculate_social_score(metrics)

        # Cache result
        self.cache[cache_key] = (time.time(), metrics)

        return metrics

    def _fetch_twitter_metrics(self, symbol: str, name: str) -> Optional[Dict]:
        """
        Twitter integration disabled - not needed for this scanner.
        """
        return None

    def _fetch_lunarcrush_metrics(self, symbol: str) -> Optional[Dict]:
        """
        Fetch LunarCrush social sentiment metrics using FREE TIER endpoint.
        Uses /coins/list/v2 which is available on free tier.

        Returns:
            Dict with galaxy_score and alt_rank
        """
        if not self.lunarcrush_api_key:
            return None

        try:
            print(f"🌙 Fetching LunarCrush data for {symbol} (free tier)...")

            headers = {
                'Authorization': f'Bearer {self.lunarcrush_api_key}'
            }

            # Use free tier endpoint: coins/list/v2
            # Search for the specific coin by filtering the list
            response = requests.get(
                f"{self.lunarcrush_api_base}/coins/list/v2",
                headers=headers,
                params={'limit': 1000},  # Get more coins to find ours
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                coins = data.get('data', [])

                # Find our symbol in the list
                symbol_upper = symbol.upper()
                for coin in coins:
                    if coin.get('symbol', '').upper() == symbol_upper or coin.get('name', '').upper() == symbol_upper:
                        galaxy_score = coin.get('gs', 0.0)  # gs = Galaxy Score
                        alt_rank = coin.get('acr', 0)  # acr = AltRank

                        print(f"✅ LunarCrush data for {symbol}: galaxy_score={galaxy_score}, alt_rank={alt_rank}")

                        return {
                            'galaxy_score': galaxy_score,
                            'alt_rank': alt_rank
                        }

                print(f"⚠️  {symbol} not found in LunarCrush database")
                return None
            else:
                print(f"⚠️  LunarCrush API error: {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ Error fetching LunarCrush metrics: {e}")
            return None

    def _fetch_reddit_metrics(self, symbol: str, name: str) -> Optional[Dict]:
        """
        Reddit integration disabled - not needed for this scanner.
        """
        return None

    def _refresh_reddit_token(self):
        """Reddit integration disabled"""
        return None

    def _fetch_whale_activity(self, token_address: str) -> Optional[Dict]:
        """
        Fetch whale wallet activity using Helius API.

        Returns:
            Dict with whale buys, sells, and net flow
        """
        if not self.helius_api_key:
            print(f"⚠️  Helius API key not configured")
            return None

        try:
            print(f"🐋 Fetching Helius whale data for {token_address[:8]}...")
            # Get recent transactions for the token
            params = {
                'api-key': self.helius_api_key
            }

            payload = {
                'query': {
                    'tokenAddress': token_address,
                    'timeRange': '24h'
                }
            }

            response = requests.post(
                f"{self.helius_api_base}/transactions",
                params=params,
                json=payload,
                timeout=15
            )

            print(f"Helius API response status: {response.status_code}")

            if response.status_code == 200:
                transactions = response.json()

                whale_buys = 0
                whale_sells = 0
                net_flow = 0.0

                # Define whale threshold: transactions > $10,000
                WHALE_THRESHOLD = 10000

                for tx in transactions:
                    tx_type = tx.get('type')
                    amount_usd = tx.get('nativeTransfers', [{}])[0].get('amountUSD', 0)

                    if amount_usd >= WHALE_THRESHOLD:
                        if tx_type in ['SWAP_BUY', 'TRANSFER_IN']:
                            whale_buys += 1
                            net_flow += amount_usd
                        elif tx_type in ['SWAP_SELL', 'TRANSFER_OUT']:
                            whale_sells += 1
                            net_flow -= amount_usd

                print(f"✅ Helius data: {whale_buys} whale buys, {whale_sells} whale sells")

                return {
                    'buys': whale_buys,
                    'sells': whale_sells,
                    'net_flow': net_flow
                }
            else:
                print(f"⚠️  Helius API error: {response.status_code} - {response.text[:200]}")

            return None

        except Exception as e:
            print(f"Error fetching whale activity: {e}")
            return None

    def _fetch_pumpfun_status(self, token_address: str) -> Optional[Dict]:
        """
        Check if token is a Pump.fun launch and its graduation status.

        Returns:
            Dict with is_launch and graduated status
        """
        try:
            # Check Pump.fun API for token info
            response = requests.get(
                f"{self.pumpfun_api_base}/coins/{token_address}",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()

                return {
                    'is_launch': True,
                    'graduated': data.get('complete', False)  # Graduated to Raydium
                }

            return {
                'is_launch': False,
                'graduated': False
            }

        except Exception as e:
            # Token not found on Pump.fun or API error
            return {
                'is_launch': False,
                'graduated': False
            }

    def _calculate_social_score(self, metrics: SocialMetrics) -> Tuple[float, str]:
        """
        Calculate aggregated social score (0-100) from all metrics.

        Scoring breakdown:
        - Twitter activity (25 pts): Mentions + engagement + sentiment
        - LunarCrush score (20 pts): Galaxy score and alt rank
        - Reddit buzz (15 pts): Mentions + upvotes + sentiment
        - Whale activity (20 pts): Net whale flow and buy/sell ratio
        - Pump.fun bonus (10 pts): Launch detection and graduation
        - Discord activity (10 pts): Message volume and active members

        Returns:
            Tuple of (score, confidence_level)
        """
        score = 0.0

        # Twitter Score (25 pts)
        if metrics.twitter_mentions > 0:
            mention_score = min(10, metrics.twitter_mentions / 10)  # 10 pts max
            engagement_score = min(10, metrics.twitter_engagement / 1000)  # 10 pts max
            sentiment_score = max(0, metrics.twitter_sentiment * 5)  # 5 pts max
            score += mention_score + engagement_score + sentiment_score

        # LunarCrush Score (20 pts)
        if metrics.lunarcrush_galaxy_score > 0:
            score += min(20, metrics.lunarcrush_galaxy_score / 5)  # Galaxy score is 0-100

        # Reddit Score (15 pts)
        if metrics.reddit_mentions > 0:
            mention_score = min(7, metrics.reddit_mentions * 0.5)  # 7 pts max
            upvote_score = min(5, metrics.reddit_upvotes / 100)  # 5 pts max
            sentiment_score = max(0, metrics.reddit_sentiment * 3)  # 3 pts max
            score += mention_score + upvote_score + sentiment_score

        # Whale Activity Score (20 pts)
        if metrics.whale_buys_24h > 0 or metrics.whale_sells_24h > 0:
            total_whale_txs = metrics.whale_buys_24h + metrics.whale_sells_24h
            buy_ratio = metrics.whale_buys_24h / total_whale_txs if total_whale_txs > 0 else 0.5

            # More whale buys = higher score
            ratio_score = buy_ratio * 10  # 10 pts max
            flow_score = min(10, abs(metrics.whale_net_flow) / 100000)  # 10 pts max

            # Positive flow adds, negative flow reduces
            if metrics.whale_net_flow > 0:
                score += ratio_score + flow_score
            else:
                score += max(0, ratio_score - flow_score)

        # Pump.fun Bonus (10 pts)
        if metrics.is_pump_fun_launch:
            score += 5  # Fresh launch bonus
            if metrics.pump_fun_graduation:
                score += 5  # Graduated to Raydium = passed initial test

        # Discord Score (10 pts) - placeholder for future Discord integration
        if metrics.discord_messages > 0:
            message_score = min(5, metrics.discord_messages / 100)
            member_score = min(5, metrics.discord_active_members / 50)
            score += message_score + member_score

        # Determine confidence level
        if score >= 70:
            confidence = "VERY_HIGH"
        elif score >= 50:
            confidence = "HIGH"
        elif score >= 30:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return round(score, 2), confidence


def get_social_analyzer() -> SocialSentimentAnalyzer:
    """Get singleton instance of social sentiment analyzer"""
    return SocialSentimentAnalyzer()
