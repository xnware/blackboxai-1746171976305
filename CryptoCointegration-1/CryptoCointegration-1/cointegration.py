import asyncio
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import text
from statsmodels.tsa.stattools import adfuller, coint

from config import config
from database import DatabaseManager
from indicators import calculate_hedge_ratio, calculate_half_life

logger = logging.getLogger(__name__)

def perform_adf_test(series: pd.Series) -> Dict[str, float]:
    """
    Perform Augmented Dickey-Fuller test on a time series
    
    Args:
        series: Time series data
        
    Returns:
        Dictionary with test results
    """
    try:
        # Ensure we have minimum observations
        if len(series) < config.analysis.adf_min_obs:
            return {
                "adf_stat": 0.0,
                "p_value": 1.0,
                "is_stationary": False
            }
        
        # Perform ADF test
        result = adfuller(
            series, 
            maxlag=config.analysis.max_lags, 
            regression='c'
        )
        
        adf_stat, p_value, _, _, critical_values, _ = result
        
        is_stationary = (
            adf_stat < config.analysis.adf_critical_value and 
            p_value < config.analysis.p_value_threshold
        )
        
        return {
            "adf_stat": float(adf_stat),
            "p_value": float(p_value),
            "is_stationary": bool(is_stationary)
        }
        
    except Exception as e:
        logger.error(f"Error performing ADF test: {str(e)}")
        return {
            "adf_stat": 0.0,
            "p_value": 1.0,
            "is_stationary": False
        }

class CointegrationAnalyzer:
    """Analyzes cryptocurrency pairs for cointegration"""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    async def analyze_pair(self, symbol1: str, symbol2: str) -> Optional[Dict[str, Any]]:
        """
        Test a pair of symbols for cointegration
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            
        Returns:
            Cointegration test results or None if error
        """
        try:
            # Get candle data
            with self.db.get_connection() as conn:
                query = """
                    SELECT cd.symbol, cd.candles_json
                    FROM contract_details cd
                    WHERE cd.symbol IN (:s1, :s2)
                      AND cd.interval = :interval
                """
                result = conn.execute(
                    text(query),
                    {"s1": symbol1, "s2": symbol2, "interval": config.analysis.candle_interval}
                ).fetchall()
                
            if len(result) < 2:
                logger.warning(f"Missing candle data for {symbol1} or {symbol2}")
                return None
                
            # Parse candle data
            candles = {}
            for row in result:
                symbol = row[0]
                candles_json = row[1]
                candles[symbol] = json.loads(candles_json)
                
            # Check for minimum candle count
            if (len(candles[symbol1]) < config.analysis.adf_min_obs or 
                len(candles[symbol2]) < config.analysis.adf_min_obs):
                logger.warning(f"Insufficient candle data for {symbol1} or {symbol2}")
                return None
                
            # Find common timeframe
            timestamps1 = set(int(c['t']) for c in candles[symbol1])
            timestamps2 = set(int(c['t']) for c in candles[symbol2])
            common_timestamps = sorted(timestamps1.intersection(timestamps2))
            
            # Check for minimum overlap
            min_candles = min(len(timestamps1), len(timestamps2))
            overlap_ratio = len(common_timestamps) / min_candles
            
            if overlap_ratio < config.analysis.min_timeframe_overlap:
                logger.warning(f"Insufficient timeframe overlap for {symbol1}/{symbol2}: {overlap_ratio:.2f}")
                return None
                
            # Create price series
            price_map1 = {int(c['t']): float(c['c']) for c in candles[symbol1]}
            price_map2 = {int(c['t']): float(c['c']) for c in candles[symbol2]}
            
            prices1 = [price_map1[t] for t in common_timestamps if t in price_map1]
            prices2 = [price_map2[t] for t in common_timestamps if t in price_map2]
            
            if len(prices1) != len(prices2):
                logger.warning(f"Mismatched price series length for {symbol1}/{symbol2}")
                return None
                
            # Convert to pandas Series for analysis
            series1 = pd.Series(prices1)
            series2 = pd.Series(prices2)
            
            # Test for cointegration
            result = coint(series1, series2, maxlag=config.analysis.max_lags)
            t_stat, p_value, critical_values = result
            
            # Calculate hedge ratio
            hedge_ratio = calculate_hedge_ratio(series1, series2)
            
            # Calculate spread using hedge ratio
            spread = series1 - hedge_ratio * series2
            
            # Calculate half-life of mean reversion
            half_life = calculate_half_life(spread)
            
            # Determine if cointegrated
            is_cointegrated = (
                p_value < config.analysis.p_value_threshold and
                t_stat < critical_values[1]  # 5% critical value
            )
            
            # Set validity period
            now = datetime.now(timezone.utc)
            valid_until = now + timedelta(hours=1)  # Default 1 hour validity
            
            # Format final result
            result = {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "coint_t_stat": float(t_stat),
                "p_value": float(p_value),
                "is_cointegrated": bool(is_cointegrated),
                "critical_values": str(critical_values),
                "hedge_ratio": float(hedge_ratio),
                "half_life": float(half_life),
                "interval": config.analysis.candle_interval,
                "valid_until": valid_until
            }
            
            # Save to database
            self.db.save_cointegration_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol1}/{symbol2}: {str(e)}")
            return None
    
    async def generate_pairs(self) -> List[Tuple[str, str]]:
        """
        Generate candidate pairs for cointegration testing
        
        Returns:
            List of symbol pairs
        """
        try:
            # Get contract details with ADF results
            with self.db.get_connection() as conn:
                query = """
                    SELECT cd.symbol, cd.adf_statistic, cd.p_value, s.volume_24h_usd
                    FROM contract_details cd
                    JOIN symbols s ON cd.symbol = s.symbol
                    WHERE 
                        s.volume_24h_usd > :min_volume AND
                        cd.interval = :interval
                    ORDER BY s.volume_24h_usd DESC
                """
                result = conn.execute(
                    text(query),
                    {
                        "min_volume": config.analysis.volume_threshold_usdt,
                        "interval": config.analysis.candle_interval
                    }
                ).fetchall()
                
            if not result:
                logger.warning("No contracts found for pair generation")
                return []
                
            # Convert to DataFrame for analysis
            df = pd.DataFrame(result, columns=['symbol', 'adf_statistic', 'p_value', 'volume_24h_usd'])
            
            # Filter by p-value to focus on non-stationary series
            # For cointegration, we typically want non-stationary price series
            df = df[df['p_value'] > config.analysis.p_value_threshold]
            
            # If we have very few symbols, just pair them all
            if len(df) <= 10:
                pairs = []
                symbols = df['symbol'].tolist()
                for i in range(len(symbols)):
                    for j in range(i+1, len(symbols)):
                        pairs.append((symbols[i], symbols[j]))
                return pairs
            
            # Use K-means clustering to group similar assets
            # Import here to avoid circular imports
            from sklearn.cluster import KMeans
            
            # Prepare features for clustering (volume and ADF statistics)
            X = df[['volume_24h_usd', 'adf_statistic']].copy()
            
            # Scale features
            X['volume_24h_usd'] = np.log1p(X['volume_24h_usd'])  # Log transform for better scaling
            X = (X - X.mean()) / X.std()  # Standardize
            
            # Determine number of clusters (sqrt of n is a common heuristic)
            n_clusters = min(10, max(2, int(np.sqrt(len(df)) / 2)))
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['cluster'] = kmeans.fit_predict(X)
            
            # Generate pairs within clusters
            pairs = []
            for cluster in df['cluster'].unique():
                cluster_symbols = df[df['cluster'] == cluster]['symbol'].tolist()
                # Sort by volume for more meaningful pairs
                cluster_symbols.sort(key=lambda s: df[df['symbol'] == s]['volume_24h_usd'].values[0], reverse=True)
                
                for i in range(len(cluster_symbols)):
                    # Limit the number of pairs per symbol to avoid combinatorial explosion
                    max_pairs_per_symbol = 5
                    for j in range(i+1, min(i+1+max_pairs_per_symbol, len(cluster_symbols))):
                        pairs.append((cluster_symbols[i], cluster_symbols[j]))
            
            logger.info(f"Generated {len(pairs)} candidate pairs for testing")
            return pairs
            
        except Exception as e:
            logger.error(f"Error generating pairs: {str(e)}")
            return []
    
    async def analyze_all_pairs(self) -> Dict[str, int]:
        """
        Run cointegration analysis on all candidate pairs
        
        Returns:
            Statistics about the analysis
        """
        try:
            # Generate candidate pairs
            pairs = await self.generate_pairs()
            
            if not pairs:
                logger.warning("No pairs generated for analysis")
                return {"total": 0, "tested": 0, "cointegrated": 0}
            
            # Process in batches to avoid memory issues
            batch_size = 10
            total_tested = 0
            total_cointegrated = 0
            
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i+batch_size]
                
                # Create tasks for concurrent processing
                tasks = [self.analyze_pair(s1, s2) for s1, s2 in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Error in batch analysis: {str(result)}")
                        continue
                        
                    if result:
                        total_tested += 1
                        if result["is_cointegrated"]:
                            total_cointegrated += 1
                
                # Prevent excessive database records
                if total_cointegrated >= config.analysis.max_cointegration_records:
                    logger.info(f"Reached maximum cointegration records: {total_cointegrated}")
                    break
            
            stats = {
                "total": len(pairs),
                "tested": total_tested,
                "cointegrated": total_cointegrated
            }
            
            logger.info(f"Cointegration analysis completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error in analyze_all_pairs: {str(e)}")
            return {"total": 0, "tested": 0, "cointegrated": 0}
