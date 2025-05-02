import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans

from config import config
from database import DatabaseManager
from api_client import APIClient
from utils import (
    adf_test, 
    calculate_hedge_ratio, 
    calculate_spread,
    calculate_z_score,
    cointegration_test,
    calculate_half_life,
    convert_candles_to_dataframe,
    calculate_percent_change
)

logger = logging.getLogger(__name__)

class DataCollector:
    """Collects data from cryptocurrency exchange API"""
    
    def __init__(self, api_client: APIClient, db_manager: DatabaseManager):
        self.api = api_client
        self.db = db_manager
        self.concurrent_limit = config.api.concurrent_requests
    
    async def fetch_and_save_contracts(self):
        """
        Fetch all available contracts and save to database
        """
        try:
            # Get all contracts with tickers for volume info
            contracts = await self.api.get_futures_contracts()
            tickers = await self.api.get_futures_tickers()
            
            # Create volume lookup dict
            volume_map = {t.get('contract'): float(t.get('volume_24h_usd', 0)) 
                          for t in tickers if 'contract' in t}
            
            # Filter and prepare data for storage
            threshold = config.analysis.volume_threshold
            contract_data = []
            
            for contract in contracts:
                symbol = contract.get('name')
                volume = volume_map.get(symbol, 0)
                
                # Skip low volume contracts
                if volume < threshold:
                    continue
                
                contract_data.append({
                    "symbol": symbol,
                    "volume_24h_usd": volume
                })
            
            # Save basic contract data
            if contract_data:
                df = pd.DataFrame(contract_data)
                self.db.save_dataframe(df, "contract_details", if_exists="append")
                logger.info(f"Saved {len(contract_data)} contracts with volume > {threshold}")
            
            return contract_data
        
        except Exception as e:
            logger.error(f"Error fetching contracts: {str(e)}")
            return []
    
    async def process_symbol(self, symbol: str):
        """
        Process a single symbol: fetch candles and calculate metrics
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict with symbol metrics or None on error
        """
        try:
            # Fetch candlestick data
            candles = await self.api.get_futures_candles(
                symbol, 
                interval=config.analysis.candle_interval,
                limit=config.analysis.candle_limit
            )
            
            if not candles or len(candles) < config.analysis.candle_limit:
                logger.warning(f"Insufficient candle data for {symbol}")
                return None
            
            # Convert to dataframe and extract close prices
            df = convert_candles_to_dataframe(candles)
            if df.empty or 'close' not in df.columns:
                return None
                
            close_prices = df['close']
            
            # Perform stationarity test (ADF)
            adf_stat, p_value, _ = adf_test(close_prices)
            
            # Optional volume calculation
            volume_24h = df['volume'].tail(288).sum() if 'volume' in df.columns else 0
            
            return {
                "symbol": symbol,
                "adf_stat": adf_stat,
                "p_value": p_value,
                "volume_24h_usd": volume_24h,
                "candles": df.to_json(orient='records'),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {str(e)}")
            return None
    
    async def fetch_all_symbols(self):
        """
        Fetch and process all active symbols
        """
        # Get active symbols from database
        symbols = self.db.get_active_symbols()
        
        if not symbols:
            logger.info("No active symbols found, fetching contracts first")
            await self.fetch_and_save_contracts()
            symbols = self.db.get_active_symbols()
        
        # Process symbols with rate limiting
        semaphore = asyncio.Semaphore(self.concurrent_limit)
        
        async def process_with_semaphore(symbol):
            async with semaphore:
                return await self.process_symbol(symbol)
        
        tasks = [process_with_semaphore(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results
        valid_results = [r for r in results if r]
        
        # Save results to database
        for result in valid_results:
            self.db.save_contract_details(result)
        
        logger.info(f"Processed {len(valid_results)} symbols out of {len(symbols)}")
        return valid_results

class PairAnalyzer:
    """Analyzes cryptocurrency pairs for cointegration and statistical arbitrage"""
    
    def __init__(self, api_client: APIClient, db_manager: DatabaseManager):
        self.api = api_client
        self.db = db_manager
        self.concurrent_limit = config.api.concurrent_requests
    
    def generate_pairs(self, symbols: List[str], method: str = "kmeans") -> List[Tuple[str, str]]:
        """
        Generate pairs of symbols for testing using clustering
        
        Args:
            symbols: List of symbols to pair
            method: Pairing method ('all', 'kmeans')
            
        Returns:
            List of symbol pairs (tuples)
        """
        if not symbols or len(symbols) < 2:
            return []
        
        if method == "all":
            # Generate all possible pairs (warning: can be a lot for large symbol lists)
            return [(symbols[i], symbols[j]) 
                    for i in range(len(symbols)) 
                    for j in range(i+1, len(symbols))]
        
        elif method == "kmeans":
            # Get features for clustering from database
            query = """
            SELECT symbol, volume_24h_usd, adf_statistic 
            FROM contract_details 
            WHERE symbol IN :symbols
            """
            
            with config.db.connection() as conn:
                df = pd.read_sql(
                    sa.text(query), 
                    conn, 
                    params={"symbols": tuple(symbols)}
                )
            
            if df.empty:
                logger.warning("No data for clustering")
                return []
            
            # Fill missing values
            df = df.fillna(0)
            
            # Normalize features
            features = df[['volume_24h_usd', 'adf_statistic']]
            features = (features - features.mean()) / features.std()
            
            # Apply KMeans clustering
            n_clusters = min(config.app.cluster_num, len(df) // 2)
            if n_clusters < 2:
                n_clusters = 2
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['cluster'] = kmeans.fit_predict(features)
            
            # Generate pairs within clusters
            pairs = []
            for cluster_id in df['cluster'].unique():
                cluster_symbols = df[df['cluster'] == cluster_id]['symbol'].tolist()
                cluster_pairs = [(cluster_symbols[i], cluster_symbols[j])
                                for i in range(len(cluster_symbols))
                                for j in range(i+1, len(cluster_symbols))]
                pairs.extend(cluster_pairs)
            
            return pairs
        
        else:
            logger.error(f"Unknown pairing method: {method}")
            return []
    
    async def test_cointegration(self, symbol1: str, symbol2: str) -> Optional[Dict]:
        """
        Test a pair of symbols for cointegration
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            
        Returns:
            Dictionary with cointegration results or None on failure
        """
        try:
            # Fetch candle data for both symbols
            candles1 = await self.api.get_futures_candles(symbol1)
            candles2 = await self.api.get_futures_candles(symbol2)
            
            # Convert to dataframes
            df1 = convert_candles_to_dataframe(candles1)
            df2 = convert_candles_to_dataframe(candles2)
            
            if df1.empty or df2.empty:
                return None
            
            # Extract close prices
            price1 = df1['close']
            price2 = df2['close']
            
            # Align timestamps
            # First convert to DataFrame for consistent merging
            df = pd.DataFrame({
                'time1': df1['timestamp'],
                'price1': price1,
                'time2': df2['timestamp'],
                'price2': price2
            })
            
            # Check for sufficient overlap
            overlap_ratio = len(df.dropna()) / max(len(df1), len(df2))
            if overlap_ratio < config.analysis.min_timeframe_overlap:
                logger.warning(f"Insufficient timeframe overlap for {symbol1}/{symbol2}: {overlap_ratio:.2f}")
                return None
            
            # Use aligned data for testing
            aligned_df = df.dropna()
            if len(aligned_df) < config.analysis.adf_min_obs:
                logger.warning(f"Insufficient data points after alignment for {symbol1}/{symbol2}")
                return None
                
            price1 = aligned_df['price1']
            price2 = aligned_df['price2']
            
            # Calculate hedge ratio
            hedge_ratio = calculate_hedge_ratio(price1, price2)
            
            # Calculate spread
            spread = calculate_spread(price1, price2, hedge_ratio)
            
            # Test stationarity of spread
            adf_stat, p_value, is_stationary = adf_test(spread)
            
            # Calculate half-life if cointegrated
            half_life = None
            if is_stationary:
                half_life = calculate_half_life(spread)
            
            # Store results
            result = {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "p_value": p_value,
                "adf_statistic": adf_stat,
                "hedge_ratio": hedge_ratio,
                "half_life": half_life,
                "is_cointegrated": is_stationary
            }
            
            # Save to database if cointegrated
            if is_stationary:
                self.db.save_cointegration_result(
                    symbol1, symbol2, p_value, adf_stat, hedge_ratio, half_life
                )
                
                # Compute and save spread data
                spread_data = []
                for i in range(len(aligned_df)):
                    timestamp = aligned_df.iloc[i]['time1']
                    p1 = aligned_df.iloc[i]['price1']
                    p2 = aligned_df.iloc[i]['price2']
                    
                    # Get price changes - can be NaN for first entry
                    if i > 0:
                        pc1 = (p1 / aligned_df.iloc[i-1]['price1'] - 1) * 100
                        pc2 = (p2 / aligned_df.iloc[i-1]['price2'] - 1) * 100
                    else:
                        pc1 = pc2 = 0
                    
                    # Calculate spread
                    sprd = p1 - hedge_ratio * p2
                    
                    spread_data.append({
                        "timestamp": timestamp,
                        "spread": sprd,
                        "price1": p1,
                        "price2": p2,
                        "pc1": pc1,
                        "pc2": pc2,
                        "z_score": 0  # Will be updated later with rolling calculation
                    })
                
                # Calculate z-scores with rolling window
                spread_series = pd.Series([d['spread'] for d in spread_data])
                z_scores = calculate_z_score(spread_series)
                
                # Update z-scores in data
                for i in range(len(spread_data)):
                    if i < len(z_scores):
                        spread_data[i]['z_score'] = z_scores[i] if not np.isnan(z_scores[i]) else 0
                
                # Get pair ID from database
                pair = self.db.get_pair_details(symbol1, symbol2)
                if pair and 'id' in pair:
                    self.db.save_spread_data(pair['id'], spread_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing cointegration for {symbol1}/{symbol2}: {str(e)}")
            return None
    
    async def analyze_all_pairs(self):
        """
        Analyze all pairs for cointegration
        """
        # Get active symbols
        symbols = self.db.get_active_symbols()
        
        if not symbols:
            logger.warning("No active symbols found")
            return []
        
        # Generate pairs using clustering
        pairs = self.generate_pairs(symbols, method="kmeans")
        logger.info(f"Generated {len(pairs)} pairs for testing")
        
        # Process pairs with rate limiting
        semaphore = asyncio.Semaphore(self.concurrent_limit)
        
        async def test_with_semaphore(pair):
            symbol1, symbol2 = pair
            async with semaphore:
                return await self.test_cointegration(symbol1, symbol2)
        
        tasks = [test_with_semaphore(pair) for pair in pairs]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results and non-cointegrated pairs
        valid_results = [r for r in results if r and r.get('is_cointegrated', False)]
        
        logger.info(f"Found {len(valid_results)} cointegrated pairs out of {len(pairs)} tested")
        return valid_results

async def run_analysis_cycle():
    """
    Run a full analysis cycle:
    1. Collect data
    2. Find cointegrated pairs
    3. Calculate metrics
    """
    db = DatabaseManager()
    
    async with APIClient() as api:
        # Initialize collectors and analyzers
        collector = DataCollector(api, db)
        analyzer = PairAnalyzer(api, db)
        
        # Collect contract and candlestick data
        logger.info("Starting data collection")
        await collector.fetch_all_symbols()
        
        # Analyze pairs for cointegration
        logger.info("Starting pair analysis")
        cointegrated_pairs = await analyzer.analyze_all_pairs()
        
        logger.info(f"Analysis cycle complete, found {len(cointegrated_pairs)} cointegrated pairs")
        return cointegrated_pairs
