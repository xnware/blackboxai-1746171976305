import asyncio
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from aiolimiter import AsyncLimiter

from config import config
from api_client import APIClient
from database import DatabaseManager

logger = logging.getLogger(__name__)

class DataCollector:
    """Collects data from cryptocurrency exchanges"""
    
    def __init__(self):
        self.api_client = APIClient()
        self.db = DatabaseManager()
        self.limiter = AsyncLimiter(
            config.api.rate_limit / config.api.concurrent_requests,
            config.api.rate_period / config.api.concurrent_requests
        )
    
    async def collect_symbols(self) -> List[str]:
        """Collect and save all active trading symbols"""
        try:
            symbols_data = await self.api_client.get_futures_contracts()
            
            if not symbols_data:
                logger.warning("No symbols data received from API")
                return []
            
            # Transform data for database
            db_symbols = []
            active_symbols = []
            
            for symbol in symbols_data:
                if not isinstance(symbol, dict):
                    continue
                    
                volume = float(symbol.get('volume_24h_usd', 0)) 
                if volume < config.analysis.volume_threshold_usdt:
                    continue
                    
                db_symbols.append({
                    'symbol': symbol.get('name', ''),
                    'base_currency': symbol.get('underlying', ''),
                    'quote_currency': 'USDT',  # Futures are USDT settled
                    'volume_24h_usd': volume,
                    'last_price': float(symbol.get('last_price', 0)),
                    'active': True
                })
                
                active_symbols.append(symbol.get('name', ''))
            
            # Save to database
            if db_symbols:
                self.db.save_symbols(db_symbols)
                logger.info(f"Saved {len(db_symbols)} symbols to database")
            
            return active_symbols
            
        except Exception as e:
            logger.error(f"Error collecting symbols: {str(e)}")
            return []
    
    async def collect_candles(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect and save candle data for a symbol"""
        try:
            async with self.limiter:
                candles = await self.api_client.get_candles(
                    symbol, 
                    config.analysis.candle_interval,
                    config.analysis.candle_limit
                )
            
            if not candles or len(candles) < config.analysis.adf_min_obs:
                logger.warning(f"Insufficient candle data for {symbol}")
                return None
            
            # Save candles to database
            self.db.save_candles(symbol, candles, config.analysis.candle_interval)
            
            # Calculate ADF test
            closes = [float(c['c']) for c in candles]
            
            # Import here to avoid circular import
            from cointegration import perform_adf_test
            adf_result = perform_adf_test(pd.Series(closes))
            
            # Get latest volume data
            volume_24h = 0
            if candles:
                # Sum last 24h of volume (assuming 5m candles = 288 candles in 24h)
                last_24h = min(len(candles), 288)
                volume_24h = sum(float(c['v']) * float(c['c']) for c in candles[-last_24h:])
            
            # Prepare contract details
            details = {
                'symbol': symbol,
                'adf_stat': adf_result['adf_stat'],
                'p_value': adf_result['p_value'],
                'volume_24h_usd': volume_24h,
                'candles': json.dumps(candles)
            }
            
            # Save contract details
            self.db.save_contract_details(details)
            
            return details
            
        except Exception as e:
            logger.error(f"Error collecting candles for {symbol}: {str(e)}")
            return None
    
    async def collect_all_candles(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Collect candle data for multiple symbols concurrently"""
        try:
            tasks = []
            for symbol in symbols:
                tasks.append(self.collect_candles(symbol))
            
            # Process in batches to avoid hitting rate limits
            batch_size = config.api.concurrent_requests
            results = []
            
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error in batch processing: {str(result)}")
                    elif result:
                        results.append(result)
                
                # Add delay between batches
                if i + batch_size < len(tasks):
                    await asyncio.sleep(config.api.request_delay)
            
            logger.info(f"Collected candle data for {len(results)} symbols")
            return [r for r in results if r]
            
        except Exception as e:
            logger.error(f"Error in collect_all_candles: {str(e)}")
            return []
    
    async def update_spread_data(self, symbol1: str, symbol2: str, hedge_ratio: float) -> bool:
        """Update spread data for a pair of symbols"""
        try:
            # Get latest candle for each symbol
            candle1 = await self.api_client.get_candles(symbol1, config.analysis.candle_interval, 1)
            candle2 = await self.api_client.get_candles(symbol2, config.analysis.candle_interval, 1)
            
            if not candle1 or not candle2:
                logger.warning(f"Missing candle data for {symbol1} or {symbol2}")
                return False
            
            # Get current prices
            price1 = float(candle1[0]['c'])
            price2 = float(candle2[0]['c'])
            
            # Calculate spread
            spread = price1 - hedge_ratio * price2
            
            # Get historical spread data
            query = f"""
                SELECT spread
                FROM spread_data
                WHERE 
                    symbol1 = '{symbol1}' AND 
                    symbol2 = '{symbol2}' AND
                    interval = '{config.analysis.candle_interval}'
                ORDER BY timestamp DESC
                LIMIT 20
            """
            
            with self.db.get_connection() as conn:
                df = pd.read_sql(query, conn)
            
            if len(df) > 0:
                # Calculate z-score
                historical_spreads = df['spread'].tolist()
                historical_spreads.append(spread)
                spreads_series = pd.Series(historical_spreads)
                
                mean = spreads_series.mean()
                std = spreads_series.std()
                
                z_score = 0.0
                if std > 1e-8:  # Avoid division by zero
                    z_score = (spread - mean) / std
            else:
                z_score = 0.0
            
            # Save spread data
            now = datetime.now(timezone.utc)
            spread_data = {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'timestamp': now,
                'price1': price1,
                'price2': price2,
                'spread': spread,
                'z_score': z_score,
                'interval': config.analysis.candle_interval
            }
            
            self.db.save_spread_data(spread_data)
            return True
            
        except Exception as e:
            logger.error(f"Error updating spread data for {symbol1}/{symbol2}: {str(e)}")
            return False
    
    async def update_all_spreads(self) -> int:
        """Update spread data for all cointegrated pairs"""
        try:
            # Get cointegrated pairs
            pairs_df = self.db.load_cointegrated_pairs()
            
            if pairs_df.empty:
                logger.warning("No cointegrated pairs found")
                return 0
            
            count = 0
            for _, row in pairs_df.iterrows():
                success = await self.update_spread_data(
                    row['symbol1'], 
                    row['symbol2'], 
                    row['hedge_ratio']
                )
                if success:
                    count += 1
            
            logger.info(f"Updated spread data for {count} pairs")
            return count
            
        except Exception as e:
            logger.error(f"Error in update_all_spreads: {str(e)}")
            return 0
    
    async def collect_all_data(self) -> Dict[str, int]:
        """Run a complete data collection cycle"""
        results = {
            "symbols": 0,
            "candles": 0,
            "spreads": 0
        }
        
        try:
            # 1. Collect symbols
            symbols = await self.collect_symbols()
            results["symbols"] = len(symbols)
            
            if not symbols:
                logger.warning("No symbols collected, aborting data collection cycle")
                return results
            
            # 2. Collect candle data
            candle_data = await self.collect_all_candles(symbols)
            results["candles"] = len(candle_data)
            
            # 3. Update spread data for cointegrated pairs
            spreads_count = await self.update_all_spreads()
            results["spreads"] = spreads_count
            
            logger.info(f"Data collection cycle completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in data collection cycle: {str(e)}")
            return results
