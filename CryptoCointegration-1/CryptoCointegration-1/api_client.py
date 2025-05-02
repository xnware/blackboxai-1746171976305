import asyncio
import aiohttp
import hmac
import hashlib
import logging
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import config

logger = logging.getLogger(__name__)

class APIClient:
    """Client for interacting with Gate.io API"""
    
    def __init__(self):
        self.api_url = config.api.api_url
        self.api_key = config.api.api_key
        self.api_secret = config.api.api_secret
        self.session = None
        self.limiter = AsyncLimiter(
            config.api.rate_limit, 
            config.api.rate_period
        )
    
    async def __aenter__(self):
        """Initialize session with context manager"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close session on exit"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _generate_signature(self, method: str, endpoint: str, query_string: str = "") -> Dict[str, str]:
        """
        Generate authentication signature for Gate.io API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint without base URL
            query_string: URL query string if any
            
        Returns:
            Headers dictionary with authentication information
        """
        if not self.api_key or not self.api_secret:
            logger.warning("API key or secret not configured")
            return {}
            
        timestamp = str(int(datetime.now(timezone.utc).timestamp()))
        
        # Create signature string
        if query_string and not query_string.startswith('?'):
            query_string = '?' + query_string
            
        signature_string = f"{method}\n{endpoint}{query_string}\n\n{timestamp}"
        
        # Create HMAC signature
        signature = hmac.new(
            self.api_secret.encode(), 
            signature_string.encode(), 
            hashlib.sha512
        ).hexdigest()
        
        # Return headers
        return {
            "KEY": self.api_key,
            "Timestamp": timestamp,
            "SIGN": signature
        }
    
    @retry(
        stop=stop_after_attempt(config.api.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            aiohttp.ClientError, 
            asyncio.TimeoutError,
            json.JSONDecodeError
        ))
    )
    async def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Any:
        """
        Make an authenticated request to the Gate.io API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint without base URL
            params: Query parameters or request body
            
        Returns:
            API response data
        """
        # Create session if needed
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
        # Build URL and query string
        url = f"{self.api_url}{endpoint}"
        query_string = ""
        
        if params and method == "GET":
            query_string = urlencode(params)
            url = f"{url}?{query_string}"
            
        # Generate signature
        headers = self._generate_signature(method, endpoint, query_string)
        
        # Apply rate limiting
        async with self.limiter:
            try:
                if method == "GET":
                    async with self.session.get(url, headers=headers) as response:
                        response.raise_for_status()
                        return await response.json()
                elif method == "POST":
                    async with self.session.post(url, json=params, headers=headers) as response:
                        response.raise_for_status()
                        return await response.json()
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                    
            except aiohttp.ClientResponseError as e:
                logger.error(f"API error: {e.status} - {str(e)}")
                
                # Handle rate limiting specially
                if e.status == 429:
                    logger.warning("Rate limit exceeded, backing off")
                    await asyncio.sleep(5)
                    
                raise
            except aiohttp.ClientError as e:
                logger.error(f"HTTP error: {str(e)}")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in response: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise
    
    async def get_futures_contracts(self) -> List[Dict[str, Any]]:
        """
        Get list of USDT futures contracts
        
        Returns:
            List of contract data
        """
        try:
            endpoint = "/futures/usdt/contracts"
            response = await self._make_request("GET", endpoint)
            
            if not isinstance(response, list):
                logger.error(f"Unexpected response format: {response}")
                return []
                
            return response
            
        except Exception as e:
            logger.error(f"Error getting futures contracts: {str(e)}")
            return []
    
    async def get_candles(self, symbol: str, interval: str = "5m", limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get candlestick data for a symbol
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval (e.g., 5m, 15m, 1h, 4h, 1d)
            limit: Maximum number of candles to retrieve
            
        Returns:
            List of candle data
        """
        try:
            endpoint = "/futures/usdt/candlesticks"
            params = {
                "contract": symbol,
                "interval": interval,
                "limit": limit
            }
            
            response = await self._make_request("GET", endpoint, params)
            
            if not isinstance(response, list):
                logger.error(f"Unexpected candle response format: {response}")
                return []
                
            return response
            
        except Exception as e:
            logger.error(f"Error getting candles for {symbol}: {str(e)}")
            return []
    
    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current ticker data for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Ticker data or None on error
        """
        try:
            endpoint = "/futures/usdt/tickers"
            params = {"contract": symbol}
            
            response = await self._make_request("GET", endpoint, params)
            
            if not response:
                logger.warning(f"No ticker data for {symbol}")
                return None
                
            return response[0] if isinstance(response, list) else response
            
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {str(e)}")
            return None
