import os
import asyncio
import logging
import signal
import platform
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
from dotenv import load_dotenv
import aiohttp
from pathlib import Path
import json
from collections import deque
from statsmodels.tsa.stattools import adfuller, coint
from aiolimiter import AsyncLimiter
import hmac
import hashlib
import math
from tqdm.asyncio import tqdm
import modin.pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from config import Config
from tenacity import retry, stop_after_attempt, wait_exponential
import modin
from config import config

# Фикс для Windows
if platform.system() == 'Windows':
    from asyncio import WindowsSelectorEventLoopPolicy
    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

# Пути и загрузка настроек
BASE_DIR = Path(__file__).resolve().parent
DOTENV_PATH = BASE_DIR / '.env'
if not DOTENV_PATH.exists():
    raise FileNotFoundError(f"Файл .env не найден: {DOTENV_PATH}")
load_dotenv(DOTENV_PATH)

# Логирование
class EnhancedJSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
            "exc_info": self.formatException(record.exc_info) if record.exc_info else None
        }
        return json.dumps(log_data, ensure_ascii=False)

def setup_logging():
    logger = logging.getLogger("FuturesMonitor")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("futures_monitor.log", encoding='utf-8')
    fh.setFormatter(EnhancedJSONFormatter())
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(module)s:%(lineno)d]"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging()

config = Config()
shutdown_event = asyncio.Event()
limiter = AsyncLimiter(100, 10)

class ClusterManager:
    def __init__(self):
        self.model = KMeans(n_clusters=config.app.cluster_num)
        self.features = None

    async def load_features(self):
        """Загрузка данных для кластеризации"""
        query = """
            SELECT symbol, volume_24h_usd, adf_statistic 
            FROM contract_details 
            WHERE interval = '5m'
        """
        self.features = pd.read_sql(query, config.DB_CONFIG)
        
    def cluster_symbols(self):
        """Кластеризация символов"""
        if self.features is None:
            raise ValueError("Данные не загружены")
        return self.model.fit_predict(self.features[['volume_24h_usd', 'adf_statistic']])

class DatabaseManager:
    def __init__(self):
        self.cluster_manager = ClusterManager()
        
    async def connect(self):
        """Инициализация подключения и данных"""
        await self.cluster_manager.load_features()
        logger.info("Кластеры инициализированы")

    async def get_cointegrated_pairs(self):
        """Генерация пар для тестирования"""
        clusters = self.cluster_manager.cluster_symbols()
        pairs = []
        for cluster_id in np.unique(clusters):
            symbols = self.cluster_manager.features[clusters == cluster_id]['symbol']
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    pairs.append((symbols[i], symbols[j]))
        return pairs

class APIClient:
    def __init__(self):
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc):
        await self.session.close()

    def _sign_request(self, method, url):
        """Подпись API-запросов"""
        timestamp = str(int(datetime.now(timezone.utc).timestamp()))
        msg = '\n'.join([method, urlparse(url).path, '', timestamp])
        signature = hmac.new(config.API_SECRET, msg.encode(), hashlib.sha512).hexdigest()
        return {
            "KEY": config.API_KEY,
            "Timestamp": timestamp,
            "SIGN": signature
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_candles(self, symbol):
        """Получение свечных данных"""
        url = f"{config.API_URL}/futures/usdt/candlesticks"
        params = {
            "contract": symbol,
            "interval": "5m",
            "limit": config.CANDLE_LIMIT
        }
        
        async with limiter:
            async with self.session.get(
                url, 
                headers=self._sign_request("GET", url),
                params=params
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

async def process_symbol(symbol: str, client: APIClient):
    """Обработка одного символа"""
    try:
        candles = await client.get_candles(symbol)
        if len(candles) < config.CANDLE_LIMIT:
            return None
            
        closes = [float(c['c']) for c in candles]
        adf_stat, p_value = adfuller(closes, maxlag=1)
        
        return {
            "symbol": symbol,
            "adf_stat": adf_stat,
            "p_value": p_value,
            "candles": json.dumps(candles)
        }
        
    except Exception as e:
        logger.error(f"Ошибка обработки {symbol}: {str(e)}")
        return None

async def main_cycle():
    """Основной цикл обработки"""
    db = DatabaseManager()
    await db.connect()
    
    async with APIClient() as client:
        while not shutdown_event.is_set():
            start_time = datetime.now()
            
            # Получение активных символов
            symbols = [...]  # Загрузка символов из БД
            
            # Параллельная обработка
            tasks = [process_symbol(symbol, client) for symbol in symbols]
            results = await asyncio.gather(*tasks)
            
            # Фильтрация и сохранение результатов
            valid_results = [r for r in results if r]
            if valid_results:
                pd.DataFrame(valid_results).to_sql(
                    'contract_details', 
                    con=config.DB_CONFIG,
                    if_exists='append',
                    index=False
                )
                logger.info(f"Сохранено {len(valid_results)} записей")
                
            # Тестирование коинтеграции
            pairs = await db.get_cointegrated_pairs()
            logger.info(f"Тестирование {len(pairs)} пар")
            
            # Расчет временных затрат
            elapsed = (datetime.now() - start_time).total_seconds()
            await asyncio.sleep(max(300 - elapsed, 5))

def signal_handler(sig, frame):
    """Обработчик сигналов завершения"""
    logger.info("Получен сигнал завершения")
    shutdown_event.set()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(main_cycle())
    except KeyboardInterrupt:
        logger.info("Работа завершена пользователем")