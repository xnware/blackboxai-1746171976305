import os
import asyncio
import logging
import signal
import platform
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from contextlib import asynccontextmanager

import aiohttp
import numpy as np
import pandas as pd
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import adfuller
from tenacity import retry, stop_after_attempt, wait_exponential

from config import Config
from database import DatabaseManager

# Настройки окружения
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')

# Фикс для Windows
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class EnhancedJSONFormatter(logging.Formatter):
    """Кастомный форматтер логов в JSON"""
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": f"{record.module}:{record.lineno}",
            "message": record.getMessage(),
            "exception": self.formatException(record.exc_info) if record.exc_info else None
        }, ensure_ascii=False)

def setup_logging():
    """Инициализация системы логирования"""
    logger = logging.getLogger("CryptoMonitor")
    logger.setLevel(logging.DEBUG)
    
    # File handler с JSON-форматом
    fh = logging.FileHandler("monitor.log", encoding='utf-8')
    fh.setFormatter(EnhancedJSONFormatter())
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging()
config = Config()
limiter = AsyncLimiter(100, 10)
shutdown_event = asyncio.Event()

class SymbolProcessor:
    """Обработчик данных для одного символа"""
    def __init__(self, client):
        self.client = client

    @retry(stop=stop_after_attempt(3), 
           wait=wait_exponential(multiplier=1, min=2, max=10))
    async def process(self, symbol: str):
        """Основной метод обработки символа"""
        try:
            candles = await self._fetch_candles(symbol)
            if len(candles) < config.CANDLE_LIMIT:
                return None
                
            return self._analyze_candles(symbol, candles)
            
        except Exception as e:
            logger.error(f"Ошибка обработки {symbol}: {str(e)}")
            return None

    async def _fetch_candles(self, symbol: str):
        """Получение свечных данных"""
        async with self.client.session.get(
            url=f"{config.API_URL}/futures/usdt/candlesticks",
            headers=self._sign_request("GET", symbol),
            params={
                "contract": symbol,
                "interval": "5m",
                "limit": config.CANDLE_LIMIT
            }
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    def _sign_request(self, method: str, symbol: str):
        """Генерация подписи запроса"""
        timestamp = str(int(datetime.now(timezone.utc).timestamp()))
        path = f"/futures/usdt/candlesticks?contract={symbol}"
        message = f"{method}\n{path}\n\n{timestamp}"
        
        return {
            "KEY": config.API_KEY,
            "Timestamp": timestamp,
            "SIGN": hmac.new(
                config.API_SECRET, 
                message.encode(), 
                hashlib.sha512
            ).hexdigest()
        }

    def _analyze_candles(self, symbol: str, candles: list):
        """Анализ свечных данных"""
        closes = [float(c['c']) for c in candles]
        adf_stat, p_value = adfuller(closes, maxlag=1)
        
        return {
            "symbol": symbol,
            "adf_stat": adf_stat,
            "p_value": p_value,
            "candles": json.dumps(candles)
        }

class ClusterAnalyzer:
    """Анализатор кластеров символов"""
    def __init__(self):
        self.model = KMeans(n_clusters=config.CLUSTER_NUM)
        self.features = None

    async def load_data(self, db: DatabaseManager):
        """Загрузка данных для кластеризации"""
        query = "SELECT symbol, volume_24h_usd, adf_statistic FROM contract_details"
        self.features = await db.execute_query(query)

    def generate_pairs(self):
        """Генерация пар для тестирования коинтеграции"""
        if self.features is None:
            raise ValueError("Данные для кластеризации не загружены")
            
        clusters = self.model.fit_predict(self.features[['volume_24h_usd', 'adf_statistic']])
        pairs = []
        
        for cluster_id in np.unique(clusters):
            symbols = self.features[clusters == cluster_id]['symbol']
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    pairs.append((symbols.iloc[i], symbols.iloc[j]))
        
        return pairs

@asynccontextmanager
async def api_client():
    """Контекстный менеджер для API-клиента"""
    async with aiohttp.ClientSession() as session:
        yield session

async def main_cycle():
    """Основной рабочий цикл приложения"""
    db = DatabaseManager()
    cluster_analyzer = ClusterAnalyzer()
    
    async with api_client() as client:
        processor = SymbolProcessor(client)
        
        while not shutdown_event.is_set():
            start_time = datetime.now(timezone.utc)
            
            # Загрузка активных символов
            symbols = await db.get_active_symbols()
            
            # Параллельная обработка символов
            tasks = [processor.process(sym) for sym in symbols]
            results = await asyncio.gather(*tasks)
            
            # Сохранение результатов
            valid_results = [r for r in results if r]
            if valid_results:
                await db.save_data(pd.DataFrame(valid_results), "contract_details")
                logger.info(f"Сохранено {len(valid_results)} записей")
            
            # Анализ кластеров
            await cluster_analyzer.load_data(db)
            pairs = cluster_analyzer.generate_pairs()
            
            # Тестирование коинтеграции
            logger.info(f"Начато тестирование {len(pairs)} пар")
            # Здесь должна быть реализация тестирования
            
            # Управление временем цикла
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            await asyncio.sleep(max(300 - elapsed, 5))

def signal_handler(sig, frame):
    """Обработчик сигналов завершения"""
    logger.info("Получен сигнал завершения работы")
    shutdown_event.set()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(main_cycle())
    except KeyboardInterrupt:
        logger.info("Работа приложения завершена пользователем")
    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}")