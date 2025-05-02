import logging
from typing import List
import pandas as pd
from database import DatabaseManager

logger = logging.getLogger(__name__)

class HistoricalDataFetcher:
    """Класс для получения исторических данных из базы данных"""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def get_active_symbols(self) -> List[str]:
        """Получить список активных торговых символов"""
        try:
            symbols = self.db.get_active_symbols()
            logger.info(f"Получено {len(symbols)} активных символов")
            return symbols
        except Exception as e:
            logger.error(f"Ошибка при получении активных символов: {str(e)}")
            return []
    
    def get_candles(self, symbol: str, interval: str) -> pd.DataFrame:
        """Получить исторические свечные данные для символа и интервала"""
        try:
            df = self.db.load_candles(symbol, interval)
            logger.info(f"Загружено {len(df)} свечей для {symbol} с интервалом {interval}")
            return df
        except Exception as e:
            logger.error(f"Ошибка при загрузке свечей для {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_spread_data(self, symbol1: str, symbol2: str, start_date: str) -> pd.DataFrame:
        """Получить исторические данные спреда для пары символов начиная с start_date"""
        try:
            df = self.db.get_spread_data(symbol1, symbol2, start_date)
            logger.info(f"Загружено {len(df)} записей спреда для пары {symbol1}/{symbol2}")
            return df
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных спреда для {symbol1}/{symbol2}: {str(e)}")
            return pd.DataFrame()
