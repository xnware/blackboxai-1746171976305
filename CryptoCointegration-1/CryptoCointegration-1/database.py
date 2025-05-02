import logging
import pandas as pd
import sqlalchemy as sa
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

from config import config

# Настройка логгера для модуля работы с базой данных
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Менеджер для операций с базой данных
    
    Класс предоставляет методы для инициализации таблиц, сохранения и загрузки данных
    о символах, свечах, результатах коинтеграции и спредах.
    """
    
    def __init__(self):
        self.engine = config.db.engine
        self._init_tables()
    
    def _init_tables(self):
        """Инициализация таблиц в базе данных, если они ещё не существуют"""
        try:
            metadata = sa.MetaData()
            
            # Таблица символов торговых пар
            sa.Table(
                'symbols', metadata,
                sa.Column('symbol', sa.String, primary_key=True),
                sa.Column('base_currency', sa.String),
                sa.Column('quote_currency', sa.String),
                sa.Column('volume_24h_usd', sa.Float),
                sa.Column('last_price', sa.Float),
                sa.Column('active', sa.Boolean, default=True),
                sa.Column('last_updated', sa.DateTime(timezone=True))
            )
            
            # Таблица свечных данных
            sa.Table(
                'candles', metadata,
                sa.Column('id', sa.Integer, primary_key=True),
                sa.Column('symbol', sa.String),
                sa.Column('timestamp', sa.DateTime(timezone=True)),
                sa.Column('open', sa.Float),
                sa.Column('high', sa.Float),
                sa.Column('low', sa.Float),
                sa.Column('close', sa.Float),
                sa.Column('volume', sa.Float),
                sa.Column('interval', sa.String),
                sa.UniqueConstraint('symbol', 'timestamp', 'interval', name='uix_candle')
            )
            
            # Таблица деталей контрактов
            sa.Table(
                'contract_details', metadata,
                sa.Column('id', sa.Integer, primary_key=True),
                sa.Column('symbol', sa.String),
                sa.Column('adf_statistic', sa.Float),
                sa.Column('p_value', sa.Float),
                sa.Column('volume_24h_usd', sa.Float),
                sa.Column('candles_json', sa.Text),
                sa.Column('interval', sa.String),
                sa.Column('last_updated', sa.DateTime(timezone=True))
            )
            
            # Таблица результатов коинтеграции
            sa.Table(
                'cointegration_results', metadata,
                sa.Column('id', sa.Integer, primary_key=True),
                sa.Column('symbol1', sa.String),
                sa.Column('symbol2', sa.String),
                sa.Column('coint_t_stat', sa.Float),
                sa.Column('p_value', sa.Float),
                sa.Column('is_cointegrated', sa.Boolean),
                sa.Column('critical_values', sa.Text),  # Хранится в формате JSON
                sa.Column('hedge_ratio', sa.Float),
                sa.Column('half_life', sa.Float),
                sa.Column('interval', sa.String),
                sa.Column('created_at', sa.DateTime(timezone=True)),
                sa.Column('valid_until', sa.DateTime(timezone=True)),
                sa.UniqueConstraint('symbol1', 'symbol2', 'interval', name='uix_pair_interval')
            )
            
            # Таблица данных по спредам
            sa.Table(
                'spread_data', metadata,
                sa.Column('id', sa.Integer, primary_key=True),
                sa.Column('symbol1', sa.String),
                sa.Column('symbol2', sa.String),
                sa.Column('timestamp', sa.DateTime(timezone=True)),
                sa.Column('price1', sa.Float),
                sa.Column('price2', sa.Float),
                sa.Column('spread', sa.Float),
                sa.Column('z_score', sa.Float),
                sa.Column('interval', sa.String),
                sa.UniqueConstraint('symbol1', 'symbol2', 'timestamp', 'interval', name='uix_spread')
            )
            
            # Создаём все таблицы, если они ещё не существуют
            metadata.create_all(self.engine, checkfirst=True)
            logger.info("Таблицы базы данных инициализированы")
            
        except Exception as e:
            logger.error(f"Ошибка при инициализации таблиц базы данных: {str(e)}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для подключений к базе данных
        
        Используется для автоматического открытия и закрытия соединений в блоках with
        """
        with config.db.connection() as conn:
            yield conn
    
    def get_active_symbols(self) -> List[str]:
        """Получение активных торговых символов
        
        Возвращает список символов с объемом выше минимального порога,
        отсортированных по убыванию объема торгов.
        """
        try:
            query = """
                SELECT symbol FROM symbols 
                WHERE active = true AND volume_24h_usd > :min_volume
                ORDER BY volume_24h_usd DESC
            """
            with self.get_connection() as conn:
                result = conn.execute(
                    sa.text(query), 
                    {"min_volume": config.analysis.volume_threshold_usdt}
                ).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Ошибка при получении активных символов: {str(e)}")
            return []

    def load_candles(self, symbol: str, interval: str) -> pd.DataFrame:
        """Загрузка свечных данных для символа и интервала"""
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume FROM candles
                WHERE symbol = :symbol AND interval = :interval
                ORDER BY timestamp
            """
            with self.get_connection() as conn:
                df = pd.read_sql(
                    sa.text(query),
                    conn,
                    params={"symbol": symbol, "interval": interval}
                )
            return df
        except Exception as e:
            logger.error(f"Ошибка при загрузке свечей для {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def save_symbols(self, symbols_data: List[Dict[str, Any]]) -> int:
        """Save symbols data to database"""
        try:
            if not symbols_data:
                return 0
                
            now = datetime.now(timezone.utc)
            
            # Create DataFrame for batch insert
            df = pd.DataFrame(symbols_data)
            df['last_updated'] = now
            
            # Convert DataFrame to list of dicts for batch insert
            records = df.to_dict(orient='records')
            
            with self.get_connection() as conn:
                # Use batch insert with ON CONFLICT DO UPDATE for upsert
                insert_stmt = sa.text("""
                    INSERT INTO symbols (symbol, base_currency, quote_currency, volume_24h_usd, last_price, active, last_updated)
                    VALUES (:symbol, :base_currency, :quote_currency, :volume_24h_usd, :last_price, :active, :last_updated)
                    ON CONFLICT (symbol) DO UPDATE SET
                        volume_24h_usd = EXCLUDED.volume_24h_usd,
                        last_price = EXCLUDED.last_price,
                        active = EXCLUDED.active,
                        last_updated = EXCLUDED.last_updated
                """)
                conn.execute(insert_stmt, records)
                
            return len(symbols_data)
        except Exception as e:
            logger.error(f"Failed to save symbols: {str(e)}")
            return 0
    
    def save_candles(self, symbol: str, candles: List[Dict], interval: str) -> int:
        """Save candle data to database"""
        try:
            if not candles:
                return 0
                
            # Transform candles to list of dicts for batch insert
            data = []
            for candle in candles:
                timestamp = datetime.fromtimestamp(int(candle['t']), timezone.utc)
                data.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': float(candle['o']),
                    'high': float(candle['h']),
                    'low': float(candle['l']),
                    'close': float(candle['c']),
                    'volume': float(candle['v']),
                    'interval': interval
                })
            
            with self.get_connection() as conn:
                insert_stmt = sa.text("""
                    INSERT INTO candles (symbol, timestamp, open, high, low, close, volume, interval)
                    VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume, :interval)
                    ON CONFLICT (symbol, timestamp, interval) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """)
                conn.execute(insert_stmt, data)
                
            return len(data)
        except Exception as e:
            logger.error(f"Failed to save candles for {symbol}: {str(e)}")
            return 0
    
    def save_contract_details(self, details: Dict[str, Any]) -> bool:
        """Save contract analysis details"""
        try:
            now = datetime.now(timezone.utc)
            with self.get_connection() as conn:
                # Check if entry exists
                result = conn.execute(sa.text("""
                    SELECT id FROM contract_details 
                    WHERE symbol = :symbol AND interval = :interval
                """), {
                    "symbol": details["symbol"],
                    "interval": config.analysis.candle_interval
                }).fetchone()
                
                if result:
                    # Update existing
                    conn.execute(sa.text("""
                        UPDATE contract_details
                        SET adf_statistic = :adf_stat,
                            p_value = :p_value,
                            volume_24h_usd = :volume,
                            candles_json = :candles,
                            last_updated = :updated
                        WHERE id = :id
                    """), {
                        "id": result[0],
                        "adf_stat": details["adf_stat"],
                        "p_value": details["p_value"],
                        "volume": details.get("volume_24h_usd", 0),
                        "candles": details["candles"],
                        "updated": now
                    })
                else:
                    # Insert new
                    conn.execute(sa.text("""
                        INSERT INTO contract_details 
                        (symbol, adf_statistic, p_value, volume_24h_usd, candles_json, interval, last_updated)
                        VALUES (:symbol, :adf_stat, :p_value, :volume, :candles, :interval, :updated)
                    """), {
                        "symbol": details["symbol"],
                        "adf_stat": details["adf_stat"],
                        "p_value": details["p_value"],
                        "volume": details.get("volume_24h_usd", 0),
                        "candles": details["candles"],
                        "interval": config.analysis.candle_interval,
                        "updated": now
                    })
                    
            return True
        except Exception as e:
            logger.error(f"Failed to save contract details for {details.get('symbol')}: {str(e)}")
            return False
    
    def save_cointegration_result(self, result: Dict[str, Any]) -> bool:
        """Save cointegration test results"""
        try:
            now = datetime.now(timezone.utc)
            with self.get_connection() as conn:
                # Check if entry exists
                existing = conn.execute(sa.text("""
                    SELECT id FROM cointegration_results 
                    WHERE symbol1 = :s1 AND symbol2 = :s2 AND interval = :interval
                """), {
                    "s1": result["symbol1"],
                    "s2": result["symbol2"],
                    "interval": result["interval"]
                }).fetchone()
                
                if existing:
                    # Update existing
                    conn.execute(sa.text("""
                        UPDATE cointegration_results
                        SET coint_t_stat = :t_stat,
                            p_value = :p_value,
                            is_cointegrated = :is_cointegrated,
                            critical_values = :critical_values,
                            hedge_ratio = :hedge_ratio,
                            half_life = :half_life,
                            created_at = :created_at,
                            valid_until = :valid_until
                        WHERE id = :id
                    """), {
                        "id": existing[0],
                        "t_stat": result["coint_t_stat"],
                        "p_value": result["p_value"],
                        "is_cointegrated": result["is_cointegrated"],
                        "critical_values": result["critical_values"],
                        "hedge_ratio": result["hedge_ratio"],
                        "half_life": result.get("half_life", 0),
                        "created_at": now,
                        "valid_until": result["valid_until"]
                    })
                else:
                    # Insert new
                    conn.execute(sa.text("""
                        INSERT INTO cointegration_results 
                        (symbol1, symbol2, coint_t_stat, p_value, is_cointegrated, 
                         critical_values, hedge_ratio, half_life, interval, created_at, valid_until)
                        VALUES (:s1, :s2, :t_stat, :p_value, :is_cointegrated, 
                                :critical_values, :hedge_ratio, :half_life, :interval, :created_at, :valid_until)
                    """), {
                        "s1": result["symbol1"],
                        "s2": result["symbol2"],
                        "t_stat": result["coint_t_stat"],
                        "p_value": result["p_value"],
                        "is_cointegrated": result["is_cointegrated"],
                        "critical_values": result["critical_values"],
                        "hedge_ratio": result["hedge_ratio"],
                        "half_life": result.get("half_life", 0),
                        "interval": result["interval"],
                        "created_at": now,
                        "valid_until": result["valid_until"]
                    })
                    
            return True
        except Exception as e:
            logger.error(f"Failed to save cointegration result: {str(e)}")
            return False
    
    def save_spread_data(self, data: Dict[str, Any]) -> bool:
        """Save spread data point"""
        try:
            with self.get_connection() as conn:
                conn.execute(sa.text("""
                    INSERT INTO spread_data
                    (symbol1, symbol2, timestamp, price1, price2, spread, z_score, interval)
                    VALUES (:s1, :s2, :timestamp, :p1, :p2, :spread, :z_score, :interval)
                    ON CONFLICT (symbol1, symbol2, timestamp, interval) DO UPDATE
                    SET price1 = EXCLUDED.price1,
                        price2 = EXCLUDED.price2,
                        spread = EXCLUDED.spread,
                        z_score = EXCLUDED.z_score
                """), {
                    "s1": data["symbol1"],
                    "s2": data["symbol2"],
                    "timestamp": data["timestamp"],
                    "p1": data["price1"],
                    "p2": data["price2"],
                    "spread": data["spread"],
                    "z_score": data["z_score"],
                    "interval": data["interval"]
                })
            return True
        except Exception as e:
            logger.error(f"Failed to save spread data: {str(e)}")
            return False
    
    def load_cointegrated_pairs(self, limit: int = 100) -> pd.DataFrame:
        """Load cointegrated pairs from database"""
        try:
            now = datetime.now(timezone.utc)
            query = """
                SELECT
                    cr.symbol1,
                    cr.symbol2,
                    cr.coint_t_stat,
                    cr.p_value,
                    cr.hedge_ratio,
                    cr.half_life,
                    s1.volume_24h_usd as volume1,
                    s2.volume_24h_usd as volume2
                FROM cointegration_results cr
                JOIN symbols s1 ON cr.symbol1 = s1.symbol
                JOIN symbols s2 ON cr.symbol2 = s2.symbol
                WHERE 
                    cr.is_cointegrated = true AND
                    cr.valid_until > :now
                ORDER BY cr.p_value ASC
                LIMIT :limit
            """
            with self.get_connection() as conn:
                df = pd.read_sql(
                    sa.text(query),
                    conn,
                    params={"now": now, "limit": limit}
                )
            return df
        except Exception as e:
            logger.error(f"Failed to load cointegrated pairs: {str(e)}")
            return pd.DataFrame()
    
    def get_spread_data(self, symbol1: str, symbol2: str, start_date: str) -> pd.DataFrame:
        """Get spread data for a pair of symbols"""
        try:
            query = """
                SELECT 
                    sd.timestamp,
                    sd.price1,
                    sd.price2,
                    sd.spread,
                    sd.z_score,
                    (sd.price1 / LAG(sd.price1) OVER (ORDER BY sd.timestamp) - 1) * 100 as pc1,
                    (sd.price2 / LAG(sd.price2) OVER (ORDER BY sd.timestamp) - 1) * 100 as pc2
                FROM spread_data sd
                WHERE 
                    sd.symbol1 = :s1 AND 
                    sd.symbol2 = :s2 AND
                    sd.timestamp >= :start_date
                ORDER BY sd.timestamp
                LIMIT :limit
            """
            with self.get_connection() as conn:
                df = pd.read_sql(
                    sa.text(query),
                    conn,
                    params={
                        "s1": symbol1,
                        "s2": symbol2,
                        "start_date": start_date,
                        "limit": config.analysis.spread_data_limit
                    }
                )
            return df
        except Exception as e:
            logger.error(f"Failed to get spread data for {symbol1}/{symbol2}: {str(e)}")
            return pd.DataFrame()
