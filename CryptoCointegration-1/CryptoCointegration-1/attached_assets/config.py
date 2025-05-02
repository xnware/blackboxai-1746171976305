import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from contextlib import contextmanager
from typing import Dict, Any
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Конфигурация подключения к базе данных"""
    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", 5432))
        self.user = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "")
        self.database = os.getenv("DB_NAME", "gate_io_data")
        self._engine = None

    @property
    def dsn(self) -> str:
        """Строка подключения DSN"""
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @property
    def engine(self):
        """Кэшированный экземпляр движка SQLAlchemy"""
        if not self._engine:
            self._engine = create_engine(
                self.dsn,
                pool_size=20,
                max_overflow=10,
                connect_args={"connect_timeout": 5}
            )
        return self._engine

    @contextmanager
    def connection(self):
        """Контекстный менеджер для подключения к БД"""
        conn = None
        try:
            conn = self.engine.connect()
            yield conn
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()

class AppConfig:
    """Конфигурация параметров приложения"""
    def __init__(self):
        self.cluster_num = int(os.getenv("CLUSTER_NUM", 10))
        self.default_pairs = int(os.getenv("DEFAULT_NUM_PAIRS", 10))
        self.candle_interval = os.getenv("CANDLE_INTERVAL", "5m")
        self.max_retries = int(os.getenv("MAX_RETRIES", 3))
        self.rate_limit = int(os.getenv("RATE_LIMIT", 100))
        self.modin_engine = os.getenv("MODIN_ENGINE", "ray")
        self.modin_partition_size = int(os.getenv("MODIN_PARTITION_SIZE", 10000))

    @property
    def modin_config(self) -> Dict[str, Any]:
        """Конфигурация Modin"""
        return {
            "partition_size": self.modin_partition_size,
            "engine": self.modin_engine
        }

class Config:
    """Главный класс конфигурации приложения"""
    def __init__(self):
        self.db = DatabaseConfig()
        self.app = AppConfig()

# Единый экземпляр конфигурации
config = Config()