import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from contextlib import contextmanager
from typing import Dict, Any, Optional
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database connection configuration"""
    def __init__(self):
        self.host = os.getenv("DB_HOST", os.getenv("PGHOST", "localhost"))
        self.port = int(os.getenv("DB_PORT", os.getenv("PGPORT", 5432)))
        self.user = os.getenv("DB_USER", os.getenv("PGUSER", "postgres"))
        self.password = os.getenv("DB_PASSWORD", os.getenv("PGPASSWORD", ""))
        self.database = os.getenv("DB_NAME", os.getenv("PGDATABASE", "gate_io_data"))
        self._engine = None

    @property
    def dsn(self) -> str:
        """Database connection string (DSN)"""
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @property
    def engine(self):
        """Cached SQLAlchemy engine instance"""
        if not self._engine:
            try:
                self._engine = create_engine(
                    self.dsn,
                    pool_size=20,
                    max_overflow=10,
                    connect_args={"connect_timeout": 5}
                )
                logger.info("Database engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize database engine: {str(e)}")
                raise
        return self._engine

    @contextmanager
    def connection(self):
        """Context manager for database connections"""
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

class APIConfig:
    """API configuration settings"""
    def __init__(self):
        self.api_url = os.getenv("API_URL", "https://api.gateio.ws/api/v4")
        self.api_key = os.getenv("API_KEY", "")
        self.api_secret = os.getenv("API_SECRET", "")
        # Rate limiting settings
        self.rate_limit = int(os.getenv("RATE_LIMIT", 100))
        self.rate_period = int(os.getenv("RATE_PERIOD", 10))
        # Request settings
        self.concurrent_requests = int(os.getenv("CONCURRENT_REQUESTS", 5))
        self.max_retries = int(os.getenv("MAX_RETRIES", 5))
        self.request_delay = int(os.getenv("REQUEST_DELAY", 3))

class AnalysisConfig:
    """Configuration for statistical analysis"""
    def __init__(self):
        # Analysis parameters
        self.volume_threshold_usdt = float(os.getenv("VOLUME_24H_THRESHOLD_USDT", 1000000))
        self.candle_limit = int(os.getenv("CANDLE_LIMIT", 1000))
        self.candle_interval = os.getenv("CANDLE_INTERVAL", "5m")
        self.p_value_threshold = float(os.getenv("COINTEGRATION_P_VALUE_THRESHOLD", 0.05))
        self.max_lags = int(os.getenv("COINTEGRATION_MAX_LAGS", 2))
        self.max_cointegration_records = int(os.getenv("MAX_COINTEGRATION_RECORDS", 1000))
        self.adf_min_obs = int(os.getenv("ADF_MIN_OBS", 12))
        self.adf_critical_value = float(os.getenv("ADF_CRITICAL_VALUE", -2.89))
        self.spread_data_limit = int(os.getenv("SPREAD_DATA_LIMIT", 50))
        self.cointegration_validity = os.getenv("COINTEGRATION_VALIDITY", "1 HOUR")
        self.min_timeframe_overlap = float(os.getenv("MIN_TIMEFRAME_OVERLAP", 0.95))
        self.interval_minutes = int(os.getenv("INTERVAL_MINUTES", 5))

class AppConfig:
    """Main application configuration"""
    def __init__(self):
        self.default_start_date = None  # Will be initialized in Config class
        self.default_num_pairs = 10
        self.default_sort_by = "z_score"  # Options: z_score, pct_change
        
class Config:
    """Main configuration class"""
    def __init__(self):
        self.db = DatabaseConfig()
        self.api = APIConfig()
        self.analysis = AnalysisConfig()
        self.app = AppConfig()
        
        # Set default start date if not provided
        self.DEFAULT_START_DATE = os.getenv("DEFAULT_START_DATE")
        self.DEFAULT_NUM_PAIRS = int(os.getenv("DEFAULT_NUM_PAIRS", 10))
        self.DEFAULT_SORT_BY = os.getenv("DEFAULT_SORT_BY", "z_score")
        
        # Initialize Modin settings
        os.environ["MODIN_ENGINE"] = os.getenv("MODIN_ENGINE", "ray")
        os.environ["MODIN_CPUS"] = os.getenv("MODIN_CPUS", "4")

# Create a singleton instance
config = Config()
