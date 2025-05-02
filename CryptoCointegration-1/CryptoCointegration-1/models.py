from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

@dataclass
class ContractDetails:
    """Details of a futures contract"""
    symbol: str
    volume_24h_usd: float
    adf_statistic: Optional[float] = None
    p_value: Optional[float] = None
    last_updated: Optional[datetime] = None

@dataclass
class CointegrationResult:
    """Results of a cointegration test between two assets"""
    symbol1: str
    symbol2: str
    p_value: float
    adf_statistic: float
    hedge_ratio: float
    half_life: Optional[float] = None
    is_cointegrated: bool = False

@dataclass
class SpreadDataPoint:
    """Single data point for a spread time series"""
    timestamp: datetime
    spread: float
    price1: float
    price2: float
    pc1: Optional[float] = None  # Percentage change asset 1
    pc2: Optional[float] = None  # Percentage change asset 2
    z_score: Optional[float] = None

@dataclass
class PairViewModel:
    """View model for displaying pair data in UI"""
    symbols: tuple
    timestamps: List[str]
    price_changes: Dict[str, List[float]]
    spread: List[float]
    pct_changes: List[float]
    spread_changes: List[float]
    zscore: List[float]
