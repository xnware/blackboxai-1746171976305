import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List, Any
from statsmodels.tsa.stattools import adfuller

# Настройка логирования для модуля индикаторов
logger = logging.getLogger(__name__)

import functools

@functools.cache
def compute_indicators(spread: pd.Series) -> Tuple[Optional[list], Optional[list], Optional[list]]:
    """
    Вычисляет производные показатели на основе временного ряда спреда.
    
    Возвращает кортеж из:
    1) Процентное изменение от начального значения
    2) Абсолютное изменение в процентах
    3) Z-показатель (стандартизированное отклонение) изменений
    
    Аргументы:
        spread: Временной ряд значений спреда (pandas.Series)
        
    Возвращает:
        Кортеж (pct_changes, delta_pct, z_scores) или (None, None, None) при ошибке
    """
    try:
        # Проверка входных данных на пустоту и достаточность
        if spread.dropna().empty or len(spread.dropna()) < 2:
            logger.warning("Недостаточно данных для расчета индикаторов")
            return None, None, None

        # 1. Расчет процентного изменения от начального значения
        base_value = spread.iloc[0]  # Базовое значение - первый элемент ряда
        if base_value == 0:
            logger.error("Нулевое начальное значение спреда")
            return None, None, None
            
        pct_changes = ((spread / base_value) - 1) * 100  # Расчет процентного изменения

        # 2. Расчет абсолютного изменения в процентах
        delta_pct = pct_changes.diff().fillna(0)  # Разница между соседними значениями

        # 3. Расчет Z-показателя (стандартизированного отклонения)
        mean_val = delta_pct.mean()  # Среднее значение
        std_val = delta_pct.std(ddof=0)  # Стандартное отклонение
        
        if std_val < 1e-8:  # Проверка на очень малое стандартное отклонение
            z_scores = pd.Series(0.0, index=delta_pct.index)  # Если стд очень мало, то возвращаем нули
        else:
            z_scores = (delta_pct - mean_val) / std_val  # Формула Z-показателя

        # Преобразование в стандартные Python-списки
        return (
            pct_changes.tolist(),
            delta_pct.tolist(),
            z_scores.tolist()
        )

    except (ZeroDivisionError, ValueError, TypeError) as e:
        # Обработка известных типов ошибок
        logger.error(f"Ошибка при расчете: {str(e)}", exc_info=True)
        return None, None, None
    except Exception as e:
        # Обработка непредвиденных ошибок
        logger.critical(f"Непредвиденная ошибка: {str(e)}", exc_info=True)
        return None, None, None

def calculate_volatility(series: pd.Series, window: int = 20) -> float:
    """
    Расчет скользящей волатильности (стандартного отклонения) временного ряда
    
    Функция определяет, является ли входной ряд ценами или доходностями, 
    и соответствующим образом преобразует данные перед расчетом волатильности.
    
    Аргументы:
        series: Временной ряд цен или доходностей
        window: Размер скользящего окна для расчета
        
    Возвращает:
        Годовая волатильность в процентах
    """
    try:
        # Определяем, является ли входной ряд ценами или доходностями
        if series.pct_change().std() < 0.1:  # Эвристика для определения типа ряда
            returns = series.pct_change().dropna()  # Если это цены, конвертируем в доходности
        else:
            returns = series.dropna()  # Если это уже доходности, просто убираем NaN значения
            
        # Расчёт скользящего стандартного отклонения
        vol = returns.rolling(window=window).std().dropna()  # Стандартное отклонение в окне
        
        # Берём последнее значение, если оно доступно
        if len(vol) > 0:
            # Аннуализируем в зависимости от типичного количества периодов
            # Для дневных данных умножаем на корень из 252 (рабочие дни в году)
            return float(vol.iloc[-1] * 100)  # Преобразуем в проценты
        return 0  # Если нет данных, возвращаем 0
        
    except Exception as e:
        # Обработка ошибок при расчете волатильности
        logger.error(f"Ошибка при расчете волатильности: {str(e)}")
        return 0

def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> Dict[str, List[float]]:
    """
    Расчёт полос Боллинджера для временного ряда
    
    Полосы Боллинджера представляют собой индикатор волатильности,
    основанный на скользящем среднем и стандартном отклонении.
    
    Аргументы:
        series: Временной ряд цен
        window: Размер скользящего окна
        num_std: Количество стандартных отклонений для полос
        
    Возвращает:
        Словарь с верхней, средней и нижней полосами
    """
    try:
        # Расчёт средней линии (скользящее среднее)
        middle_band = series.rolling(window=window).mean()
        # Расчёт стандартного отклонения для окна
        std_dev = series.rolling(window=window).std()
        
        # Верхняя полоса: среднее + (стандартное отклонение * коэффициент)
        upper_band = middle_band + (std_dev * num_std)
        # Нижняя полоса: среднее - (стандартное отклонение * коэффициент)
        lower_band = middle_band - (std_dev * num_std)
        
        # Формируем результирующий словарь с преобразованием в обычные списки
        return {
            "upper": upper_band.tolist(),  # Верхняя полоса
            "middle": middle_band.tolist(),  # Средняя полоса
            "lower": lower_band.tolist()  # Нижняя полоса
        }
        
    except Exception as e:
        # Обработка ошибок при расчете полос Боллинджера
        logger.error(f"Ошибка при расчете полос Боллинджера: {str(e)}")
        return {"upper": [], "middle": [], "lower": []}  # Возвращаем пустые списки при ошибке

def calculate_rsi(series: pd.Series, window: int = 14) -> List[float]:
    """
    Calculate Relative Strength Index
    
    Args:
        series: Price series
        window: RSI period
        
    Returns:
        List of RSI values
    """
    try:
        delta = series.diff().dropna()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=window).mean()
        avg_loss = losses.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.tolist()
        
    except Exception as e:
        logger.error(f"RSI calculation error: {str(e)}")
        return []

def calculate_correlation(series1: pd.Series, series2: pd.Series, window: int = 30) -> List[float]:
    """
    Calculate rolling correlation between two series
    
    Args:
        series1: First price series
        series2: Second price series
        window: Rolling window size
        
    Returns:
        List of correlation values
    """
    try:
        # Ensure both series are aligned
        df = pd.DataFrame({'s1': series1, 's2': series2})
        
        # Calculate rolling correlation
        corr = df['s1'].rolling(window=window).corr(df['s2'])
        
        return corr.tolist()
        
    except Exception as e:
        logger.error(f"Correlation calculation error: {str(e)}")
        return []

def calculate_momentum(series: pd.Series, period: int = 10) -> List[float]:
    """
    Calculate momentum indicator
    
    Args:
        series: Price series
        period: Momentum period
        
    Returns:
        List of momentum values
    """
    try:
        momentum = series / series.shift(period) - 1
        return momentum.tolist()
        
    except Exception as e:
        logger.error(f"Momentum calculation error: {str(e)}")
        return []

def augmented_hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst exponent to measure mean reversion or momentum
    H < 0.5: Mean-reverting
    H = 0.5: Random walk
    H > 0.5: Momentum (trend-following)
    
    Args:
        series: Price series
        max_lag: Maximum lag for calculation
        
    Returns:
        Hurst exponent value
    """
    try:
        # Convert to numpy array
        series = series.dropna().values
        
        if len(series) < max_lag * 2:
            logger.warning(f"Series too short for Hurst calculation: {len(series)} points")
            return 0.5  # Default to random walk
            
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        
        # Estimate Hurst as slope of log-log plot
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        
        return reg[0]  # Return slope
        
    except Exception as e:
        logger.error(f"Hurst exponent calculation error: {str(e)}")
        return 0.5

def calculate_hedge_ratio(series1: pd.Series, series2: pd.Series) -> float:
    """
    Calculate hedge ratio between two price series using OLS regression
    
    Args:
        series1: First price series (Y)
        series2: Second price series (X)
        
    Returns:
        Hedge ratio (coefficient)
    """
    try:
        # Make sure we have equal length series
        min_len = min(len(series1), len(series2))
        s1 = series1.iloc[-min_len:].reset_index(drop=True)
        s2 = series2.iloc[-min_len:].reset_index(drop=True)
        
        # Simple OLS: y = β*x + α
        # Hedge ratio is β
        df = pd.DataFrame({'y': s1, 'x': s2})
        df = df.dropna()
        
        if len(df) < 2:
            logger.warning(f"Insufficient data for hedge ratio calculation: {len(df)} points")
            return 1.0  # Default to 1:1 ratio
        
        # Calculate hedge ratio using OLS
        import statsmodels.api as sm
        X = sm.add_constant(df['x'])
        model = sm.OLS(df['y'], X).fit()
        
        hedge_ratio = model.params['x']
        return hedge_ratio
        
    except Exception as e:
        logger.error(f"Hedge ratio calculation error: {str(e)}")
        return 1.0

def calculate_half_life(spread: pd.Series) -> float:
    """
    Calculate half-life of mean reversion for a spread series
    
    Args:
        spread: Spread time series
        
    Returns:
        Half-life in periods
    """
    try:
        # Make sure spread is a pandas Series
        spread = pd.Series(spread).dropna()
        
        if len(spread) < 20:
            logger.warning(f"Spread series too short for half-life calculation: {len(spread)} points")
            return 0  
        
        # Calculate lag of spread
        lag_spread = spread.shift(1)
        # Calculate delta of spread
        delta_spread = spread - lag_spread
        # Remove NaN values
        lag_spread = lag_spread.dropna()
        delta_spread = delta_spread.dropna()
        
        # Create dataframe for OLS
        df = pd.DataFrame({'y': delta_spread, 'x': lag_spread})
        
        # Run OLS regression
        import statsmodels.api as sm
        model = sm.OLS(df['y'], df['x']).fit()
        
        # Extract coefficient
        coef = model.params['x']
        
        # Calculate half-life
        if coef < 0:
            half_life = -np.log(2) / coef
            return max(1.0, half_life)  # Ensure minimum half-life of 1 period
        else:
            # If coefficient is positive, there's no mean reversion
            return 0
        
    except Exception as e:
        logger.error(f"Half-life calculation error: {str(e)}")
        return 0
