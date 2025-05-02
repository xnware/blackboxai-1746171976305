import logging
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta

from database import DatabaseManager
from indicators import compute_indicators, calculate_hedge_ratio, calculate_half_life
from config import config
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Логгер для модуля работы с торговыми парами
logger = logging.getLogger(__name__)

def fetch_pairs(start_date: str, num_pairs: int, sort_by: str, periods: int = 100) -> List[Dict[str, Any]]:
    """
    Загружает коинтегрированные пары и их данные
    
    Функция получает из базы данных список коинтегрированных пар,
    загружает данные по спредам и рассчитывает индикаторы.
    
    Аргументы:
        start_date: Начальная дата для данных (формат YYYY-MM-DD)
        num_pairs: Количество пар для возврата
        sort_by: Метод сортировки (z_score, pct_change, volume)
        periods: Максимальное количество точек данных на пару
        
    Возвращает:
        Список данных по парам для визуализации и анализа
    """
    try:
        # Инициализация подключения к базе данных
        db = DatabaseManager()
        
        # Получение коинтегрированных пар
        pairs_df = db.load_cointegrated_pairs(limit=100)  # Загружаем больше пар для фильтрации
        
        if pairs_df.empty:
            logger.warning("В базе данных не найдено коинтегрированных пар")
            return []
        
        # Обработка каждой пары
        out = []
        
        for _, row in pairs_df.iterrows():
            s1, s2 = row.symbol1, row.symbol2
            
            # Получение данных по спредам
            df = db.get_spread_data(s1, s2, start_date)
            
            if df.empty or len(df) < 2:
                logger.debug(f"Недостаточно данных по спредам для пары {s1}/{s2}")
                continue
            
            # Расчёт индикаторов
            # spread_changes - разница между процентными изменениями активов
            spread_changes = (abs(df["pc1"]) - abs(df["pc2"])).tolist()
            
            # Другие индикаторы на основе временного ряда спредов
            pct_changes, _, z_scores = compute_indicators(df["spread"])
            
            if pct_changes is None:
                logger.debug(f"Не удалось рассчитать индикаторы для пары {s1}/{s2}")
                continue
            
            # Создание структуры данных по паре
            pair_data = {
                "symbols": (s1, s2),
                "timestamps": df["timestamp"].astype(str).tolist(),
                "price_changes": {
                    s1: df["pc1"].fillna(0).tolist(),
                    s2: df["pc2"].fillna(0).tolist(),
                },
                "spread": df["spread"].tolist(),
                "pct_changes": pct_changes,
                "spread_changes": spread_changes,  # Разница между процентными изменениями
                "zscore": z_scores if z_scores else [0] * len(df),
                "hedge_ratio": float(row.hedge_ratio),
                "half_life": float(row.half_life),
                "volume": (float(row.volume1) + float(row.volume2)) / 2
            }
            
            out.append(pair_data)
        
        # Сортировка пар по выбранному критерию
        if sort_by == "z_score":
            key = lambda x: abs(x["zscore"][-1]) if x["zscore"] and len(x["zscore"]) > 0 else 0
        elif sort_by == "pct_change":
            key = lambda x: abs(x["pct_changes"][-1]) if x["pct_changes"] and len(x["pct_changes"]) > 0 else 0
        else:  # "volume"
            key = lambda x: x["volume"]
        
        out = sorted(out, key=key, reverse=True)[:num_pairs]
        logger.info(f"Загружено {len(out)} коинтегрированных пар")
        
        return out
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке пар: {str(e)}")
        return []

import concurrent.futures
import asyncio

def analyze_pair(db: DatabaseManager, s1: str, s2: str):
    """
    Анализ одной пары на коинтеграцию
    
    Выполняет:
    - Загрузку свечных данных
    - Проверку стационарности с помощью ADF теста
    - Вычисление коинтеграции с помощью теста Йохансена
    - Расчет коэффициента хеджирования и периода полураспада
    - Сохранение результатов в базу данных
    - Сохранение данных спреда
    """
    try:
        candles1 = db.load_candles(s1, config.analysis.candle_interval)
        candles2 = db.load_candles(s2, config.analysis.candle_interval)

        if candles1.empty or candles2.empty:
            logger.debug(f"Нет данных свечей для пары {s1}/{s2}")
            return

        df1 = candles1.set_index('timestamp')['close']
        df2 = candles2.set_index('timestamp')['close']
        df = pd.DataFrame({'s1': df1, 's2': df2}).dropna()

        if len(df) < config.analysis.min_data_points:
            logger.debug(f"Недостаточно данных для анализа пары {s1}/{s2}")
            return

        adf_result_s1 = adfuller(df['s1'])
        adf_result_s2 = adfuller(df['s2'])

        if adf_result_s1[1] > config.analysis.adf_pvalue_threshold and adf_result_s2[1] > config.analysis.adf_pvalue_threshold:
            johansen_result = coint_johansen(df, det_order=0, k_ar_diff=1)
            is_cointegrated = any(johansen_result.lr1 > johansen_result.cvt[:, 1])

            hedge_ratio = calculate_hedge_ratio(df['s1'], df['s2'])
            half_life = calculate_half_life(df['s1'] - hedge_ratio * df['s2'])

            result = {
                "symbol1": s1,
                "symbol2": s2,
                "coint_t_stat": johansen_result.lr1[0] if len(johansen_result.lr1) > 0 else 0,
                "p_value": adf_result_s1[1],
                "is_cointegrated": is_cointegrated,
                "critical_values": johansen_result.cvt.tolist(),
                "hedge_ratio": hedge_ratio,
                "half_life": half_life,
                "interval": config.analysis.candle_interval,
                "valid_until": datetime.now(timezone.utc) + timedelta(hours=6)
            }

            db.save_cointegration_result(result)

            spread_series = df['s1'] - hedge_ratio * df['s2']
            z_scores = compute_indicators(spread_series)[2]

            for timestamp, price1, price2, spread, z_score in zip(
                df.index, df['s1'], df['s2'], spread_series, z_scores
            ):
                spread_data = {
                    "symbol1": s1,
                    "symbol2": s2,
                    "timestamp": timestamp,
                    "price1": price1,
                    "price2": price2,
                    "spread": spread,
                    "z_score": z_score,
                    "interval": config.analysis.candle_interval
                }
                db.save_spread_data(spread_data)
        else:
            logger.debug(f"Пары {s1}/{s2} не коинтегрированы по ADF тесту")
    except Exception as e:
        logger.error(f"Ошибка анализа пары {s1}/{s2}: {e}")

async def analyze_cointegration_pairs():
    """
    Асинхронная функция для анализа коинтеграции пар
    
    Выполняет:
    - Получение активных символов
    - Вычисление коинтеграции для пар
    - Сохранение результатов в базу данных
    - Обновление данных спредов
    """
    logger.info("Начало анализа коинтеграции пар")
    db = DatabaseManager()

    try:
        active_symbols = db.get_active_symbols()
        logger.info(f"Найдено активных символов: {len(active_symbols)}")

        pairs_to_analyze = []
        n = len(active_symbols)
        for i in range(n):
            for j in range(i + 1, n):
                pairs_to_analyze.append((active_symbols[i], active_symbols[j]))

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            tasks = [
                loop.run_in_executor(executor, analyze_pair, db, s1, s2)
                for s1, s2 in pairs_to_analyze
            ]
            await asyncio.gather(*tasks)

        logger.info("Анализ коинтеграции завершен")
    except Exception as e:
        logger.error(f"Ошибка в анализе коинтеграции: {e}")
