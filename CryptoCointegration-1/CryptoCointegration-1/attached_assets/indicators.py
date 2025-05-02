import modin.pandas as pd
from typing import Tuple, Optional
import logging

# Настройка логгера
logger = logging.getLogger(__name__)

def compute_indicators(spread: pd.Series) -> Tuple[Optional[list], Optional[list], Optional[list]]:
    """
    Вычисляет производные показатели на основе временного ряда спреда.
    
    Возвращает кортеж из:
    1) Процентное изменение от начального значения
    2) Абсолютное изменение процентов
    3) Z-показатель изменений
    
    Args:
        spread: Временной ряд значений спреда (modin.pandas.Series)
        
    Returns:
        Кортеж (pct_changes, delta_pct, z_scores) или (None, None, None) при ошибке
    """
    try:
        # Проверка входных данных
        if spread.dropna().empty or len(spread.dropna()) < 2:
            logger.warning("Недостаточно данных для вычисления индикаторов")
            return None, None, None

        # 1. Процентное изменение от начального значения
        base_value = spread.iloc[0]
        if base_value == 0:
            logger.error("Нулевое начальное значение спреда")
            return None, None, None
            
        pct_changes = ((spread / base_value) - 1) * 100

        # 2. Абсолютное изменение процентов
        delta_pct = pct_changes.diff().fillna(0)

        # 3. Расчет Z-показателя
        mean_val = delta_pct.mean()
        std_val = delta_pct.std(ddof=0)
        
        if std_val < 1e-8:
            z_scores = pd.Series(0.0, index=delta_pct.index)
        else:
            z_scores = (delta_pct - mean_val) / std_val

        # Конвертация в стандартные списки
        return (
            pct_changes.tolist(),
            delta_pct.tolist(),
            z_scores.tolist()
        )

    except (ZeroDivisionError, ValueError, TypeError) as e:
        logger.error(f"Ошибка вычислений: {str(e)}", exc_info=True)
        return None, None, None
    except Exception as e:
        logger.critical(f"Непредвиденная ошибка: {str(e)}", exc_info=True)
        return None, None, None