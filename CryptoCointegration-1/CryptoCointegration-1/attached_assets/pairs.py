from database import get_cointegrated_pairs, get_spread_series_from_contracts
from indicators import compute_indicators
from database import DatabaseManager

def get_cointegrated_pairs():
    db = DatabaseManager()
    return db.load_cointegrated_pairs()  # Предполагаем метод в DatabaseManager

def get_spread_series_from_contracts(s1, s2, date):
    db = DatabaseManager()
    return db.get_spread_data(s1, s2, date)

def fetch_pairs(start_date: str, num_pairs: int, sort_by: str, periods: int = 100):
    pairs_df = get_cointegrated_pairs()
    out = []

    for _, row in pairs_df.iterrows():
        s1, s2 = row.symbol1, row.symbol2
        df = get_spread_series_from_contracts(s1, s2, start_date)

        if df.empty or len(df) < 2:
            continue

        # Рассчитываем индикаторы
        # spread_changes теперь будет разницей между % изменениями цен активов (pc1 и pc2)
        spread_changes = (abs(df["pc1"]) - abs(df["pc2"])).tolist()
        
        # Остальные индикаторы рассчитываем как прежде
        pct_changes, _, z_scores = compute_indicators(df["spread"])
        
        if pct_changes is None:
            continue

        out.append({
            "symbols": (s1, s2),
            "timestamps": df["timestamp"].astype(str).tolist(),
            "price_changes": {
                s1: df["pc1"].tolist(),
                s2: df["pc2"].tolist(),
            },
            "spread": df["spread"].tolist(),
            "pct_changes": pct_changes,
            "spread_changes": spread_changes,  # Теперь это разница между % изменениями цен
            "zscore": z_scores,
        })

    # Сортировка
    if sort_by == "z_score":
        key = lambda x: abs(x["zscore"][-1]) if x["zscore"] and len(x["zscore"]) > 0 else 0
    else:  # "pct_change"
        key = lambda x: abs(x["spread"][-1]) if x["spread"] and len(x["spread"]) > 0 else 0
    
    out = sorted(out, key=key, reverse=True)[:num_pairs]
    return out