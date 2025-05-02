from datetime import datetime


def validate_date(date_str: str) -> bool:
    """Проверяет, что строка соответствует формату YYYY-MM-DD"""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False