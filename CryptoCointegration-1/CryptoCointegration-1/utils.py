from datetime import datetime, timezone
import re
import logging

logger = logging.getLogger(__name__)

def validate_date(date_str: str) -> bool:
    """
    Validates that a string matches the YYYY-MM-DD format
    
    Args:
        date_str: Date string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if not date_str:
            return False
            
        # Basic format validation with regex
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return False
            
        # Parse date to validate values
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def format_number(num: float) -> str:
    """
    Format a number with proper suffixes (K, M, B)
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string
    """
    try:
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.2f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.2f}K"
        else:
            return f"{num:.2f}"
    except (TypeError, ValueError) as e:
        logger.error(f"Error formatting number: {str(e)}")
        return str(num)

def get_current_time() -> str:
    """
    Get current time formatted as a string
    
    Returns:
        Formatted current time
    """
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

def parse_timeframe(timeframe: str) -> int:
    """
    Parse a timeframe string into minutes
    
    Args:
        timeframe: Timeframe string (e.g., '5m', '1h', '1d')
        
    Returns:
        Number of minutes
    """
    try:
        if not timeframe:
            return 5  # Default to 5 minutes
            
        match = re.match(r'^(\d+)([mhd])$', timeframe)
        if not match:
            return 5
            
        value, unit = match.groups()
        value = int(value)
        
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 60 * 24
        else:
            return 5
    except Exception as e:
        logger.error(f"Error parsing timeframe: {str(e)}")
        return 5
