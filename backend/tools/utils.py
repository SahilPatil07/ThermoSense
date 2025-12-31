import pandas as pd
import numpy as np
from typing import List, Optional

def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect time column"""
    time_keywords = ['time', 'timestamp', 'time_stamp', 'date', 'datetime', 'elapsed', 'tme']
    
    for col in df.columns:
        col_lower = str(col).lower().replace('_', '').replace(' ', '')
        if any(keyword in col_lower for keyword in time_keywords):
            return col
    
    return None

def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get all numeric columns"""
    return df.select_dtypes(include=[np.number]).columns.tolist()
