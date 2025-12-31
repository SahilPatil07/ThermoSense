import json
import numpy as np
import pandas as pd
from typing import Any

class SafeJSONEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder that handles NaN, Inf, and -Inf by converting them to None.
    This ensures JSON compliance as standard JSON does not support these values.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (float, np.float32, np.float64)):
            if np.isnan(obj) or np.isinf(obj):
                return None
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super().default(obj)

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Serialize an object to a JSON formatted string safely.
    Replaces NaN/Inf with null.
    """
    # Pre-clean the object to handle basic float types that json.dumps handles directly
    cleaned_obj = clean_for_json(obj)
    
    # Ensure we use our custom encoder for other types (numpy, pandas, etc.)
    kwargs['cls'] = SafeJSONEncoder
    # Standard json.dumps with our encoder
    return json.dumps(cleaned_obj, **kwargs)

def clean_for_json(obj: Any) -> Any:
    """
    Recursively clean an object to be JSON compliant by replacing NaN/Inf with None.
    Useful for cases where you need the cleaned object rather than a JSON string.
    """
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray, pd.Series)):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (float, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif obj is None:
        return None
    # Check for scalar NA values only
    try:
        if pd.isna(obj) and not isinstance(obj, (np.ndarray, pd.Series)):
            return None
    except:
        pass
    return obj
