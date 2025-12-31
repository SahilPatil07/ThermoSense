import logging
import json
import datetime
import uuid
import threading
from typing import Any, Dict, Optional

# Thread-local storage for context (trace_id, session_id)
_context = threading.local()

def set_context(trace_id: str = None, session_id: str = None):
    _context.trace_id = trace_id or getattr(_context, 'trace_id', str(uuid.uuid4()))
    _context.session_id = session_id or getattr(_context, 'session_id', 'unknown')

def get_context() -> Dict[str, str]:
    return {
        "trace_id": getattr(_context, 'trace_id', 'none'),
        "session_id": getattr(_context, 'session_id', 'none')
    }

class JsonFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    def format(self, record: logging.LogRecord) -> str:
        context = get_context()
        log_data = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "trace_id": context["trace_id"],
            "session_id": context["session_id"]
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)
            
        return json.dumps(log_data)

def setup_logger(name: str = "thermosense"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        
    return logger

# Global logger instance
logger = setup_logger()
