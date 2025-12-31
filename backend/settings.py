# backend/settings.py
"""
ThermoSense Configuration - Production Ready
"""
import os
from pathlib import Path
from typing import Optional

# ========== CORE SETTINGS ==========
APP_NAME = "THERMOSENSE"
VERSION = "2.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# ========== PATHS ==========
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = Path(__file__).resolve().parent

# Assets
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

KNOWLEDGE_DIR = ASSETS_DIR / "knowledge"
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

PRIOR_ERS_DIR = ASSETS_DIR / "prior_ers"
PRIOR_ERS_DIR.mkdir(parents=True, exist_ok=True)

# Templates
DEFAULT_TEMPLATE_NAME = "template.docx"
DEFAULT_TEMPLATE_PATH = ASSETS_DIR / DEFAULT_TEMPLATE_NAME

# Workspace (session storage)
WORKSPACE_ROOT = Path("./workspace").resolve()
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

UPLOADS_SUBDIR = "uploads"
OUTPUTS_SUBDIR = "outputs"
MEMORY_FILE = "memory.json"

# Charts
CHART_OUTPUT_DIR = WORKSPACE_ROOT / "charts"
CHART_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Static files for frontend
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ========== LLM SETTINGS ==========
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "30"))
SUMMARY_TIMEOUT_SEC = float(os.getenv("SUMMARY_TIMEOUT_SEC", "60"))
MAX_LLM_RETRIES = 3
LLM_TEMPERATURE = 0.7

# ========== CHAT SETTINGS ==========
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "20"))
MAX_MESSAGE_LENGTH = 5000
SESSION_TIMEOUT_HOURS = 24
AUTO_CLEANUP_ENABLED = True

# ========== FILE UPLOAD SETTINGS ==========
MAX_UPLOAD_SIZE_MB = 50
ALLOWED_EXTENSIONS = {
    'excel': ['.xlsx', '.xls', '.xlsm'],
    'csv': ['.csv'],
    'powerpoint': ['.ppt', '.pptx'],
    'pdf': ['.pdf'],
    'text': ['.txt', '.md']
}

EXCEL_HEADER_SCAN_ROWS = 15

# Flatten for validation
ALLOWED_EXTENSIONS_FLAT = set()
for exts in ALLOWED_EXTENSIONS.values():
    ALLOWED_EXTENSIONS_FLAT.update(exts)

# ========== CHART SETTINGS ==========
CHART_DPI = 300
CHART_DEFAULT_WIDTH_INCHES = 6.0
CHART_DEFAULT_HEIGHT_INCHES = 4.0
CHART_FORMATS = ['png', 'jpg', 'svg']
DEFAULT_CHART_FORMAT = 'png'

# ========== DATABASE & CACHING ==========
RAG_CACHE_FILE = "rag_metadata.json"
FEEDBACK_DB_FILE = WORKSPACE_ROOT / "feedback.json"
CHROMA_PERSIST_DIR = WORKSPACE_ROOT / "chroma_db"
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# ========== SECURITY ==========
ENABLE_CORS = True
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
API_KEY_REQUIRED = os.getenv("API_KEY_REQUIRED", "False").lower() == "true"
API_KEY = os.getenv("API_KEY", None)

# ========== TELEMETRY ==========
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_OPTOUT"] = "True"

# ========== POWERPOINT SETTINGS ==========
PPT_INCLUDE_TEXT_DEFAULT = False
PPT_IMAGE_WIDTH_INCH = 6.0
PPT_MAX_SLIDES = 100

# ========== PERFORMANCE ==========
FAST_MODE = os.getenv("FAST_MODE", "False").lower() == "true"
MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT_SEC = 300

# ========== LOGGING ==========
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = WORKSPACE_ROOT / "thermosense.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ========== FEATURE FLAGS ==========
ENABLE_FEEDBACK = True
ENABLE_RAG = True
ENABLE_CHART_GENERATION = True
ENABLE_PPT_EXTRACTION = True
ENABLE_COMPARISON = True
ENABLE_ANOMALY_DETECTION = True

# ========== HELPER FUNCTIONS ==========
def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS_FLAT)

def get_session_dir(session_id: str) -> Path:
    """Get session directory path"""
    session_dir = WORKSPACE_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir

def get_file_type(filename: str) -> Optional[str]:
    """Get file type category"""
    filename_lower = filename.lower()
    for file_type, extensions in ALLOWED_EXTENSIONS.items():
        if any(filename_lower.endswith(ext) for ext in extensions):
            return file_type
    return None

def validate_config():
    """Validate configuration on startup"""
    issues = []
    
    # Check critical directories
    for dir_path in [ASSETS_DIR, WORKSPACE_ROOT, KNOWLEDGE_DIR, STATIC_DIR]:
        if not dir_path.exists():
            issues.append(f"Missing directory: {dir_path}")
    
    # Check LLM connection
    if not OLLAMA_BASE_URL:
        issues.append("OLLAMA_BASE_URL not configured")
    
    # Check file size limits
    if MAX_UPLOAD_SIZE_MB > 100:
        issues.append(f"MAX_UPLOAD_SIZE_MB too large: {MAX_UPLOAD_SIZE_MB}")
    
    if issues:
        print("Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("Configuration validated successfully")
    return True

# ========== ENVIRONMENT INFO ==========
def print_config_info():
    """Print configuration summary"""
    print(f"""
==========================================
     {APP_NAME} v{VERSION}                 
==========================================

Directories:
   - Assets:     {ASSETS_DIR}
   - Workspace:  {WORKSPACE_ROOT}
   - Knowledge:  {KNOWLEDGE_DIR}

LLM:
   - Model:      {OLLAMA_MODEL}
   - URL:        {OLLAMA_BASE_URL}
   - Timeout:    {LLM_TIMEOUT_SEC}s

Chat:
   - Max Context: {MAX_CONTEXT_MESSAGES} messages
   - Timeout:     {SESSION_TIMEOUT_HOURS}h

Features:
   - RAG:         {'[ON]' if ENABLE_RAG else '[OFF]'}
   - Charts:      {'[ON]' if ENABLE_CHART_GENERATION else '[OFF]'}
   - Feedback:    {'[ON]' if ENABLE_FEEDBACK else '[OFF]'}
   - PPT Extract: {'[ON]' if ENABLE_PPT_EXTRACTION else '[OFF]'}

Mode: {'FAST' if FAST_MODE else 'NORMAL'} | Debug: {'ON' if DEBUG else 'OFF'}
""")

# Auto-validate on import
if __name__ != "__main__":
    validate_config()
