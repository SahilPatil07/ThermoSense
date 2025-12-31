import pandas as pd
import logging
from pathlib import Path
from typing import Union, List, Optional

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Robust data loader for handling various file formats and ensuring data quality.
    """

    @staticmethod
    def load_file(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Load data from a file (CSV or Excel) with robust error handling.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        try:
            if file_path.suffix.lower() == '.csv':
                # Try different encodings
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='latin1')
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return None

            # Normalize columns
            df = DataLoader.normalize_columns(df)
            return df

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names: strip whitespace, convert to string.
        """
        if df is not None:
            df.columns = df.columns.astype(str).str.strip()
        return df

    @staticmethod
    def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Check if required columns exist in the DataFrame.
        """
        if df is None:
            return False
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return False
        return True
