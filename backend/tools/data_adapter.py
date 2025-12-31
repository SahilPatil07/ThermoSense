"""
Data Adapter for unified Pandas/Polars interface
Automatically selects optimal backend based on file size and operations
"""
import logging
from pathlib import Path
from typing import Union, Optional, List, Any
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import Polars
try:
    import polars as pl
    POLARS_AVAILABLE = True
    logger.info("Polars is available for high-performance data processing")
except ImportError:
    POLARS_AVAILABLE = False
    logger.warning("Polars not available, falling back to Pandas only")

# File size threshold for using Polars (10MB)
POLARS_THRESHOLD_BYTES = 10 * 1024 * 1024


class DataAdapter:
    """
    Unified interface for data processing with automatic backend selection
    
    Automatically chooses between Pandas and Polars based on:
    - File size (use Polars for files > 10MB)
    - Operation type (some operations only available in Pandas)
    - User preference
    """
    
    def __init__(self, prefer_polars: bool = True):
        """
        Initialize data adapter
        
        Args:
            prefer_polars: Whether to prefer Polars when available
        """
        self.prefer_polars = prefer_polars and POLARS_AVAILABLE
        self.backend = None  # 'pandas' or 'polars'
        self.df = None
    
    @staticmethod
    def should_use_polars(file_path: Union[str, Path]) -> bool:
        """
        Determine if Polars should be used for a file
        
        Args:
            file_path: Path to data file
        
        Returns:
            True if Polars should be used, False otherwise
        """
        if not POLARS_AVAILABLE:
            return False
        
        file_path = Path(file_path)
        if not file_path.exists():
            return False
        
        file_size = file_path.stat().st_size
        return file_size > POLARS_THRESHOLD_BYTES
    
    def read_csv(
        self,
        file_path: Union[str, Path],
        force_backend: Optional[str] = None,
        **kwargs
    ) -> 'DataAdapter':
        """
        Read CSV file with automatic backend selection
        
        Args:
            file_path: Path to CSV file
            force_backend: Force specific backend ('pandas' or 'polars')
            **kwargs: Additional arguments passed to read function
        
        Returns:
            Self for method chaining
        """
        file_path = Path(file_path)
        
        # Determine backend
        if force_backend:
            self.backend = force_backend
        elif self.prefer_polars and self.should_use_polars(file_path):
            self.backend = 'polars'
        else:
            self.backend = 'pandas'
        
        logger.info(f"Reading CSV with {self.backend}: {file_path.name}")
        
        # Read data
        if self.backend == 'polars':
            self.df = pl.read_csv(file_path, **kwargs)
        else:
            self.df = pd.read_csv(file_path, **kwargs)
        
        return self
    
    def read_excel(
        self,
        file_path: Union[str, Path],
        force_backend: Optional[str] = None,
        **kwargs
    ) -> 'DataAdapter':
        """
        Read Excel file with automatic backend selection
        
        Args:
            file_path: Path to Excel file
            force_backend: Force specific backend ('pandas' or 'polars')
            **kwargs: Additional arguments passed to read function
        
        Returns:
            Self for method chaining
        """
        file_path = Path(file_path)
        
        # Determine backend
        if force_backend:
            self.backend = force_backend
        elif self.prefer_polars and self.should_use_polars(file_path):
            self.backend = 'polars'
        else:
            self.backend = 'pandas'
        
        logger.info(f"Reading Excel with {self.backend}: {file_path.name}")
        
        # Read data
        if self.backend == 'polars':
            self.df = pl.read_excel(file_path, **kwargs)
        else:
            self.df = pd.read_excel(file_path, **kwargs)
        
        return self
    
    def to_pandas(self) -> pd.DataFrame:
        """
        Convert to Pandas DataFrame
        
        Returns:
            Pandas DataFrame
        """
        if self.backend == 'polars':
            return self.df.to_pandas()
        return self.df
    
    def to_polars(self) -> 'pl.DataFrame':
        """
        Convert to Polars DataFrame
        
        Returns:
            Polars DataFrame (if available)
        """
        if not POLARS_AVAILABLE:
            raise RuntimeError("Polars is not available")
        
        if self.backend == 'pandas':
            return pl.from_pandas(self.df)
        return self.df
    
    def get_columns(self) -> List[str]:
        """Get column names"""
        if self.backend == 'polars':
            return self.df.columns
        return self.df.columns.tolist()
    
    def get_numeric_columns(self) -> List[str]:
        """Get numeric column names"""
        if self.backend == 'polars':
            return [col for col in self.df.columns if self.df[col].dtype in pl.NUMERIC_DTYPES]
        else:
            return self.df.select_dtypes(include=['number']).columns.tolist()
    
    def shape(self) -> tuple:
        """Get shape (rows, columns)"""
        return self.df.shape
    
    def head(self, n: int = 5):
        """Get first n rows"""
        return self.df.head(n)
    
    def describe(self):
        """Get statistical summary"""
        if self.backend == 'polars':
            return self.df.describe()
        return self.df.describe()
    
    def select(self, columns: List[str]) -> 'DataAdapter':
        """
        Select specific columns
        
        Args:
            columns: List of column names
        
        Returns:
            New DataAdapter with selected columns
        """
        adapter = DataAdapter(prefer_polars=self.prefer_polars)
        adapter.backend = self.backend
        
        if self.backend == 'polars':
            adapter.df = self.df.select(columns)
        else:
            adapter.df = self.df[columns]
        
        return adapter
    
    def filter(self, condition) -> 'DataAdapter':
        """
        Filter rows based on condition
        
        Args:
            condition: Filter condition
        
        Returns:
            New DataAdapter with filtered rows
        """
        adapter = DataAdapter(prefer_polars=self.prefer_polars)
        adapter.backend = self.backend
        
        if self.backend == 'polars':
            adapter.df = self.df.filter(condition)
        else:
            adapter.df = self.df[condition]
        
        return adapter
    
    def __len__(self) -> int:
        """Get number of rows"""
        return len(self.df)
    
    def __repr__(self) -> str:
        return f"DataAdapter(backend={self.backend}, shape={self.shape()})"


def read_data_file(
    file_path: Union[str, Path],
    prefer_polars: bool = True,
    **kwargs
) -> DataAdapter:
    """
    Convenience function to read data file with automatic format detection
    
    Args:
        file_path: Path to data file
        prefer_polars: Whether to prefer Polars backend
        **kwargs: Additional arguments passed to read function
    
    Returns:
        DataAdapter instance
    """
    file_path = Path(file_path)
    adapter = DataAdapter(prefer_polars=prefer_polars)
    
    if file_path.suffix.lower() == '.csv':
        return adapter.read_csv(file_path, **kwargs)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        return adapter.read_excel(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
