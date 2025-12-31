import pandas as pd
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from thefuzz import fuzz, process

logger = logging.getLogger(__name__)

class SensorHarvester:
    def __init__(self, config_path: str = "backend/config/sensors.json"):
        self.config_path = Path(config_path)
        self.sensor_map = self._load_config()
        
    def _load_config(self) -> Dict[str, List[str]]:
        """Load sensor mapping configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load sensor config: {e}")
            return {}

    def get_file_columns(self, file_path: str) -> Tuple[List[str], Optional[str], List[str]]:
        """
        Robustly extract columns, time column, and numeric columns from a file.
        Handles transposed Excel files and different header rows.
        """
        path = Path(file_path)
        columns = []
        time_column = None
        numeric_columns = []
        
        try:
            if path.suffix.lower() == '.csv':
                df = pd.read_csv(path)
                columns = [c for c in df.columns.tolist() if not str(c).startswith("Unnamed:")]
            else:
                # Use read_excel_table for robust header detection
                from backend.tools.excel_tools import read_excel_table, list_sheetnames
                sheets = list_sheetnames(str(path))
                sheet_name = sheets[0]
                df = read_excel_table(str(path), sheet_name)
                columns = df.columns.tolist()
            
            # Auto-detect time and numeric
            from backend.tools.utils import detect_time_column, get_numeric_columns
            time_column = detect_time_column(df)
            numeric_columns = get_numeric_columns(df)
            
        except Exception as e:
            logger.error(f"Error in get_file_columns for {path.name}: {e}")
            
        return columns, time_column, numeric_columns

    def harvest_sensors(self, file_paths: List[str], required_sensors: Optional[List[str]] = None, strict_mode: bool = True, structured_output: bool = False, orchestrator: Any = None, sheet_name: Optional[str] = None) -> Tuple[Any, Dict[str, str]]:
        """
        Extract specific sensors from multiple Excel/CSV files and align them by Time.
        Returns: (Master DataFrame OR Dict of DataFrames, Metadata about source sheets)
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        logger.info(f"Harvesting from {len(file_paths)} files: {file_paths} (Strict: {strict_mode}, Structured: {structured_output})")
        
        # 1. Load all sheets and detect headers
        all_sheets = {}
        for fpath in file_paths:
            try:
                path = Path(fpath)
                if path.suffix.lower() == '.csv':
                    df = pd.read_csv(path)
                    all_sheets[f"{path.name}::Main"] = df
                else:
                    from backend.tools.excel_tools import read_excel_table, list_sheetnames
                    
                    # If sheet_name is provided, only process that sheet
                    xl_sheets = list_sheetnames(str(path))
                    target_sheets = [sheet_name] if sheet_name and sheet_name in xl_sheets else xl_sheets
                    
                    for s_name in target_sheets:
                        try:
                            df = read_excel_table(str(path), s_name)
                            # Drop completely empty columns/rows
                            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
                            
                            if not df.empty:
                                all_sheets[f"{path.name}::{s_name}"] = df
                                try:
                                    logger.info(f"Loaded {s_name} from {path.name} using read_excel_table")
                                except:
                                    pass # Don't let logging failure stop extraction
                        except Exception as e:
                            logger.warning(f"Skipping sheet '{s_name}' in {path.name}: {e}")
            except Exception as e:
                logger.error(f"Failed to load {fpath}: {e}")

        if not all_sheets:
            return ({} if structured_output else pd.DataFrame()), {}

        # 2. Find Sensors and Align
        target_sensors = required_sensors if required_sensors else list(self.sensor_map.keys())
        source_map = {}
        
        # Helper to find time column in a specific dataframe
        def find_time_col(df):
            # Priority: 1. Unique Time, 2. Unique Timestamp, 3. Unique Date, 4. Any Time/Date
            time_keywords = ['time', 'timestamp', 'date', 'elapsed', 'tme']
            cols = [str(c).lower().strip() for c in df.columns]
            
            # First pass: Look for unique columns matching keywords
            for k in time_keywords:
                for i, c in enumerate(cols):
                    if k in c:
                        actual_col = df.columns[i]
                        if not df[actual_col].duplicated().any():
                            return actual_col
            
            # Second pass: Any match
            for k in time_keywords:
                for i, c in enumerate(cols):
                    if k in c:
                        return df.columns[i]
            return None

        if structured_output:
            # Structured output: Return a dict of DataFrames, one per sheet
            structured_results = {}
            for sheet_id, df in all_sheets.items():
                sheet_df = pd.DataFrame()
                time_col = find_time_col(df)
                
                if time_col:
                    sheet_df['Time'] = df[time_col]
                else:
                    sheet_df['Time'] = df.index
                
                found_any = False
                for sensor in target_sensors:
                    sensor_clean = sensor.strip()
                    keywords = self.sensor_map.get(sensor_clean, [sensor_clean])
                    
                    df.columns = [str(c).strip() for c in df.columns]
                    cols_lower = [c.lower() for c in df.columns]
                    
                    matched_col = None
                    # Exact match
                    for kw in keywords:
                        if kw in df.columns:
                            matched_col = kw
                            break
                        if kw.lower() in cols_lower:
                            matched_col = df.columns[cols_lower.index(kw.lower())]
                            break
                    
                    # Fuzzy match if no exact match and not strict
                    if matched_col is None and not strict_mode:
                        choices = df.columns.tolist()
                        if choices:
                            match = process.extractOne(keywords[0], choices, scorer=fuzz.WRatio)
                            if match and match[1] >= 85:
                                matched_col = match[0]
                                logger.info(f"Fuzzy match in {sheet_id}: {keywords[0]} -> {matched_col} ({match[1]})")

                    # Intelligent match if still no match and orchestrator provided
                    if matched_col is None and not strict_mode and orchestrator:
                        intelligent_map = orchestrator.map_sensors_intelligently([sensor_clean], df.columns.tolist())
                        if sensor_clean in intelligent_map:
                            matched_col = intelligent_map[sensor_clean]
                            logger.info(f"Intelligent match in {sheet_id}: {sensor_clean} -> {matched_col}")

                    if matched_col:
                        # Try numeric conversion but keep original if it fails (for metadata)
                        val = df[matched_col]
                        num_val = pd.to_numeric(val, errors='coerce')
                        # If more than 50% are NaNs after conversion but weren't before, keep original
                        if num_val.isna().sum() > val.isna().sum() + (len(val) * 0.5):
                            sheet_df[sensor] = val
                        else:
                            sheet_df[sensor] = num_val
                        
                        source_map[sensor] = f"{sheet_id}::{matched_col}"
                        found_any = True
                
                if found_any:
                    # Drop rows where ALL sensors are NaN (keep Time)
                    sensor_cols = [s for s in target_sensors if s in sheet_df.columns]
                    try:
                        logger.info(f"Sheet {sheet_id}: found {len(sensor_cols)} sensors. Rows before dropna: {len(sheet_df)}")
                    except:
                        pass
                    
                    sheet_df = sheet_df.dropna(subset=sensor_cols, how='all')
                    
                    try:
                        logger.info(f"Sheet {sheet_id}: rows after dropna: {len(sheet_df)}")
                    except:
                        pass
                        
                    if not sheet_df.empty:
                        structured_results[sheet_id] = sheet_df
                    else:
                        try:
                            logger.warning(f"Sheet {sheet_id}: resulting DataFrame is empty after dropping NaNs")
                        except:
                            pass
                else:
                    try:
                        logger.warning(f"Sheet {sheet_id}: no sensors found from target list")
                    except:
                        pass
            
            logger.info(f"Structured extraction complete. Found data in {len(structured_results)} sheets.")
            return structured_results, source_map

        else:
            # Original behavior: Merge everything into one master_df
            master_df = None
            for sensor in target_sensors:
                sensor_clean = sensor.strip()
                keywords = self.sensor_map.get(sensor_clean, [sensor_clean])
                
                sensor_found = False
                for sheet_id, df in all_sheets.items():
                    df.columns = [str(c).strip() for c in df.columns]
                    cols_lower = [c.lower() for c in df.columns]
                    
                    matched_col = None
                    # Exact match
                    for kw in keywords:
                        if kw in df.columns:
                            matched_col = kw
                            break
                        if kw.lower() in cols_lower:
                            matched_col = df.columns[cols_lower.index(kw.lower())]
                            break
                    
                    # Fuzzy match if no exact match and not strict
                    if matched_col is None and not strict_mode:
                        choices = df.columns.tolist()
                        if choices:
                            match = process.extractOne(keywords[0], choices, scorer=fuzz.WRatio)
                            if match and match[1] >= 85:
                                matched_col = match[0]
                                logger.info(f"Fuzzy match in {sheet_id}: {keywords[0]} -> {matched_col} ({match[1]})")

                    # Intelligent match if still no match and orchestrator provided
                    if matched_col is None and not strict_mode and orchestrator:
                        intelligent_map = orchestrator.map_sensors_intelligently([sensor_clean], df.columns.tolist())
                        if sensor_clean in intelligent_map:
                            matched_col = intelligent_map[sensor_clean]
                            logger.info(f"Intelligent match in {sheet_id}: {sensor_clean} -> {matched_col}")

                    if matched_col:
                        sensor_data = df[matched_col]
                        time_col = find_time_col(df)
                        
                        temp_df = pd.DataFrame()
                        if time_col:
                            temp_df['Time'] = df[time_col]
                        else:
                            temp_df['Time'] = df.index
                        
                        # Try numeric conversion but keep original if it fails (for metadata)
                        val = sensor_data
                        num_val = pd.to_numeric(val, errors='coerce')
                        # If more than 50% are NaNs after conversion but weren't before, keep original
                        if num_val.isna().sum() > val.isna().sum() + (len(val) * 0.5):
                            temp_df[sensor] = val
                        else:
                            temp_df[sensor] = num_val
                        
                        temp_df = temp_df.dropna(subset=[sensor])
                        temp_df['row_idx'] = temp_df.groupby('Time').cumcount()

                        if not temp_df.empty:
                            sensor_found = True
                            if master_df is None:
                                master_df = temp_df
                            else:
                                if sensor in master_df.columns:
                                    master_df = pd.merge(master_df, temp_df, on=['Time', 'row_idx'], how='outer', suffixes=('', '_new'))
                                    if f"{sensor}_new" in master_df.columns:
                                        master_df[sensor] = master_df[sensor].fillna(master_df[f"{sensor}_new"])
                                        master_df = master_df.drop(columns=[f"{sensor}_new"])
                                else:
                                    master_df = pd.merge(master_df, temp_df, on=['Time', 'row_idx'], how='outer')
                            
                            source_map[sensor] = f"{sheet_id}::{matched_col}"
                            
                            if master_df is not None and master_df.size > 10_000_000:
                                logger.error("Master DataFrame exceeded safety size limit.")
                                break
                
                if not sensor_found:
                    source_map[sensor] = "NOT FOUND"

            if master_df is None or master_df.empty:
                return pd.DataFrame(), {}

            if 'row_idx' in master_df.columns:
                master_df = master_df.drop(columns=['row_idx'])

            if 'Time' in master_df.columns:
                cols = ['Time'] + [c for c in master_df.columns if c != 'Time']
                master_df = master_df[cols]
            
            logger.info(f"Extraction complete. Master DF shape: {master_df.shape}")
            return master_df, source_map

    def _find_time_column(self, sheets: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        """Legacy helper, kept for compatibility if needed elsewhere"""
        time_keywords = ['time', 'timestamp', 'date', 'elapsed', 'tme']
        for sheet_name, df in sheets.items():
            cols = [str(c).lower().strip() for c in df.columns]
            for k in time_keywords:
                if k in cols:
                    idx = cols.index(k)
                    return {'sheet': sheet_name, 'col': df.columns[idx], 'data': df[df.columns[idx]]}
        return None

    # Alias for backward compatibility
    def extract_sensors(self, *args, **kwargs):
        """Alias for harvest_sensors"""
        return self.harvest_sensors(*args, **kwargs)
