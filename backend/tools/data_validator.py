import pandas as pd
import pandera as pa
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, config_path: str = "backend/config/limits.yaml"):
        self.config_path = Path(config_path)
        self.limits = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            return {}
        except Exception as e:
            logger.error(f"Failed to load limits config: {e}")
            return {}

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DataFrame against limits and check for quality issues.
        Returns a report dictionary.
        """
        report = {
            "status": "PASS",
            "issues": [],
            "missing_stats": {},
            "sanity_checks": {}
        }
        
        # 1. Missing Data Check
        missing = df.isnull().sum()
        total_rows = len(df)
        for col in df.columns:
            missing_count = missing[col]
            missing_pct = (missing_count / total_rows) * 100
            report["missing_stats"][col] = f"{missing_pct:.1f}%"
            
            if missing_pct > 10:
                report["issues"].append(f"High missing data in {col}: {missing_pct:.1f}%")
                if report["status"] == "PASS": report["status"] = "WARNING"

        # 2. Schema/Limit Validation (Pandera)
        # Build schema dynamically based on columns present in DF
        schema_dict = {}
        for col in df.columns:
            if col in self.limits:
                limit = self.limits[col]
                checks = []
                if 'min' in limit:
                    checks.append(pa.Check.ge(limit['min'], error=f"{col} < {limit['min']}"))
                if 'max' in limit:
                    checks.append(pa.Check.le(limit['max'], error=f"{col} > {limit['max']}"))
                
                if checks:
                    schema_dict[col] = pa.Column(float, checks=checks, nullable=True, coerce=True)
        
        if schema_dict:
            schema = pa.DataFrameSchema(schema_dict)
            try:
                schema.validate(df, lazy=True)
                report["sanity_checks"]["schema"] = "PASSED"
            except pa.errors.SchemaErrors as err:
                report["status"] = "FAIL"
                report["sanity_checks"]["schema"] = "FAILED"
                
                # Summarize errors
                for failure in err.failure_cases.itertuples():
                    # failure_cases structure: index, failure_case, check, column, etc.
                    msg = f"Sanity Fail: {failure.column} value {failure.failure_case} violates {failure.check}"
                    if msg not in report["issues"]: # Avoid duplicates
                        report["issues"].append(msg)
                        
        # 3. Logical Checks (Example: Outlet > Inlet for Heating)
        # This would need specific logic based on test type (Heating/Cooling)
        # For now, we just log that we skipped it
        
        return report
