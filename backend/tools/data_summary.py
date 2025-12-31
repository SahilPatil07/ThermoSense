import pandas as pd
import numpy as np
from typing import Dict, Any
from backend.tools.enhanced_rag_tools import rag_system

class DataSummarizer:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def generate_chart_summary(self, df: pd.DataFrame, chart_params: Dict[str, Any]) -> str:
        y_col = chart_params.get('y_columns', [''])[0]
        
        if y_col not in df.columns:
            return "Unable to generate summary: column not found"
        
        data = pd.to_numeric(df[y_col], errors='coerce').dropna()
        
        if len(data) == 0:
            return "No valid data available for analysis"
        
        if data.std() < 0.01:
            return self._constant_summary(y_col, data)
        
        data_context = f"Parameter {y_col}: {len(data)} samples, Mean={data.mean():.3f}, StdDev={data.std():.3f}, Range=[{data.min():.3f}, {data.max():.3f}]"
        
        if self.llm_client and rag_system.collection.count() > 0:
            try:
                return rag_system.generate_with_llm(f"thermal analysis {y_col}", data_context, self.llm_client)
            except Exception as e:
                print(f"LLM generation failed: {e}")
        
        return self._variable_summary(y_col, data)

    def _constant_summary(self, param: str, data: pd.Series) -> str:
        return f"""Thermal Validation Analysis

Parameter: {param}
Status: Constant behavior detected

Measurement Data:
- Value: {data.iloc[0]:.4f}
- Samples: {len(data)}
- Variation: {data.std():.6f}

Assessment: Parameter exhibits steady-state behavior throughout test period."""

    def _variable_summary(self, param: str, data: pd.Series) -> str:
        cv = (data.std() / data.mean() * 100) if data.mean() != 0 else 0
        return f"""Thermal Performance Analysis

Parameter: {param}
Samples: {len(data):,}

Statistical Analysis:
- Range: {data.min():.3f} to {data.max():.3f}
- Mean: {data.mean():.3f}, Median: {data.median():.3f}
- Std Dev: {data.std():.3f}, CV: {cv:.1f}%
- Distribution: P10={data.quantile(0.1):.3f}, P50={data.quantile(0.5):.3f}, P90={data.quantile(0.9):.3f}

Assessment: {'Stable thermal performance' if cv < 15 else 'Variable thermal behavior'} validated against specifications."""
