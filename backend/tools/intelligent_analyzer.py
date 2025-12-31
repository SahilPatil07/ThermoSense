import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
import json
from backend.tools.json_utils import safe_json_dumps

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """
    Intelligent Data Analyzer for Engineering Datasets.
    Identifies physical quantities and suggests relevant charts.
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
        
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the dataset to understand its semantic content.
        """
        summary = {
            "row_count": len(df),
            "columns": [],
            "potential_time_col": None,
            "numeric_cols": [],
            "categorical_cols": []
        }
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            
            col_info = {
                "name": col,
                "type": col_type,
                "sample": df[col].dropna().head(3).tolist() if not df[col].empty else []
            }
            
            if is_numeric:
                summary["numeric_cols"].append(col)
                # Simple heuristic for physical quantities based on name
                col_lower = col.lower()
                if any(x in col_lower for x in ['temp', 'deg', 'celsius', 'fahrenheit']):
                    col_info["semantic_type"] = "Temperature"
                elif any(x in col_lower for x in ['pres', 'bar', 'psi', 'pascal']):
                    col_info["semantic_type"] = "Pressure"
                elif any(x in col_lower for x in ['flow', 'kg/s', 'l/min']):
                    col_info["semantic_type"] = "Flow Rate"
                elif any(x in col_lower for x in ['speed', 'rpm', 'velocity']):
                    col_info["semantic_type"] = "Speed/Velocity"
                elif any(x in col_lower for x in ['time', 'date', 'clock']):
                    col_info["semantic_type"] = "Time"
                    if not summary["potential_time_col"]:
                        summary["potential_time_col"] = col
                else:
                    col_info["semantic_type"] = "Numeric"
            else:
                summary["categorical_cols"].append(col)
                col_info["semantic_type"] = "Categorical"
                
            summary["columns"].append(col_info)
            
        return summary

    def suggest_charts(self, df_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Suggest relevant engineering charts based on the dataset summary.
        Uses LLM if available, otherwise falls back to heuristics.
        """
        if self.llm:
            return self._suggest_with_llm(df_summary)
        else:
            return self._suggest_heuristically(df_summary)

    def _suggest_with_llm(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use LLM to suggest charts.
        """
        prompt = f"""
        You are a Senior Data Engineer. Analyze this dataset summary and suggest 3-5 high-value engineering charts.
        
        Dataset Summary:
        - Rows: {summary['row_count']}
        - Time Column: {summary['potential_time_col']}
        - Columns: {safe_json_dumps(summary['columns'], indent=2)}
        
        Rules:
        1. If a Time column exists, prioritize Time Series plots.
        2. Look for relationships between physical quantities (e.g., Pressure vs Temperature).
        3. Suggest a mix of Line charts, Scatter plots, and Histograms if relevant.
        4. Return ONLY valid JSON in the following format:
        [
            {{
                "title": "Chart Title",
                "chart_type": "line|scatter|bar|histogram|heatmap",
                "x_column": "Exact Column Name",
                "y_columns": ["Exact Column Name 1", "Exact Column Name 2"],
                "reason": "Why this chart is useful"
            }}
        ]
        """
        
        try:
            response = self.llm.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            content = response.choices[0].message.content.strip()
            # Try to extract JSON if wrapped in code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            suggestions = json.loads(content)
            return suggestions
        except Exception as e:
            logger.error(f"LLM chart suggestion failed: {e}")
            return self._suggest_heuristically(summary)

    def _suggest_heuristically(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fallback logic for chart suggestions.
        """
        suggestions = []
        time_col = summary.get("potential_time_col")
        numeric_cols = [c for c in summary["numeric_cols"] if c != time_col]
        
        # 1. Time Series (if time exists)
        if time_col and numeric_cols:
            # Suggest plotting the first 2-3 numeric columns against time
            suggestions.append({
                "title": "Key Parameters over Time",
                "chart_type": "line",
                "x_column": time_col,
                "y_columns": numeric_cols[:3],
                "reason": "Overview of system behavior over time."
            })
            
        # 2. Correlation (Scatter)
        if len(numeric_cols) >= 2:
            suggestions.append({
                "title": f"Correlation: {numeric_cols[0]} vs {numeric_cols[1]}",
                "chart_type": "scatter",
                "x_column": numeric_cols[0],
                "y_columns": [numeric_cols[1]],
                "reason": "Check for correlation between variables."
            })
            
        # 3. Distribution
        if numeric_cols:
            suggestions.append({
                "title": f"Distribution of {numeric_cols[0]}",
                "chart_type": "histogram",
                "x_column": numeric_cols[0], # Histogram uses x_column for the data source in some implementations, or y. Let's assume y in our system usually? 
                # Wait, our chart_tools.py histogram uses y_col.
                "y_columns": [numeric_cols[0]], 
                "x_column": "Frequency", # Placeholder for X
                "reason": "Analyze data distribution and stability."
            })
            
        return suggestions
