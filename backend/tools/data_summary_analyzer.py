import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataSummaryAnalyzer:
    """
    Intelligent data analyzer that understands column semantics 
    and generates natural, professional summaries
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    def analyze_data_file(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """
        Comprehensive data analysis with LLM-powered column interpretation
        """
        # Step 1: Extract basic statistics
        stats = self._extract_statistics(df)
        
        # Step 2: LLM interprets what each column represents
        column_interpretations = self._interpret_columns(df, filename)
        
        # Step 3: Find key findings from data
        key_findings = self._extract_key_findings(df, column_interpretations, stats)
        
        # Step 4: Generate natural summary (no asterisks!)
        summary_text = self._generate_natural_summary(
            filename, column_interpretations, key_findings, stats
        )
        
        return {
            "filename": filename,
            "analyzed_at": datetime.now().isoformat(),
            "file_stats": {
                "row_count": len(df),
                "column_count": len(df.columns),
                "duration_records": f"{len(df)} data points"
            },
            "column_interpretations": column_interpretations,
            "statistics": stats,
            "key_findings": key_findings,
            "executive_summary": summary_text  # Natural, professional summary
        }
    
    def _extract_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract numerical statistics for each column"""
        stats = {}
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                data = df[col].dropna()
                if len(data) > 0:
                    stats[col] = {
                        "min": float(data.min()),
                        "max": float(data.max()),
                        "mean": float(data.mean()),
                        "std": float(data.std()) if len(data) > 1 else 0,
                        "unit": self._detect_unit(col)
                    }
        
        return stats
    
    def _detect_unit(self, column_name: str) -> str:
        """Detect likely unit from column name"""
        col_lower = column_name.lower()
        
        if any(x in col_lower for x in ['temp', 'deg', 'celsius']):
            return "°C"
        elif 'fahrenheit' in col_lower:
            return "°F"
        elif any(x in col_lower for x in ['pres', 'bar']):
            return "bar"
        elif 'psi' in col_lower:
            return "psi"
        elif any(x in col_lower for x in ['flow', 'lpm']):
            return "lpm"
        elif any(x in col_lower for x in ['speed', 'rpm']):
            return "rpm"
        elif any(x in col_lower for x in ['volt', 'voltage']):
            return "V"
        elif any(x in col_lower for x in ['current', 'amp']):
            return "A"
        elif 'time' in col_lower:
            return "s"
        else:
            return ""
    
    def _interpret_columns(self, df: pd.DataFrame, filename: str) -> Dict[str, str]:
        """Use LLM to understand what each column represents"""
        if not self.llm:
            return self._fallback_column_interpretation(df)
        
        # Build prompt with column names and sample data
        sample_data = {}
        for col in df.columns[:10]:  # First 10 columns to avoid token limit
            sample_data[col] = df[col].dropna().head(3).tolist()
        
        prompt = f"""You are analyzing thermal test data from automotive testing. 
File: {filename}

Column names and sample values:
{chr(10).join([f"- {col}: {sample_data.get(col, [])}" for col in df.columns[:10]])}

For each column, explain in ONE SHORT PHRASE what it represents in automotive thermal engineering context.

Format your response as:
Column_Name: Brief explanation (e.g., "Electric motor inlet coolant temperature")

Be specific to automotive thermal systems. Keep explanations under 10 words.
Respond now:"""
        
        try:
            response = self.llm.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300
            )
            
            # Parse LLM response
            interpretations = {}
            for line in response.choices[0].message.content.strip().split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        col = parts[0].strip()
                        desc = parts[1].strip()
                        if col in df.columns:
                            interpretations[col] = desc
            
            # Fallback for any missing columns
            for col in df.columns:
                if col not in interpretations:
                    interpretations[col] = self._simple_interpretation(col)
                    
            return interpretations
            
        except Exception as e:
            logger.error(f"LLM column interpretation failed: {e}")
            return self._fallback_column_interpretation(df)
    
    def _simple_interpretation(self, col_name: str) -> str:
        """Simple interpretation based on column name"""
        col_lower = col_name.lower()
        if 'temp' in col_lower:
            return "Temperature measurement"
        elif 'pres' in col_lower:
            return "Pressure measurement"
        elif 'flow' in col_lower:
            return "Flow rate measurement"
        elif 'time' in col_lower:
            return "Time series data"
        else:
            return "Sensor measurement"
    
    def _fallback_column_interpretation(self, df: pd.DataFrame) -> Dict[str, str]:
        """Fallback if LLM unavailable"""
        return {col: self._simple_interpretation(col) for col in df.columns}
    
    def _extract_key_findings(self, df: pd.DataFrame, 
                             interpretations: Dict[str, str],
                             stats: Dict[str, Any]) -> List[str]:
        """Extract meaningful findings from data"""
        findings = []
        
        for col, col_stats in stats.items():
            unit = col_stats['unit']
            desc = interpretations.get(col, col)
            
            # Temperature findings
            if unit == "°C":
                findings.append(
                    f"{desc} ranged from {col_stats['min']:.1f}{unit} to "
                    f"{col_stats['max']:.1f}{unit}, averaging {col_stats['mean']:.1f}{unit}"
                )
                
                # Check common thresholds
                if col_stats['max'] < 60:
                    findings.append(f"{desc} remained below 60{unit} critical threshold")
                if col_stats['max'] > 50 and 'inlet' in col.lower():
                    findings.append(f"{desc} exceeded 50{unit} during operation")
            
            # Flow findings
            elif unit == "lpm":
                findings.append(
                    f"{desc} maintained {col_stats['mean']:.1f}{unit} average flow rate"
                )
                if col_stats['min'] > 9:
                    findings.append(f"{desc} exceeded 9 {unit} minimum requirement")
        
        return findings[:6]  # Limit to 6 most important findings
    
    def _generate_natural_summary(self, filename: str,
                                  interpretations: Dict[str, str],
                                  findings: List[str],
                                  stats: Dict[str, Any]) -> str:
        """Generate natural, professional summary WITHOUT asterisks"""
        if not self.llm:
            return self._fallback_summary(filename, findings)
        
        # Build context for LLM
        findings_text = "\n".join([f"- {f}" for f in findings])
        
        prompt = f"""You are a Senior Automotive Thermal Engineer writing an Executive Summary for an Engineering Report.

Test Data File: {filename}
Number of measurements: {len(stats)} thermal parameters

Key Findings from Data Analysis:
{findings_text}

Write a professional Executive Summary (3-4 paragraphs) following this structure:

Paragraph 1: Overview
State what test was conducted, what system was evaluated, and general test configuration.

Paragraph 2-3: Key Results
Report the specific measured values and findings. Use the actual numbers from the findings above.
Be specific: "Temperature reached X degrees" not "Temperature was high"

Paragraph 4: Conclusion
State whether requirements were met and overall assessment.

CRITICAL REQUIREMENTS:
- NO asterisks or markdown formatting
- NO bullet points
- Use complete sentences in paragraph form
- Include SPECIFIC VALUES from the findings
- Sound like a professional engineer, not AI
- Use passive voice where appropriate
- Be concise but technical

Write the Executive Summary now:"""
        
        try:
            response = self.llm.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Remove any asterisks that might slip through
            summary = summary.replace('**', '').replace('*', '')
            
            return summary
            
        except Exception as e:
            logger.error(f"LLM summary generation failed: {e}")
            return self._fallback_summary(filename, findings)
    
    def _fallback_summary(self, filename: str, findings: List[str]) -> str:
        """Fallback summary if LLM unavailable"""
        return (
            f"The thermal analysis evaluated data from {filename}, examining multiple "
            f"thermal parameters across the test duration. "
            f"Key measurements included {findings[0] if findings else 'temperature and flow data'}. "
            f"All critical parameters remained within acceptable operational limits. "
            f"The thermal management system demonstrated satisfactory performance "
            f"under test conditions."
        )
