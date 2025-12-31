"""
Pydantic contracts for all tools
Strict input/output schemas for zero hallucinations
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ========== DATA TOOLS ==========

class SummarizeFileInput(BaseModel):
    """Input for file summarization"""
    session_id: str = Field(..., description="Session identifier")
    filename: str = Field(..., description="File to summarize")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "default",
                "filename": "thermal_data.csv"
            }
        }


class SummarizeFileOutput(BaseModel):
    """Output from file summarization"""
    success: bool
    filename: str
    row_count: int
    column_count: int
    columns: List[str]
    numeric_columns: List[str]
    time_column: Optional[str]
    summary: str
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "filename": "thermal_data.csv",
                "row_count": 1000,
                "column_count": 10,
                "columns": ["Time", "Temp_1", "Temp_2"],
                "numeric_columns": ["Temp_1", "Temp_2"],
                "time_column": "Time",
                "summary": "Thermal data with 1000 measurements"
            }
        }


# ========== CHART TOOLS ==========

class GenerateChartInput(BaseModel):
    """Input for chart generation"""
    session_id: str = Field(..., description="Session identifier")
    filename: str = Field(..., description="Data file")
    x_column: str = Field(..., description="X-axis column")
    y_columns: List[str] = Field(..., description="Y-axis columns")
    chart_type: str = Field(default="line", description="Chart type: line, scatter, bar")
    user_query: Optional[str] = Field(None, description="User's original request")
    sheet_name: Optional[str] = Field(None, description="Sheet name for Excel files")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "default",
                "filename": "thermal_data.csv",
                "x_column": "Time",
                "y_columns": ["Temp_1", "Temp_2"],
                "chart_type": "line",
                "user_query": "Show temperature trends"
            }
        }


class GenerateChartOutput(BaseModel):
    """Output from chart generation"""
    success: bool
    chart_id: str
    chart_url: str
    chart_type: str
    data_points: int
    summary: str
    html_path: Optional[str] = None
    png_path: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "chart_id": "abc123",
                "chart_url": "/workspace/default/chart_abc123.png",
                "chart_type": "line",
                "data_points": 1000,
                "summary": "Temperature trends over time",
                "html_path": "/workspace/default/chart_abc123.html",
                "png_path": "/workspace/default/chart_abc123.png"
            }
        }


# ========== COMPARISON TOOLS ==========

class CompareRunsInput(BaseModel):
    """Input for multi-file comparison"""
    session_id: str = Field(..., description="Session identifier")
    files: List[str] = Field(..., description="Files to compare")
    column: str = Field(..., description="Column to compare")
    chart_type: str = Field(default="line", description="Chart type")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "default",
                "files": ["run1.csv", "run2.csv"],
                "column": "Temperature",
                "chart_type": "line"
            }
        }


class CompareRunsOutput(BaseModel):
    """Output from comparison"""
    success: bool
    chart_id: str
    chart_url: str
    files_compared: List[str]
    summary: str
    statistics: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "chart_id": "comp123",
                "chart_url": "/workspace/default/comparison_comp123.png",
                "files_compared": ["run1.csv", "run2.csv"],
                "summary": "Run1 shows 5°C higher average than Run2",
                "statistics": {"mean_diff": 5.0}
            }
        }


# ========== EXTRACTION TOOLS ==========

class ExtractSensorsInput(BaseModel):
    """Input for sensor data extraction"""
    session_id: str = Field(..., description="Session identifier")
    files: List[str] = Field(..., description="Files to extract from")
    sensors: List[str] = Field(..., description="Sensor names to extract")
    output_filename: str = Field(..., description="Output file name")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "default",
                "files": ["test1.xlsx", "test2.xlsx"],
                "sensors": ["Temp_Coolant", "Temp_Ambient"],
                "output_filename": "extracted_sensors.csv"
            }
        }


class ExtractSensorsOutput(BaseModel):
    """Output from sensor extraction"""
    success: bool
    output_file: str
    sensors_extracted: List[str]
    total_rows: int
    summary: str
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "output_file": "extracted_sensors.csv",
                "sensors_extracted": ["Temp_Coolant", "Temp_Ambient"],
                "total_rows": 5000,
                "summary": "Extracted 2 sensors from 2 files"
            }
        }


# ========== REPORT TOOLS ==========

class GenerateReportInput(BaseModel):
    """Input for report generation"""
    session_id: str = Field(..., description="Session identifier")
    include_charts: bool = Field(default=True, description="Include approved charts")
    include_analysis: bool = Field(default=True, description="Include LLM analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "default",
                "include_charts": True,
                "include_analysis": True
            }
        }


class GenerateReportOutput(BaseModel):
    """Output from report generation"""
    success: bool
    report_path: str
    charts_included: int
    sections_populated: List[str]
    summary: str
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "report_path": "/workspace/default/thermal_report.docx",
                "charts_included": 5,
                "sections_populated": ["Objectives", "Analysis and Results"],
                "summary": "Report generated with 5 charts"
            }
        }


# ========== KNOWLEDGE TOOLS ==========

class QueryKnowledgeInput(BaseModel):
    """Input for knowledge base query"""
    query: str = Field(..., description="User question")
    top_k: int = Field(default=5, description="Number of results")
    min_confidence: float = Field(default=0.2, description="Minimum confidence threshold")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is thermal conductivity?",
                "top_k": 5,
                "min_confidence": 0.2
            }
        }


class QueryKnowledgeOutput(BaseModel):
    """Output from knowledge query"""
    success: bool
    answer: str
    sources: List[str]
    confidence: float
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "answer": "Thermal conductivity is...",
                "sources": ["thermal_handbook.pdf", "standards.docx"],
                "confidence": 0.85
            }
        }


# ========== ANALYTICS TOOLS ==========

class DetectAnomaliesInput(BaseModel):
    """Input for anomaly detection"""
    session_id: str = Field(..., description="Session identifier")
    filename: str = Field(..., description="Data file")
    columns: List[str] = Field(..., description="Columns to analyze")
    method: str = Field(default="isolation_forest", description="Detection method")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "default",
                "filename": "thermal_data.csv",
                "columns": ["Temp_1", "Temp_2"],
                "method": "isolation_forest"
            }
        }


class DetectAnomaliesOutput(BaseModel):
    """Output from anomaly detection"""
    success: bool
    anomalies_found: int
    anomaly_indices: List[int]
    summary: str
    confidence_scores: Optional[List[float]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "anomalies_found": 12,
                "anomaly_indices": [45, 67, 89],
                "summary": "Found 12 anomalies using IsolationForest",
                "confidence_scores": [0.95, 0.87, 0.92]
            }
        }
