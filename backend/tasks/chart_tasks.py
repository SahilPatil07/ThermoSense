"""
Background tasks for chart generation
Handles large datasets and time-consuming chart operations
"""
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
from celery import Task
from backend.celery_app import app
from backend.tools.chart_tools_plotly import PlotlyChartGenerator
from backend.tools.comparative_analyzer import ComparativeAnalyzer
import pandas as pd

logger = logging.getLogger(__name__)


class ChartGenerationTask(Task):
    """Base task with error handling and retry logic"""
    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3, "countdown": 5}
    retry_backoff = True


@app.task(bind=True, base=ChartGenerationTask, name="backend.tasks.chart_tasks.generate_chart")
def generate_chart_task(
    self,
    session_id: str,
    filename: str,
    x_column: str,
    y_columns: List[str],
    chart_type: str = "line",
    user_query: str = None
) -> Dict[str, Any]:
    """
    Background task for chart generation
    
    Args:
        session_id: Session identifier
        filename: Data file name
        x_column: X-axis column name
        y_columns: List of Y-axis column names
        chart_type: Type of chart ('line', 'scatter', 'bar', etc.)
        user_query: Optional user query for context
    
    Returns:
        Dictionary with chart paths and metadata
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting chart generation task: {self.request.id}")
        logger.info(f"Session: {session_id}, File: {filename}, Type: {chart_type}")
        
        # Update task state
        self.update_state(
            state="PROGRESS",
            meta={"status": "Loading data", "progress": 10}
        )
        
        # Load data
        workspace_dir = Path("workspace") / session_id
        file_path = workspace_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        
        # Read data based on file type
        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
        
        logger.info(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
        
        # Update progress
        self.update_state(
            state="PROGRESS",
            meta={"status": "Generating chart", "progress": 50}
        )
        
        # Generate chart
        chart_gen = PlotlyChartGenerator()
        output_path = str(workspace_dir / f"chart_{int(time.time())}")
        
        success, html_path, png_path, plotly_json = chart_gen.generate_chart(
            df=df,
            x_column=x_column,
            y_columns=y_columns,
            chart_type=chart_type,
            output_path=output_path,
            title=user_query or f"{chart_type.title()} Chart"
        )
        
        if not success:
            raise RuntimeError("Chart generation failed")
        
        # Calculate duration
        duration = time.time() - start_time
        
        logger.info(f"Chart generated successfully in {duration:.2f}s")
        
        return {
            "success": True,
            "html_path": html_path,
            "png_path": png_path,
            "plotly_json": plotly_json,
            "duration": duration,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"Chart generation failed: {e}", exc_info=True)
        raise


@app.task(bind=True, base=ChartGenerationTask, name="backend.tasks.chart_tasks.generate_comparison_chart")
def generate_comparison_chart_task(
    self,
    session_id: str,
    datasets: Dict[str, str],  # {label: filename}
    column: str,
    chart_type: str = "line"
) -> Dict[str, Any]:
    """
    Background task for comparison chart generation
    
    Args:
        session_id: Session identifier
        datasets: Dictionary mapping labels to filenames
        column: Column to compare
        chart_type: Type of chart
    
    Returns:
        Dictionary with chart paths and metadata
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting comparison chart task: {self.request.id}")
        
        # Update progress
        self.update_state(
            state="PROGRESS",
            meta={"status": "Loading datasets", "progress": 20}
        )
        
        # Load all datasets
        workspace_dir = Path("workspace") / session_id
        loaded_datasets = {}
        
        for label, filename in datasets.items():
            file_path = workspace_dir / filename
            if filename.endswith(".csv"):
                loaded_datasets[label] = pd.read_csv(file_path)
            elif filename.endswith((".xlsx", ".xls")):
                loaded_datasets[label] = pd.read_excel(file_path)
        
        logger.info(f"Loaded {len(loaded_datasets)} datasets for comparison")
        
        # Update progress
        self.update_state(
            state="PROGRESS",
            meta={"status": "Generating comparison chart", "progress": 60}
        )
        
        # Generate comparison chart
        analyzer = ComparativeAnalyzer()
        output_path = str(workspace_dir / f"comparison_{int(time.time())}")
        
        success, html_path, png_path, plotly_json = analyzer.generate_comparison_chart(
            datasets=loaded_datasets,
            column=column,
            output_path=output_path,
            chart_type=chart_type
        )
        
        if not success:
            raise RuntimeError("Comparison chart generation failed")
        
        duration = time.time() - start_time
        
        logger.info(f"Comparison chart generated in {duration:.2f}s")
        
        return {
            "success": True,
            "html_path": html_path,
            "png_path": png_path,
            "plotly_json": plotly_json,
            "duration": duration,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"Comparison chart generation failed: {e}", exc_info=True)
        raise


@app.task(bind=True, name="backend.tasks.chart_tasks.generate_heatmap")
def generate_heatmap_task(
    self,
    session_id: str,
    filename: str,
    columns: List[str] = None
) -> Dict[str, Any]:
    """
    Background task for heatmap generation
    
    Args:
        session_id: Session identifier
        filename: Data file name
        columns: Optional list of columns to include
    
    Returns:
        Dictionary with chart paths and metadata
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting heatmap generation task: {self.request.id}")
        
        # Load data
        workspace_dir = Path("workspace") / session_id
        file_path = workspace_dir / filename
        
        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
        
        # Select numeric columns
        if columns:
            df = df[columns]
        else:
            df = df.select_dtypes(include=["number"])
        
        # Generate correlation heatmap
        chart_gen = PlotlyChartGenerator()
        output_path = str(workspace_dir / f"heatmap_{int(time.time())}")
        
        success, html_path, png_path, plotly_json = chart_gen.generate_heatmap(
            df=df,
            output_path=output_path
        )
        
        if not success:
            raise RuntimeError("Heatmap generation failed")
        
        duration = time.time() - start_time
        
        return {
            "success": True,
            "html_path": html_path,
            "png_path": png_path,
            "plotly_json": plotly_json,
            "duration": duration,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"Heatmap generation failed: {e}", exc_info=True)
        raise
