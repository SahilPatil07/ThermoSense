"""
Tool implementations for the agent platform
Wraps existing functionality with Pydantic contracts
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import logging

from backend.agent.tool_contracts import (
    SummarizeFileInput, SummarizeFileOutput,
    GenerateChartInput, GenerateChartOutput,
    CompareRunsInput, CompareRunsOutput,
    ExtractSensorsInput, ExtractSensorsOutput,
    GenerateReportInput, GenerateReportOutput
)
from backend.tools.excel_tools import read_excel_table, list_sheetnames
from backend.tools.utils import detect_time_column

logger = logging.getLogger(__name__)


class ToolImplementations:
    """
    Wrapper implementations for existing tools
    Provides Pydantic-validated interfaces
    """
    
    def __init__(
        self,
        storage,
        sensor_harvester,
        plotly_generator,
        comparative_analyzer,
        report_generator=None
    ):
        self.storage = storage
        self.sensor_harvester = sensor_harvester
        self.plotly_generator = plotly_generator
        self.comparative_analyzer = comparative_analyzer
        self.report_generator = report_generator
    
    def summarize_file(self, **kwargs) -> Dict[str, Any]:
        """Summarize data file"""
        try:
            input_data = SummarizeFileInput(**kwargs)
            
            # Get file path
            session_dir = self.storage.get_session_dir(input_data.session_id)
            file_path = session_dir / input_data.filename
            
            if not file_path.exists():
                return SummarizeFileOutput(
                    success=False,
                    filename=input_data.filename,
                    row_count=0,
                    column_count=0,
                    columns=[],
                    numeric_columns=[],
                    time_column=None,
                    summary="File not found"
                ).dict()
            
            # Get columns
            columns, time_column, numeric_columns = self.sensor_harvester.get_file_columns(str(file_path))
            
            # Load data for row count
            if input_data.filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                sheets = list_sheetnames(str(file_path))
                if sheets:
                    df = read_excel_table(str(file_path), sheets[0])
                else:
                    df = pd.DataFrame()
            
            summary = f"File contains {len(df)} rows and {len(columns)} columns"
            if time_column:
                summary += f". Time column: {time_column}"
            if numeric_columns:
                summary += f". {len(numeric_columns)} numeric columns available for analysis"
            
            return SummarizeFileOutput(
                success=True,
                filename=input_data.filename,
                row_count=len(df),
                column_count=len(columns),
                columns=columns,
                numeric_columns=numeric_columns,
                time_column=time_column,
                summary=summary
            ).dict()
            
        except Exception as e:
            logger.error(f"File summarization failed: {e}")
            return SummarizeFileOutput(
                success=False,
                filename=kwargs.get("filename", "unknown"),
                row_count=0,
                column_count=0,
                columns=[],
                numeric_columns=[],
                time_column=None,
                summary=f"Error: {str(e)}"
            ).dict()
    
    def generate_chart(self, **kwargs) -> Dict[str, Any]:
        """Generate chart"""
        try:
            input_data = GenerateChartInput(**kwargs)
            
            # Get file path
            session_dir = self.storage.get_session_dir(input_data.session_id)
            file_path = session_dir / input_data.filename
            
            if not file_path.exists():
                return GenerateChartOutput(
                    success=False,
                    chart_id="",
                    chart_url="",
                    chart_type=input_data.chart_type,
                    data_points=0,
                    summary="File not found"
                ).dict()
            
            # Load data
            if input_data.filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                # Use robust reader
                if input_data.sheet_name:
                    sheet_name = input_data.sheet_name
                else:
                    sheets = list_sheetnames(str(file_path))
                    sheet_name = sheets[0] if sheets else None
                
                if sheet_name:
                    df = read_excel_table(str(file_path), sheet_name)
                else:
                    df = pd.DataFrame()
            
            # Clean data
            df_clean = df[[input_data.x_column] + input_data.y_columns].dropna()
            
            # Generate chart
            output_path = str(session_dir / "chart")
            success, msg, html_path, png_path, chart_id, summary, stats, plotly_json = self.plotly_generator.generate_chart(
                df=df_clean,
                params={
                    "x_column": input_data.x_column,
                    "y_columns": input_data.y_columns,
                    "chart_type": input_data.chart_type,
                    "title": f"{', '.join(input_data.y_columns)} vs {input_data.x_column}",
                    "user_query": input_data.user_query
                },
                output_path=output_path
            )
            
            if success:
                # Store in history
                self.storage.add_chart_to_history(input_data.session_id, {
                    "chart_id": chart_id,
                    "file": input_data.filename,
                    "x_column": input_data.x_column,
                    "y_columns": input_data.y_columns,
                    "chart_type": input_data.chart_type,
                    "chart_path": png_path
                })
                
                chart_url = f"/workspace/{input_data.session_id}/{Path(png_path).name}"
                
                return GenerateChartOutput(
                    success=True,
                    chart_id=chart_id,
                    chart_url=chart_url,
                    chart_type=input_data.chart_type,
                    data_points=len(df_clean),
                    summary=summary,
                    html_path=html_path,
                    png_path=png_path
                ).dict()
            else:
                return GenerateChartOutput(
                    success=False,
                    chart_id="",
                    chart_url="",
                    chart_type=input_data.chart_type,
                    data_points=0,
                    summary=f"Chart generation failed: {msg}"
                ).dict()
                
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return GenerateChartOutput(
                success=False,
                chart_id="",
                chart_url="",
                chart_type=kwargs.get("chart_type", "line"),
                data_points=0,
                summary=f"Error: {str(e)}"
            ).dict()
    
    def compare_runs(self, **kwargs) -> Dict[str, Any]:
        """Compare multiple runs"""
        try:
            input_data = CompareRunsInput(**kwargs)
            
            session_dir = self.storage.get_session_dir(input_data.session_id)
            datasets = {}
            
            # Load all files
            for fname in input_data.files:
                fpath = session_dir / fname
                if fpath.exists():
                    if fname.endswith('.csv'):
                        datasets[fname] = pd.read_csv(fpath)
                    else:
                        datasets[fname] = pd.read_excel(fpath)
            
            if not datasets:
                return CompareRunsOutput(
                    success=False,
                    chart_id="",
                    chart_url="",
                    files_compared=[],
                    summary="No valid files found",
                    statistics={}
                ).dict()
            
            # Generate comparison chart
            import uuid
            chart_id = uuid.uuid4().hex[:8]
            output_path = session_dir / f"chart_{chart_id}.png"
            
            success, html_path, png_path, plotly_json = self.comparative_analyzer.generate_comparison_chart(
                datasets=datasets,
                column=input_data.column,
                output_path=str(output_path),
                chart_type=input_data.chart_type
            )
            
            if success:
                # Get statistics
                results = self.comparative_analyzer.compare_datasets(datasets, [input_data.column])
                
                chart_url = f"/workspace/{input_data.session_id}/{Path(png_path).name}"
                
                return CompareRunsOutput(
                    success=True,
                    chart_id=chart_id,
                    chart_url=chart_url,
                    files_compared=list(datasets.keys()),
                    summary=f"Compared {len(datasets)} files for {input_data.column}",
                    statistics=results.get('statistics', {})
                ).dict()
            else:
                return CompareRunsOutput(
                    success=False,
                    chart_id="",
                    chart_url="",
                    files_compared=[],
                    summary="Comparison chart generation failed",
                    statistics={}
                ).dict()
                
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return CompareRunsOutput(
                success=False,
                chart_id="",
                chart_url="",
                files_compared=[],
                summary=f"Error: {str(e)}",
                statistics={}
            ).dict()
    
    def extract_sensors(self, **kwargs) -> Dict[str, Any]:
        """Extract sensors from files"""
        try:
            input_data = ExtractSensorsInput(**kwargs)
            
            session_dir = self.storage.get_session_dir(input_data.session_id)
            file_paths = [str(session_dir / f) for f in input_data.files]
            
            # Harvest sensors
            master_df, metadata = self.sensor_harvester.harvest_sensors(
                file_paths=file_paths,
                required_sensors=input_data.sensors,
                strict_mode=False
            )
            
            if master_df is not None and not master_df.empty:
                # Save to output file
                output_path = session_dir / input_data.output_filename
                master_df.to_csv(output_path, index=False)
                
                return ExtractSensorsOutput(
                    success=True,
                    output_file=input_data.output_filename,
                    sensors_extracted=input_data.sensors,
                    total_rows=len(master_df),
                    summary=f"Extracted {len(input_data.sensors)} sensors from {len(input_data.files)} files"
                ).dict()
            else:
                return ExtractSensorsOutput(
                    success=False,
                    output_file="",
                    sensors_extracted=[],
                    total_rows=0,
                    summary="No data extracted"
                ).dict()
                
        except Exception as e:
            logger.error(f"Sensor extraction failed: {e}")
            return ExtractSensorsOutput(
                success=False,
                output_file="",
                sensors_extracted=[],
                total_rows=0,
                summary=f"Error: {str(e)}"
            ).dict()
    
    def generate_report(self, **kwargs) -> Dict[str, Any]:
        """Generate report"""
        try:
            input_data = GenerateReportInput(**kwargs)
            
            if not self.report_generator:
                return GenerateReportOutput(
                    success=False,
                    report_path="",
                    charts_included=0,
                    sections_populated=[],
                    summary="Report generator not available"
                ).dict()
            
            session_dir = self.storage.get_session_dir(input_data.session_id)
            output_path = session_dir / "thermal_report.docx"
            
            # Generate report
            success, msg, report_path = self.report_generator.generate_report(
                session_id=input_data.session_id,
                storage=self.storage,
                output_path=output_path
            )
            
            if not success:
                 return GenerateReportOutput(
                    success=False,
                    report_path="",
                    charts_included=0,
                    sections_populated=[],
                    summary=f"Report generation failed: {msg}"
                ).dict()
            
            # Count charts
            report_content = self.storage.get_report_content(input_data.session_id)
            charts_count = sum(
                len([item for item in items if item.get('type') == 'chart'])
                for items in report_content.values()
            )
            
            sections = [s for s, items in report_content.items() if items]
            
            return GenerateReportOutput(
                success=True,
                report_path=str(report_path),
                charts_included=charts_count,
                sections_populated=sections,
                summary=f"Report generated with {charts_count} charts in {len(sections)} sections"
            ).dict()
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return GenerateReportOutput(
                success=False,
                report_path="",
                charts_included=0,
                sections_populated=[],
                summary=f"Error: {str(e)}"
            ).dict()
