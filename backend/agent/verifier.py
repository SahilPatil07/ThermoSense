"""
Output Verifier - Validates tool outputs for correctness
Ensures zero hallucinations by checking data existence and validity
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Result of output validation"""
    valid: bool = Field(..., description="Whether output is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in validation")
    
    class Config:
        schema_extra = {
            "example": {
                "valid": True,
                "errors": [],
                "warnings": ["Data contains 5% null values"],
                "confidence": 0.95
            }
        }


class OutputVerifier:
    """
    Validates tool outputs to ensure correctness
    Prevents hallucinations by checking actual data
    """
    
    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace = Path(workspace_dir)
    
    def verify_chart_output(
        self,
        output: Dict[str, Any],
        session_id: str
    ) -> ValidationResult:
        """
        Verify chart generation output
        
        Args:
            output: Chart tool output
            session_id: Session identifier
        
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ["success", "chart_id", "chart_url"]
        for field in required_fields:
            if field not in output:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return ValidationResult(valid=False, errors=errors, confidence=0.0)
        
        # Check success flag
        if not output.get("success"):
            errors.append("Chart generation reported failure")
            return ValidationResult(
                valid=False,
                errors=errors,
                confidence=0.0
            )
        
        # Verify chart file exists
        chart_url = output.get("chart_url", "")
        if chart_url:
            # Extract filename from URL
            chart_filename = Path(chart_url).name
            chart_path = self.workspace / session_id / chart_filename
            
            if not chart_path.exists():
                errors.append(f"Chart file not found: {chart_path}")
        else:
            errors.append("No chart URL provided")
        
        # Check data points
        data_points = output.get("data_points", 0)
        if data_points == 0:
            warnings.append("Chart has zero data points")
        elif data_points < 10:
            warnings.append(f"Chart has very few data points: {data_points}")
        
        # Check for metadata
        if "metadata" in output:
            dropped_pct = output["metadata"].get("dropped_null_pct", 0)
            if dropped_pct > 20:
                warnings.append(f"High percentage of null values dropped: {dropped_pct}%")
        
        valid = len(errors) == 0
        confidence = 1.0 if valid and not warnings else 0.8
        
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            confidence=confidence
        )
    
    def verify_comparison_output(
        self,
        output: Dict[str, Any],
        session_id: str
    ) -> ValidationResult:
        """Verify comparison tool output"""
        errors = []
        warnings = []
        
        # Check required fields
        if not output.get("success"):
            errors.append("Comparison reported failure")
        
        if "files_compared" not in output:
            errors.append("Missing files_compared field")
        elif len(output["files_compared"]) < 2:
            errors.append("Comparison requires at least 2 files")
        
        # Verify chart exists
        if output.get("chart_url"):
            chart_filename = Path(output["chart_url"]).name
            chart_path = self.workspace / session_id / chart_filename
            
            if not chart_path.exists():
                errors.append(f"Comparison chart not found: {chart_path}")
        
        # Check statistics
        if "statistics" not in output:
            warnings.append("No statistics provided")
        
        valid = len(errors) == 0
        confidence = 1.0 if valid else 0.0
        
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            confidence=confidence
        )
    
    def verify_report_output(
        self,
        output: Dict[str, Any],
        session_id: str
    ) -> ValidationResult:
        """Verify report generation output"""
        errors = []
        warnings = []
        
        if not output.get("success"):
            errors.append("Report generation reported failure")
        
        # Verify report file exists
        if "report_path" in output:
            report_path = Path(output["report_path"])
            if not report_path.exists():
                errors.append(f"Report file not found: {report_path}")
            else:
                # Check file size
                file_size = report_path.stat().st_size
                if file_size < 1000:  # Less than 1KB
                    warnings.append(f"Report file is very small: {file_size} bytes")
        else:
            errors.append("No report path provided")
        
        # Check charts included
        charts_included = output.get("charts_included", 0)
        if charts_included == 0:
            warnings.append("Report contains no charts")
        
        # Check sections
        sections = output.get("sections_populated", [])
        if not sections:
            warnings.append("No report sections populated")
        
        valid = len(errors) == 0
        confidence = 1.0 if valid and not warnings else 0.8
        
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            confidence=confidence
        )
    
    def verify_extraction_output(
        self,
        output: Dict[str, Any],
        session_id: str
    ) -> ValidationResult:
        """Verify sensor extraction output"""
        errors = []
        warnings = []
        
        if not output.get("success"):
            errors.append("Extraction reported failure")
        
        # Verify output file exists
        if "output_file" in output:
            output_filename = output["output_file"]
            output_path = self.workspace / session_id / output_filename
            
            if not output_path.exists():
                errors.append(f"Extracted file not found: {output_path}")
            else:
                # Check file size
                file_size = output_path.stat().st_size
                if file_size < 100:
                    warnings.append(f"Extracted file is very small: {file_size} bytes")
        else:
            errors.append("No output file specified")
        
        # Check sensors extracted
        sensors = output.get("sensors_extracted", [])
        if not sensors:
            warnings.append("No sensors were extracted")
        
        # Check row count
        total_rows = output.get("total_rows", 0)
        if total_rows == 0:
            warnings.append("Extracted file has zero rows")
        
        valid = len(errors) == 0
        confidence = 1.0 if valid else 0.0
        
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            confidence=confidence
        )
    
    def verify_data_summary(
        self,
        output: Dict[str, Any]
    ) -> ValidationResult:
        """Verify data summarization output"""
        errors = []
        warnings = []
        
        if not output.get("success"):
            errors.append("Summarization reported failure")
        
        # Check required fields
        required = ["row_count", "column_count", "columns"]
        for field in required:
            if field not in output:
                errors.append(f"Missing required field: {field}")
        
        # Validate counts
        if output.get("row_count", 0) == 0:
            warnings.append("File has zero rows")
        
        if output.get("column_count", 0) == 0:
            errors.append("File has zero columns")
        
        # Check columns list matches count
        columns = output.get("columns", [])
        column_count = output.get("column_count", 0)
        if len(columns) != column_count:
            warnings.append(f"Column count mismatch: {len(columns)} vs {column_count}")
        
        valid = len(errors) == 0
        confidence = 1.0 if valid else 0.5
        
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            confidence=confidence
        )
    
    def verify_output(
        self,
        tool_name: str,
        output: Dict[str, Any],
        session_id: str = "default"
    ) -> ValidationResult:
        """
        Verify tool output based on tool type
        
        Args:
            tool_name: Name of tool that produced output
            output: Tool output to verify
            session_id: Session identifier
        
        Returns:
            Validation result
        """
        # Route to appropriate verifier
        if "chart" in tool_name.lower():
            return self.verify_chart_output(output, session_id)
        elif "compare" in tool_name.lower():
            return self.verify_comparison_output(output, session_id)
        elif "report" in tool_name.lower():
            return self.verify_report_output(output, session_id)
        elif "extract" in tool_name.lower():
            return self.verify_extraction_output(output, session_id)
        elif "summarize" in tool_name.lower():
            return self.verify_data_summary(output)
        else:
            # Generic verification
            if not output.get("success"):
                return ValidationResult(
                    valid=False,
                    errors=["Tool reported failure"],
                    confidence=0.0
                )
            
            return ValidationResult(
                valid=True,
                errors=[],
                warnings=[],
                confidence=0.8
            )
