"""
Background tasks for report generation
Handles PDF/DOCX export and template processing
"""
import time
import logging
from pathlib import Path
from typing import Dict, Any
from celery import Task
from backend.celery_app import app
from backend.tools.report_generator import get_report_generator
from backend.tools.storage import SessionStorage

logger = logging.getLogger(__name__)


class ReportGenerationTask(Task):
    """Base task for report generation with retry logic"""
    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 2, "countdown": 10}


@app.task(bind=True, base=ReportGenerationTask, name="backend.tasks.report_tasks.generate_report")
def generate_report_task(
    self,
    session_id: str,
    llm_client=None
) -> Dict[str, Any]:
    """
    Background task for report generation
    
    Args:
        session_id: Session identifier
        llm_client: Optional LLM client for content generation
    
    Returns:
        Dictionary with report path and metadata
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting report generation task: {self.request.id}")
        logger.info(f"Session: {session_id}")
        
        # Update task state
        self.update_state(
            state="PROGRESS",
            meta={"status": "Initializing report generator", "progress": 10}
        )
        
        # Initialize components
        storage = SessionStorage(workspace_dir="workspace")
        report_gen = get_report_generator(llm_client=llm_client)
        
        # Update progress
        self.update_state(
            state="PROGRESS",
            meta={"status": "Generating report content", "progress": 30}
        )
        
        # Generate report
        workspace_dir = Path("workspace") / session_id
        output_path = workspace_dir / f"thermal_report_{int(time.time())}.docx"
        
        report_path = report_gen.generate_report(
            session_id=session_id,
            storage=storage,
            output_path=output_path
        )
        
        # Update progress
        self.update_state(
            state="PROGRESS",
            meta={"status": "Finalizing report", "progress": 90}
        )
        
        duration = time.time() - start_time
        
        logger.info(f"Report generated successfully in {duration:.2f}s")
        logger.info(f"Report path: {report_path}")
        
        return {
            "success": True,
            "report_path": str(report_path),
            "duration": duration,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        raise


@app.task(bind=True, name="backend.tasks.report_tasks.export_report_to_pdf")
def export_report_to_pdf_task(
    self,
    session_id: str,
    docx_path: str
) -> Dict[str, Any]:
    """
    Background task for PDF export (future implementation)
    
    Args:
        session_id: Session identifier
        docx_path: Path to DOCX file
    
    Returns:
        Dictionary with PDF path and metadata
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting PDF export task: {self.request.id}")
        
        # TODO: Implement PDF export using WeasyPrint or Playwright
        # For now, return placeholder
        
        duration = time.time() - start_time
        
        return {
            "success": False,
            "message": "PDF export not yet implemented",
            "duration": duration,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"PDF export failed: {e}", exc_info=True)
        raise
