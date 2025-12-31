"""
Rollback script: PostgreSQL → JSON files
Restores JSON-based storage from database backup
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session
from backend.db.database import SessionLocal
from backend.db.models import (
    Session as DBSession,
    Upload,
    Message,
    Chart,
    ReportSection
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rollback_session_data(session_id: str, db: Session, workspace_root: Path):
    """
    Rollback a single session from database to JSON
    
    Args:
        session_id: Session identifier
        db: Database session
        workspace_root: Root workspace directory
    """
    try:
        # Get session from database
        db_session = db.query(DBSession).filter(DBSession.id == session_id).first()
        
        if not db_session:
            logger.warning(f"Session not found in database: {session_id}")
            return
        
        logger.info(f"Rolling back session: {session_id}")
        
        # Create session directory
        session_dir = workspace_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Build memory structure
        memory = {
            "title": db_session.title,
            "created_at": db_session.created_at.isoformat(),
            "uploads": [],
            "target_files": [],
            "column_selection": {},
            "messages": [],
            "chart_history": [],
            "report_sections": {}
        }
        
        # Get uploads
        uploads = db.query(Upload).filter(Upload.session_id == session_id).all()
        memory["uploads"] = [upload.filename for upload in uploads]
        
        # Get messages
        messages = db.query(Message).filter(Message.session_id == session_id).order_by(Message.timestamp).all()
        memory["messages"] = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in messages
        ]
        
        # Get charts
        charts = db.query(Chart).filter(Chart.session_id == session_id).order_by(Chart.created_at).all()
        memory["chart_history"] = [
            {
                "chart_id": chart.chart_id,
                "type": chart.chart_type,
                "config": chart.config or {},
                "html_path": chart.html_path,
                "png_path": chart.png_path,
                "plotly_json": chart.plotly_json_path,
                "feedback": chart.feedback,
                "timestamp": chart.created_at.isoformat()
            }
            for chart in charts
        ]
        
        # Get report sections
        sections = db.query(ReportSection).filter(ReportSection.session_id == session_id).all()
        for section in sections:
            memory["report_sections"][section.section_name] = section.items
        
        # Write memory.json
        memory_file = session_dir / "memory.json"
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2)
        
        logger.info(f"Successfully rolled back session: {session_id}")
        
    except Exception as e:
        logger.error(f"Failed to rollback session {session_id}: {e}", exc_info=True)


def rollback_all_sessions(workspace_root: str = "workspace"):
    """
    Rollback all sessions from database to JSON
    
    Args:
        workspace_root: Root workspace directory path
    """
    workspace_path = Path(workspace_root)
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    db = SessionLocal()
    
    try:
        # Get all sessions from database
        sessions = db.query(DBSession).all()
        logger.info(f"Found {len(sessions)} sessions to rollback")
        
        # Rollback each session
        for session in sessions:
            rollback_session_data(session.id, db, workspace_path)
        
        logger.info("Rollback completed successfully!")
        
    except Exception as e:
        logger.error(f"Rollback failed: {e}", exc_info=True)
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rollback ThermoSense data from PostgreSQL to JSON")
    parser.add_argument("--workspace", default="workspace", help="Workspace directory path")
    
    args = parser.parse_args()
    
    rollback_all_sessions(args.workspace)
