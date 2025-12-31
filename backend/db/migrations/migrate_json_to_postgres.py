"""
Data migration script: JSON files → PostgreSQL
Migrates existing session data from JSON storage to database
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from sqlalchemy.orm import Session
from backend.db.database import SessionLocal, init_db
from backend.db.models import (
    Session as DBSession,
    Upload,
    Message,
    Chart,
    ReportSection,
    Feedback
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_session_data(workspace_dir: Path, db: Session):
    """
    Migrate a single session's data from JSON to database
    
    Args:
        workspace_dir: Path to session workspace directory
        db: Database session
    """
    session_id = workspace_dir.name
    memory_file = workspace_dir / "memory.json"
    
    if not memory_file.exists():
        logger.warning(f"No memory.json found for session {session_id}")
        return
    
    try:
        # Load JSON data
        with open(memory_file, 'r', encoding='utf-8') as f:
            memory = json.load(f)
        
        logger.info(f"Migrating session: {session_id}")
        
        # Create session record
        db_session = DBSession(
            id=session_id,
            title=memory.get('title', 'Untitled Session'),
            created_at=datetime.fromisoformat(memory.get('created_at', datetime.utcnow().isoformat())),
            updated_at=datetime.utcnow()
        )
        db.add(db_session)
        
        # Migrate uploads
        for filename in memory.get('uploads', []):
            upload = Upload(
                session_id=session_id,
                filename=filename,
                filepath=str(workspace_dir / filename),
                uploaded_at=datetime.utcnow()
            )
            db.add(upload)
        
        # Migrate messages
        for msg in memory.get('messages', []):
            message = Message(
                session_id=session_id,
                role=msg.get('role', 'user'),
                content=msg.get('content', ''),
                timestamp=datetime.fromisoformat(msg.get('timestamp', datetime.utcnow().isoformat()))
            )
            db.add(message)
        
        # Migrate chart history
        for chart_data in memory.get('chart_history', []):
            chart = Chart(
                session_id=session_id,
                chart_id=chart_data.get('chart_id', ''),
                chart_type=chart_data.get('type', 'unknown'),
                config=chart_data.get('config', {}),
                html_path=chart_data.get('html_path'),
                png_path=chart_data.get('png_path'),
                plotly_json_path=chart_data.get('plotly_json'),
                feedback=chart_data.get('feedback'),
                created_at=datetime.fromisoformat(chart_data.get('timestamp', datetime.utcnow().isoformat()))
            )
            db.add(chart)
        
        # Migrate report sections
        for section_name, items in memory.get('report_sections', {}).items():
            report_section = ReportSection(
                session_id=session_id,
                section_name=section_name,
                items=items,
                created_at=datetime.utcnow()
            )
            db.add(report_section)
        
        # Commit all changes for this session
        db.commit()
        logger.info(f"Successfully migrated session: {session_id}")
        
    except Exception as e:
        logger.error(f"Failed to migrate session {session_id}: {e}", exc_info=True)
        db.rollback()


def migrate_all_sessions(workspace_root: str = "workspace"):
    """
    Migrate all sessions from workspace directory
    
    Args:
        workspace_root: Root workspace directory path
    """
    workspace_path = Path(workspace_root)
    
    if not workspace_path.exists():
        logger.error(f"Workspace directory not found: {workspace_root}")
        return
    
    # Initialize database
    logger.info("Initializing database...")
    init_db()
    
    # Get database session
    db = SessionLocal()
    
    try:
        # Find all session directories
        session_dirs = [d for d in workspace_path.iterdir() if d.is_dir()]
        logger.info(f"Found {len(session_dirs)} sessions to migrate")
        
        # Migrate each session
        for session_dir in session_dirs:
            migrate_session_data(session_dir, db)
        
        logger.info("Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


def verify_migration(workspace_root: str = "workspace"):
    """
    Verify migration by comparing counts
    
    Args:
        workspace_root: Root workspace directory path
    """
    workspace_path = Path(workspace_root)
    db = SessionLocal()
    
    try:
        # Count JSON sessions
        json_sessions = len([d for d in workspace_path.iterdir() if d.is_dir() and (d / "memory.json").exists()])
        
        # Count database sessions
        db_sessions = db.query(DBSession).count()
        
        logger.info(f"JSON sessions: {json_sessions}")
        logger.info(f"Database sessions: {db_sessions}")
        
        if json_sessions == db_sessions:
            logger.info("✅ Migration verification passed!")
        else:
            logger.warning(f"⚠️ Session count mismatch: {json_sessions} JSON vs {db_sessions} DB")
        
        # Show sample data
        sample_session = db.query(DBSession).first()
        if sample_session:
            logger.info(f"Sample session: {sample_session.id} - {sample_session.title}")
            upload_count = db.query(Upload).filter(Upload.session_id == sample_session.id).count()
            message_count = db.query(Message).filter(Message.session_id == sample_session.id).count()
            chart_count = db.query(Chart).filter(Chart.session_id == sample_session.id).count()
            logger.info(f"  Uploads: {upload_count}, Messages: {message_count}, Charts: {chart_count}")
        
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate ThermoSense data from JSON to PostgreSQL")
    parser.add_argument("--workspace", default="workspace", help="Workspace directory path")
    parser.add_argument("--verify-only", action="store_true", help="Only verify migration, don't migrate")
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_migration(args.workspace)
    else:
        migrate_all_sessions(args.workspace)
        verify_migration(args.workspace)
