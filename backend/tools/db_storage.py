"""
Database-backed SessionStorage
Hybrid approach: Database for structured data, filesystem for large assets
Maintains backward compatibility with existing SessionStorage interface
"""
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session as DBSession
from backend.db.database import SessionLocal
from backend.db.models import (
    ChatSession,
    Upload,
    Message,
    Chart,
    ReportSection,
    Feedback
)

logger = logging.getLogger(__name__)


class DatabaseSessionStorage:
    """
    Database-backed session storage with filesystem for large assets
    Drop-in replacement for SessionStorage with database persistence
    """
    
    def __init__(self, workspace_dir: str = "workspace", use_database: bool = True):
        """
        Initialize storage
        
        Args:
            workspace_dir: Directory for file storage
            use_database: Whether to use database (True) or fallback to JSON (False)
        """
        self.workspace = Path(workspace_dir).absolute()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.use_database = use_database
        self.feedback_db = self.workspace / "feedback.json"
        
        # Test database connection
        if self.use_database:
            try:
                db = SessionLocal()
                db.close()
                logger.info("Database connection successful - using database storage")
            except Exception as e:
                logger.warning(f"Database unavailable, falling back to JSON: {e}")
                self.use_database = False
    
    def _get_db(self) -> DBSession:
        """Get database session"""
        return SessionLocal()
    
    def get_session_dir(self, session_id: str) -> Path:
        """Get session directory for file storage"""
        p = self.workspace / session_id
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    def _ensure_session_exists(self, session_id: str, db: DBSession):
        """Ensure session exists in database"""
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            session = ChatSession(
                id=session_id,
                title="New Session",
                created_at=datetime.utcnow()
            )
            db.add(session)
            db.commit()
    
    # ========== UPLOADS ==========
    
    def add_upload(self, session_id: str, filename: str):
        """Add uploaded file to session"""
        if not self.use_database:
            return self._json_add_upload(session_id, filename)
        
        db = self._get_db()
        try:
            self._ensure_session_exists(session_id, db)
            
            # Check if upload already exists
            existing = db.query(Upload).filter(
                Upload.session_id == session_id,
                Upload.filename == filename
            ).first()
            
            if not existing:
                file_path = self.get_session_dir(session_id) / filename
                upload = Upload(
                    session_id=session_id,
                    filename=filename,
                    filepath=str(file_path.relative_to(self.workspace)),
                    file_size=file_path.stat().st_size if file_path.exists() else None,
                    uploaded_at=datetime.utcnow()
                )
                db.add(upload)
                db.commit()
        finally:
            db.close()
    
    def remove_upload(self, session_id: str, filename: str):
        """Remove file from session uploads"""
        if not self.use_database:
            return self._json_remove_upload(session_id, filename)
        
        db = self._get_db()
        try:
            upload = db.query(Upload).filter(
                Upload.session_id == session_id,
                Upload.filename == filename
            ).first()
            if upload:
                db.delete(upload)
                db.commit()
        finally:
            db.close()
    
    def get_uploads(self, session_id: str) -> List[str]:
        """Get uploaded files for session"""
        if not self.use_database:
            return self._json_get_uploads(session_id)
        
        db = self._get_db()
        try:
            uploads = db.query(Upload).filter(Upload.session_id == session_id).all()
            return [u.filename for u in uploads]
        finally:
            db.close()
    
    def get_all_files(self) -> List[str]:
        """Get all supported files from workspace"""
        files = []
        for ext in ['*.csv', '*.xlsx', '*.xls']:
            for path in self.workspace.rglob(ext):
                if path.is_file():
                    files.append(path.name)
        return sorted(list(set(files)))
    
    # ========== MESSAGES ==========
    
    def append_message(self, session_id: str, role: str, content: str):
        """Add message to chat history"""
        if not self.use_database:
            return self._json_append_message(session_id, role, content)
        
        db = self._get_db()
        try:
            self._ensure_session_exists(session_id, db)
            
            message = Message(
                session_id=session_id,
                role=role,
                content=content,
                timestamp=datetime.utcnow()
            )
            db.add(message)
            db.commit()
            
            # Update session title if this is first user message
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if session and session.title == "New Session" and role == "user":
                session.title = (content[:30] + "...") if len(content) > 30 else content
                db.commit()
        finally:
            db.close()
    
    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for session"""
        if not self.use_database:
            return self._json_get_messages(session_id)
        
        db = self._get_db()
        try:
            messages = db.query(Message).filter(
                Message.session_id == session_id
            ).order_by(Message.timestamp).all()
            
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in messages
            ]
        finally:
            db.close()
    
    def clear_messages(self, session_id: str):
        """Clear chat history"""
        if not self.use_database:
            return self._json_clear_messages(session_id)
        
        db = self._get_db()
        try:
            db.query(Message).filter(Message.session_id == session_id).delete()
            db.commit()
        finally:
            db.close()
    
    # ========== CHARTS ==========
    
    def add_chart_to_history(self, session_id: str, chart_info: Dict[str, Any]):
        """Add chart to history"""
        if not self.use_database:
            return self._json_add_chart(session_id, chart_info)
        
        db = self._get_db()
        try:
            self._ensure_session_exists(session_id, db)
            
            chart = Chart(
                session_id=session_id,
                chart_id=chart_info.get("chart_id", str(uuid.uuid4())),
                chart_type=chart_info.get("type", "unknown"),
                config=chart_info,
                html_path=chart_info.get("html_path"),
                png_path=chart_info.get("png_path"),
                plotly_json_path=chart_info.get("plotly_json"),
                created_at=datetime.utcnow()
            )
            db.add(chart)
            db.commit()
        finally:
            db.close()
    
    def save_chart_feedback(self, session_id: str, chart_id: str, feedback: str):
        """Save chart feedback"""
        if not self.use_database:
            return self._json_save_feedback(session_id, chart_id, feedback)
        
        db = self._get_db()
        try:
            chart = db.query(Chart).filter(
                Chart.session_id == session_id,
                Chart.chart_id == chart_id
            ).first()
            
            if chart:
                chart.feedback = feedback
                chart.feedback_timestamp = datetime.utcnow()
                db.commit()
                
                # Also save to global feedback DB for learning
                self._save_global_feedback(chart_id, feedback, chart.config or {})
        finally:
            db.close()
    
    def _save_global_feedback(self, chart_id: str, feedback: str, chart_config: Dict):
        """Save to global feedback database"""
        if not self.feedback_db.exists():
            feedback_data = {"charts": []}
        else:
            with open(self.feedback_db, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
        
        feedback_data["charts"].append({
            "chart_id": chart_id,
            "feedback": feedback,
            "user_query": chart_config.get("user_query", ""),
            "files": chart_config.get("files", []),
            "x_column": chart_config.get("x_column"),
            "y_columns": chart_config.get("y_columns", []),
            "chart_type": chart_config.get("chart_type"),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        with open(self.feedback_db, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2)
    
    def get_positive_feedback_patterns(self) -> List[Dict]:
        """Get successful chart patterns"""
        if not self.feedback_db.exists():
            return []
        
        with open(self.feedback_db, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [c for c in data.get("charts", []) if c.get("feedback") == "positive"]
    
    # ========== REPORT SECTIONS ==========
    
    def add_to_report_section(self, session_id: str, section: str, item_type: str, content: Any) -> tuple:
        """Add item to report section"""
        if not self.use_database:
            return self._json_add_to_report(session_id, section, item_type, content)
        
        db = self._get_db()
        try:
            self._ensure_session_exists(session_id, db)
            
            # Normalize section name
            valid_sections = [
                "Objectives", "Requirements", "Test Objects", 
                "Analysis and Results", "Discussion", "Recommendation",
                "Conclusion", "References", "Appendices"
            ]
            
            target_section = "Analysis and Results"  # Default
            section_lower = section.lower()
            
            for s in valid_sections:
                if s.lower() == section_lower or s.lower() in section_lower or section_lower in s.lower():
                    target_section = s
                    break
            
            # Get or create report section
            report_section = db.query(ReportSection).filter(
                ReportSection.session_id == session_id,
                ReportSection.section_name == target_section
            ).first()
            
            if not report_section:
                report_section = ReportSection(
                    session_id=session_id,
                    section_name=target_section,
                    items=[]
                )
                db.add(report_section)
            
            # Prevent chart duplicates across all sections
            if item_type == 'chart' and isinstance(content, dict) and 'chart_id' in content:
                chart_id = content['chart_id']
                all_sections = db.query(ReportSection).filter(
                    ReportSection.session_id == session_id
                ).all()
                
                for sec in all_sections:
                    sec.items = [
                        x for x in sec.items
                        if not (x.get('type') == 'chart' and 
                               isinstance(x.get('content'), dict) and 
                               x['content'].get('chart_id') == chart_id)
                    ]
            
            # Add new item
            item = {
                "id": str(uuid.uuid4())[:8],
                "type": item_type,
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            items = report_section.items or []
            items.append(item)
            report_section.items = items
            report_section.updated_at = datetime.utcnow()
            
            db.commit()
            return True, target_section
        finally:
            db.close()
    
    def get_report_content(self, session_id: str) -> Dict[str, List[Dict]]:
        """Get all report content"""
        if not self.use_database:
            return self._json_get_report_content(session_id)
        
        db = self._get_db()
        try:
            sections = db.query(ReportSection).filter(
                ReportSection.session_id == session_id
            ).all()
            
            result = {}
            for section in sections:
                result[section.section_name] = section.items or []
            
            return result
        finally:
            db.close()
    
    def clear_report_section(self, session_id: str, section: str):
        """Clear report section"""
        if not self.use_database:
            return self._json_clear_report_section(session_id, section)
        
        db = self._get_db()
        try:
            report_section = db.query(ReportSection).filter(
                ReportSection.session_id == session_id,
                ReportSection.section_name == section
            ).first()
            
            if report_section:
                report_section.items = []
                report_section.updated_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()
    
    # ========== SESSION MANAGEMENT ==========
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions"""
        if not self.use_database:
            return self._json_get_all_sessions()
        
        db = self._get_db()
        try:
            sessions = db.query(ChatSession).order_by(
                ChatSession.updated_at.desc()
            ).all()
            
            result = []
            for session in sessions:
                message_count = db.query(Message).filter(
                    Message.session_id == session.id
                ).count()
                
                result.append({
                    "id": session.id,
                    "created": session.created_at.isoformat(),
                    "last_activity": session.updated_at.isoformat(),
                    "message_count": message_count,
                    "title": session.title
                })
            
            return result
        finally:
            db.close()
    
    # ========== COLUMN SELECTION (Stored in JSON for now) ==========
    
    def set_column_selection(self, session_id: str, file: str, x_col: Optional[str], y_cols: List[str]):
        """Set column selection (stored in session directory)"""
        session_dir = self.get_session_dir(session_id)
        selection_file = session_dir / "column_selection.json"
        
        data = {
            "file": file,
            "x": x_col,
            "ys": y_cols,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with open(selection_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def get_column_selection(self, session_id: str) -> Dict[str, Any]:
        """Get column selection"""
        session_dir = self.get_session_dir(session_id)
        selection_file = session_dir / "column_selection.json"
        
        if selection_file.exists():
            with open(selection_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def set_target_files(self, session_id: str, filenames: List[str]):
        """Set target files (stored in session directory)"""
        session_dir = self.get_session_dir(session_id)
        target_file = session_dir / "target_files.json"
        
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(filenames, f, indent=2)
    
    def get_target_files(self, session_id: str) -> List[str]:
        """Get target files"""
        session_dir = self.get_session_dir(session_id)
        target_file = session_dir / "target_files.json"
        
        if target_file.exists():
            with open(target_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    # ========== FILE RECOVERY ==========
    
    def find_file(self, filename: str) -> Optional[Path]:
        """Find file in workspace"""
        for path in self.workspace.rglob(filename):
            if path.is_file():
                return path
        return None
    
    def copy_file_to_session(self, source_path: Path, session_id: str) -> bool:
        """Copy file to session directory"""
        try:
            session_dir = self.get_session_dir(session_id)
            dest_path = session_dir / source_path.name
            
            if dest_path.exists():
                return True
            
            import shutil
            shutil.copy2(source_path, dest_path)
            self.add_upload(session_id, source_path.name)
            return True
        except Exception as e:
            logger.error(f"Error copying file: {e}")
            return False
    
    # ========== JSON FALLBACK METHODS (for backward compatibility) ==========
    
    def _json_add_upload(self, session_id: str, filename: str):
        """JSON fallback for add_upload"""
        from backend.tools.storage import SessionStorage
        json_storage = SessionStorage(str(self.workspace))
        json_storage.add_upload(session_id, filename)
    
    def _json_remove_upload(self, session_id: str, filename: str):
        """JSON fallback for remove_upload"""
        from backend.tools.storage import SessionStorage
        json_storage = SessionStorage(str(self.workspace))
        json_storage.remove_upload(session_id, filename)
    
    def _json_get_uploads(self, session_id: str) -> List[str]:
        """JSON fallback for get_uploads"""
        from backend.tools.storage import SessionStorage
        json_storage = SessionStorage(str(self.workspace))
        return json_storage.get_uploads(session_id)
    
    def _json_append_message(self, session_id: str, role: str, content: str):
        """JSON fallback for append_message"""
        from backend.tools.storage import SessionStorage
        json_storage = SessionStorage(str(self.workspace))
        json_storage.append_message(session_id, role, content)
    
    def _json_get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """JSON fallback for get_messages"""
        from backend.tools.storage import SessionStorage
        json_storage = SessionStorage(str(self.workspace))
        return json_storage.get_messages(session_id)
    
    def _json_clear_messages(self, session_id: str):
        """JSON fallback for clear_messages"""
        from backend.tools.storage import SessionStorage
        json_storage = SessionStorage(str(self.workspace))
        json_storage.clear_messages(session_id)
    
    def _json_add_chart(self, session_id: str, chart_info: Dict[str, Any]):
        """JSON fallback for add_chart_to_history"""
        from backend.tools.storage import SessionStorage
        json_storage = SessionStorage(str(self.workspace))
        json_storage.add_chart_to_history(session_id, chart_info)
    
    def _json_save_feedback(self, session_id: str, chart_id: str, feedback: str):
        """JSON fallback for save_chart_feedback"""
        from backend.tools.storage import SessionStorage
        json_storage = SessionStorage(str(self.workspace))
        json_storage.save_chart_feedback(session_id, chart_id, feedback)
    
    def _json_add_to_report(self, session_id: str, section: str, item_type: str, content: Any) -> tuple:
        """JSON fallback for add_to_report_section"""
        from backend.tools.storage import SessionStorage
        json_storage = SessionStorage(str(self.workspace))
        return json_storage.add_to_report_section(session_id, section, item_type, content)
    
    def _json_get_report_content(self, session_id: str) -> Dict[str, List[Dict]]:
        """JSON fallback for get_report_content"""
        from backend.tools.storage import SessionStorage
        json_storage = SessionStorage(str(self.workspace))
        return json_storage.get_report_content(session_id)
    
    def _json_clear_report_section(self, session_id: str, section: str):
        """JSON fallback for clear_report_section"""
        from backend.tools.storage import SessionStorage
        json_storage = SessionStorage(str(self.workspace))
        json_storage.clear_report_section(session_id, section)
    
    def _json_get_all_sessions(self) -> List[Dict[str, Any]]:
        """JSON fallback for get_all_sessions"""
        from backend.tools.storage import SessionStorage
        json_storage = SessionStorage(str(self.workspace))
        return json_storage.get_all_sessions()
