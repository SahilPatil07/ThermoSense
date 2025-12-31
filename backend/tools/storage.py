# backend/tools/storage.py
"""
SessionStorage - Enhanced with feedback tracking and learning
"""
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

class SessionStorage:
    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace = Path(workspace_dir).absolute()  # Make absolute to avoid path issues
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.feedback_db = self.workspace / "feedback.json"
    
    def get_session_dir(self, session_id: str) -> Path:
        p = self.workspace / session_id
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    def memory_file_path(self, session_id: str) -> Path:
        return self.get_session_dir(session_id) / "memory.json"
    
    def _default_memory(self) -> Dict[str, Any]:
        now = datetime.utcnow().isoformat()
        return {
            "created": now,
            "last_activity": now,
            "uploads": [],
            "last_uploaded": None,
            "current_target_files": [],
            "column_selection": {},
            "messages": [],
            "chart_history": [],
            "inferences": [],
            "proactive_insights": [],
            "analysis_results": {},
            "report_content": {
                "Objectives": [],
                "Requirements": [],
                "Test Objects": [],
                "Analysis and Results": [],
                "Discussion": [],
                "Recommendation": [],
                "Conclusion": [],
                "References": [],
                "Appendices": []
            }
        }
    
    def load_memory(self, session_id: str) -> Dict[str, Any]:
        path = self.memory_file_path(session_id)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return self._default_memory()
    
    def save_memory(self, session_id: str, memory: Dict[str, Any]):
        memory["last_activity"] = datetime.utcnow().isoformat()
        path = self.memory_file_path(session_id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
    
    def add_upload(self, session_id: str, filename: str):
        mem = self.load_memory(session_id)
        if filename not in mem["uploads"]:
            mem["uploads"].append(filename)
        mem["last_uploaded"] = filename
        self.save_memory(session_id, mem)
    
    def remove_upload(self, session_id: str, filename: str):
        """Remove a file from session uploads list"""
        mem = self.load_memory(session_id)
        if filename in mem["uploads"]:
            mem["uploads"].remove(filename)
        self.save_memory(session_id, mem)
    
    def get_all_files(self) -> List[str]:
        """Get all supported files from the entire workspace"""
        files = []
        # Recursive search for supported files
        for ext in ['*.csv', '*.xlsx', '*.xls', '*.pptx']:
            for path in self.workspace.rglob(ext):
                if path.is_file():
                    # We return the relative path from workspace or just the filename if unique
                    # For simplicity and to match current UI, let's return just filenames
                    # and handle duplicates if needed. For now, set of filenames.
                    files.append(path.name)
        return sorted(list(set(files)))

    def get_uploads(self, session_id: str) -> List[str]:
        """Get uploaded files for specific session"""
        return self.load_memory(session_id).get("uploads", [])
    
    def set_target_files(self, session_id: str, filenames: List[str]):
        mem = self.load_memory(session_id)
        mem["current_target_files"] = filenames
        self.save_memory(session_id, mem)
    
    def get_target_files(self, session_id: str) -> List[str]:
        return self.load_memory(session_id).get("current_target_files", [])
    
    def set_column_selection(self, session_id: str, file: str, x_col: Optional[str], y_cols: List[str]):
        mem = self.load_memory(session_id)
        mem["column_selection"] = {
            "file": file,
            "x": x_col,
            "ys": y_cols,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.save_memory(session_id, mem)
    
    def get_column_selection(self, session_id: str) -> Dict[str, Any]:
        return self.load_memory(session_id).get("column_selection", {})
    
    def append_message(self, session_id: str, role: str, content: str):
        mem = self.load_memory(session_id)
        mem["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.save_memory(session_id, mem)
    
    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chat messages for session"""
        return self.load_memory(session_id).get("messages", [])
    
    def clear_messages(self, session_id: str):
        """Clear chat history for new session"""
        mem = self.load_memory(session_id)
        mem["messages"] = []
        self.save_memory(session_id, mem)
    
    def add_chart_to_history(self, session_id: str, chart_info: Dict[str, Any]):
        """Track generated charts for feedback learning"""
        mem = self.load_memory(session_id)
        mem["chart_history"].append({
            **chart_info,
            "timestamp": datetime.utcnow().isoformat(),
            "feedback": None
        })
        self.save_memory(session_id, mem)
    
    def save_chart_feedback(self, session_id: str, chart_id: str, feedback: str):
        """Save thumbs up/down feedback for chart"""
        mem = self.load_memory(session_id)
        for chart in mem.get("chart_history", []):
            if chart.get("chart_id") == chart_id:
                chart["feedback"] = feedback
                chart["feedback_time"] = datetime.utcnow().isoformat()
        self.save_memory(session_id, mem)
        self._save_global_feedback(chart_id, feedback, mem)
    
    def _save_global_feedback(self, chart_id: str, feedback: str, session_mem: Dict):
        """Save feedback to global learning database"""
        if not self.feedback_db.exists():
            feedback_data = {"charts": []}
        else:
            with open(self.feedback_db, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
        
        for chart in session_mem.get("chart_history", []):
            if chart.get("chart_id") == chart_id:
                feedback_data["charts"].append({
                    "chart_id": chart_id,
                    "feedback": feedback,
                    "user_query": chart.get("user_query", ""),
                    "files": chart.get("files", []),
                    "x_column": chart.get("x_column"),
                    "y_columns": chart.get("y_columns", []),
                    "chart_type": chart.get("chart_type"),
                    "timestamp": datetime.utcnow().isoformat()
                })
                break
        
        with open(self.feedback_db, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2)
    
    def get_positive_feedback_patterns(self) -> List[Dict]:
        """Retrieve successful chart patterns for LLM learning"""
        if not self.feedback_db.exists():
            return []
        
        with open(self.feedback_db, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [c for c in data.get("charts", []) if c.get("feedback") == "positive"]

    def get_recommendations_for_signals(self, y_columns: List[str]) -> List[str]:
        """
        Find the most successful chart types for a given set of signals based on history.
        """
        patterns = self.get_positive_feedback_patterns()
        if not patterns:
            return []
            
        # Count chart types for these specific signals
        type_counts = {}
        target_ys = set(y.lower() for y in y_columns)
        
        for p in patterns:
            p_ys = set(y.lower() for y in p.get("y_columns", []))
            # If there's a significant overlap or exact match
            if target_ys.intersection(p_ys):
                ctype = p.get("chart_type")
                type_counts[ctype] = type_counts.get(ctype, 0) + 1
                
        # Sort by frequency
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_types]

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """List all available chat sessions with metadata"""
        sessions = []
        for item in self.workspace.iterdir():
            if item.is_dir() and (item / "memory.json").exists():
                mem = self.load_memory(item.name)
                sessions.append({
                    "id": item.name,
                    "created": mem.get("created"),
                    "last_activity": mem.get("last_activity"),
                    "message_count": len(mem.get("messages", [])),
                    "title": self._generate_session_title(mem)
                })
        
        # Sort by last activity (newest first)
        return sorted(sessions, key=lambda x: x["last_activity"] or "", reverse=True)

    def _generate_session_title(self, memory: Dict) -> str:
        """Generate a title for the session based on first user message"""
        messages = memory.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                return (content[:30] + "...") if len(content) > 30 else content
        return "New Session"

    # ========== REPORT MANAGEMENT ==========
    
    def add_to_report_section(self, session_id: str, section: str, item_type: str, content: Any) -> bool:
        """
        Add an item to a specific report section
        item_type: 'text', 'chart', 'image', 'table'
        """
        mem = self.load_memory(session_id)
        
        # Normalize section name (simple fuzzy match)
        target_section = "Analysis and Results"  # Default
        section_lower = section.lower()
        
        valid_sections = [
            "Executive Summary",
            "Objectives", 
            "Requirements", 
            "Test Objects", 
            "Analysis and Results", 
            "Discussion", 
            "Recommendation", 
            "Conclusion", 
            "References", 
            "Appendices",
            "Test equipment",
            "Purpose"
        ]
        
        # First try exact match
        for s in valid_sections:
            if s.lower() == section_lower:
                target_section = s
                break
        else:
            # Then try fuzzy match
            for s in valid_sections:
                if s.lower() in section_lower or section_lower in s.lower():
                    target_section = s
                    break
        
        if "report_content" not in mem:
            mem["report_content"] = {s: [] for s in valid_sections}
        
        if target_section not in mem["report_content"]:
            mem["report_content"][target_section] = []
        
        if "report_content" not in mem:
            mem["report_content"] = {s: [] for s in valid_sections}
            
        if target_section not in mem["report_content"]:
            mem["report_content"][target_section] = []
            
        # Prevent duplicates for charts
        if item_type == 'chart' and isinstance(content, dict) and 'chart_id' in content:
            chart_id = content['chart_id']
            # Remove this chart from ALL sections first to ensure it only appears once in the entire report
            for sec in mem["report_content"]:
                # Filter out any existing chart with the same ID
                mem["report_content"][sec] = [
                    x for x in mem["report_content"][sec] 
                    if not (x['type'] == 'chart' and isinstance(x['content'], dict) and x['content'].get('chart_id') == chart_id)
                ]
            
        item = {
            "id": str(uuid.uuid4())[:8],
            "type": item_type,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        mem["report_content"][target_section].append(item)
        self.save_memory(session_id, mem)
        return True, target_section

    def get_report_content(self, session_id: str) -> Dict[str, List[Dict]]:
        """Get all report content organized by section"""
        mem = self.load_memory(session_id)
        return mem.get("report_content", {})

    def clear_report_section(self, session_id: str, section: str):
        """Clear a specific section"""
        mem = self.load_memory(session_id)
        if "report_content" in mem and section in mem["report_content"]:
            mem["report_content"][section] = []
            self.save_memory(session_id, mem)

    def save_inference(self, session_id: str, content: str, chart_id: Optional[str] = None, section: Optional[str] = None):
        """Save a user inference/insight"""
        mem = self.load_memory(session_id)
        if "inferences" not in mem:
            mem["inferences"] = []
            
        inference = {
            "id": str(uuid.uuid4())[:8],
            "content": content,
            "chart_id": chart_id,
            "section": section,
            "timestamp": datetime.utcnow().isoformat()
        }
        mem["inferences"].append(inference)
        
        # If section is provided, also add to report_content
        if section:
            self.add_to_report_section(session_id, section, "text", content)
            
        self.save_memory(session_id, mem)
        return inference

    def get_inferences(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all inferences for a session"""
        return self.load_memory(session_id).get("inferences", [])

    def set_analysis(self, session_id: str, filename: str, analysis: Dict[str, Any]):
        """Store comprehensive analysis results for a file"""
        mem = self.load_memory(session_id)
        if "analysis_results" not in mem:
            mem["analysis_results"] = {}
        mem["analysis_results"][filename] = {
            "data": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.save_memory(session_id, mem)

    def get_analysis(self, session_id: str, filename: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored analysis for a file"""
        mem = self.load_memory(session_id)
        return mem.get("analysis_results", {}).get(filename)

    def add_proactive_insight(self, session_id: str, insight: Dict[str, Any]):
        """Add a proactive insight to the session and report"""
        mem = self.load_memory(session_id)
        if "proactive_insights" not in mem:
            mem["proactive_insights"] = []
        
        insight_data = {
            **insight,
            "timestamp": datetime.utcnow().isoformat(),
            "read": False
        }
        mem["proactive_insights"].append(insight_data)
        self.save_memory(session_id, mem)
        
        # Also add to report section automatically
        self.add_to_report_section(session_id, "Analysis and Results", "text", insight.get("content", ""))

    def get_latest_proactive_insight(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest unread proactive insight"""
        mem = self.load_memory(session_id)
        insights = mem.get("proactive_insights", [])
        for insight in reversed(insights):
            if not insight.get("read", False):
                insight["read"] = True
                self.save_memory(session_id, mem)
                return insight
        return None

    # ========== FILE RECOVERY ==========

    def find_file(self, filename: str) -> Optional[Path]:
        """
        Search for a file across all sessions in the workspace.
        Returns the path to the first occurrence found, or None.
        """
        # Search in all subdirectories of workspace
        for path in self.workspace.rglob(filename):
            if path.is_file():
                return path
        return None

    def copy_file_to_session(self, source_path: Path, session_id: str) -> bool:
        """
        Copy a file from source_path to the session directory.
        Returns True if successful, False otherwise.
        """
        try:
            session_dir = self.get_session_dir(session_id)
            dest_path = session_dir / source_path.name
            
            # Don't overwrite if it already exists (though this method is likely called because it doesn't)
            if dest_path.exists():
                return True
                
            import shutil
            shutil.copy2(source_path, dest_path)
            
            # Register upload in memory
            self.add_upload(session_id, source_path.name)
            return True
        except Exception as e:
            print(f"Error copying file: {e}")
            return False
