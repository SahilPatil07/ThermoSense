"""
Data Context Manager
Stores and manages context about uploaded data, generated charts, and analyses
Enables intelligent querying and context-aware responses
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataContextManager:
    """
    Manages context about data, charts, and analyses for intelligent interactions
    """
    
    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.contexts = {}  # In-memory cache: {session_id: context_data}
    
    def store_chart_context(
        self,
        session_id: str,
        chart_id: str,
        filename: str,
        x_column: str,
        y_columns: List[str],
        chart_type: str,
        df_summary: Dict[str, Any],
        advanced_analysis: Dict[str, Any],
        chart_path: str
    ):
        """
        Store comprehensive context about a generated chart
        """
        if session_id not in self.contexts:
            self.contexts[session_id] = {
                'charts': {},
                'files': {},
                'queries': []
            }
        
        chart_context = {
            'chart_id': chart_id,
            'filename': filename,
            'x_column': x_column,
            'y_columns': y_columns,
            'chart_type': chart_type,
            'chart_path': chart_path,
            'timestamp': datetime.now().isoformat(),
            'df_summary': df_summary,
            'advanced_analysis': advanced_analysis
        }
        
        self.contexts[session_id]['charts'][chart_id] = chart_context
        
        # Save to disk
        self._save_context(session_id)
    
    def store_file_context(
        self,
        session_id: str,
        filename: str,
        columns: List[str],
        row_count: int,
        column_types: Dict[str, str],
        data_preview: Dict[str, Any]
    ):
        """
        Store context about uploaded file
        """
        if session_id not in self.contexts:
            self.contexts[session_id] = {
                'charts': {},
                'files': {},
                'queries': []
            }
        
        file_context = {
            'filename': filename,
            'columns': columns,
            'row_count': row_count,
            'column_types': column_types,
            'data_preview': data_preview,
            'upload_timestamp': datetime.now().isoformat()
        }
        
        self.contexts[session_id]['files'][filename] = file_context
        self._save_context(session_id)
    
    def add_query(self, session_id: str, query: str, response: str):
        """
        Store query-response pairs for context building
        """
        if session_id not in self.contexts:
            self.contexts[session_id] = {
                'charts': {},
                'files': {},
                'queries': []
            }
        
        self.contexts[session_id]['queries'].append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 20 queries
        if len(self.contexts[session_id]['queries']) > 20:
            self.contexts[session_id]['queries'] = self.contexts[session_id]['queries'][-20:]
        
        self._save_context(session_id)
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get full context for a session
        """
        if session_id not in self.contexts:
            self._load_context(session_id)
        
        return self.contexts.get(session_id, {
            'charts': {},
            'files': {},
            'queries': []
        })
    
    def get_latest_chart_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get context of the most recently generated chart
        """
        context = self.get_context(session_id)
        charts = context.get('charts', {})
        
        if not charts:
            return None
        
        # Find most recent
        latest = max(charts.values(), key=lambda x: x['timestamp'])
        return latest
    
    def get_chart_context(self, session_id: str, chart_id: str) -> Optional[Dict[str, Any]]:
        """
        Get context for a specific chart
        """
        context = self.get_context(session_id)
        return context.get('charts', {}).get(chart_id)
    
    def get_file_context(self, session_id: str, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get context for a specific file
        """
        context = self.get_context(session_id)
        return context.get('files', {}).get(filename)
    
    def get_all_chart_contexts(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all chart contexts for a session, sorted by timestamp
        """
        context = self.get_context(session_id)
        charts = list(context.get('charts', {}).values())
        return sorted(charts, key=lambda x: x['timestamp'], reverse=True)
    
    def build_context_summary(self, session_id: str) -> str:
        """
        Build a human-readable summary of the session context for LLM
        """
        context = self.get_context(session_id)
        
        summary_parts = []
        
        # Files
        files = context.get('files', {})
        if files:
            summary_parts.append(f"**Uploaded Files ({len(files)}):**")
            for filename, file_ctx in files.items():
                summary_parts.append(
                    f"  - {filename}: {file_ctx['row_count']} rows, {len(file_ctx['columns'])} columns"
                )
                summary_parts.append(f"    Columns: {', '.join(file_ctx['columns'][:10])}")
        
        # Charts
        charts = context.get('charts', {})
        if charts:
            summary_parts.append(f"\n**Generated Charts ({len(charts)}):**")
            for chart_id, chart_ctx in list(charts.items())[-5:]:  # Last 5 charts
                summary_parts.append(
                    f"  - Chart {chart_id}: {chart_ctx['chart_type']} chart"
                )
                summary_parts.append(
                    f"    X-axis: {chart_ctx['x_column']}, Y-axis: {', '.join(chart_ctx['y_columns'])}"
                )
                
                # Add key insights from advanced analysis
                if 'advanced_analysis' in chart_ctx:
                    adv = chart_ctx['advanced_analysis']
                    
                    if 'anomalies' in adv:
                        total_anomalies = sum(a.get('count', 0) for a in adv['anomalies'].values())
                        if total_anomalies > 0:
                            summary_parts.append(f"    ⚠️ {total_anomalies} anomalies detected")
                    
                    if 'recommendations' in adv and adv['recommendations']:
                        summary_parts.append(f"    💡 {len(adv['recommendations'])} recommendations")
        
        # Recent queries
        queries = context.get('queries', [])
        if queries:
            summary_parts.append(f"\n**Recent Interactions ({len(queries[-5:])}):**")
            for q in queries[-5:]:
                summary_parts.append(f"  - Q: {q['query'][:100]}...")
        
        return "\n".join(summary_parts)
    
    def _save_context(self, session_id: str):
        """
        Save context to disk
        """
        try:
            session_dir = self.workspace_dir / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            
            context_file = session_dir / "context.json"
            
            with open(context_file, 'w') as f:
                json.dump(self.contexts[session_id], f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving context: {e}")
    
    def _load_context(self, session_id: str):
        """
        Load context from disk
        """
        try:
            session_dir = self.workspace_dir / session_id
            context_file = session_dir / "context.json"
            
            if context_file.exists():
                with open(context_file, 'r') as f:
                    self.contexts[session_id] = json.load(f)
            else:
                self.contexts[session_id] = {
                    'charts': {},
                    'files': {},
                    'queries': []
                }
        except Exception as e:
            logger.error(f"Error loading context: {e}")
            self.contexts[session_id] = {
                'charts': {},
                'files': {},
                'queries': []
            }


# Global instance
_context_manager = None

def get_context_manager(workspace_dir: str = "workspace") -> DataContextManager:
    global _context_manager
    if _context_manager is None:
        _context_manager = DataContextManager(workspace_dir)
    return _context_manager
