# llm_orchestrator.py (Continued - Part 2: Enhanced AI Features)
"""
Enhanced LLM Orchestrator with Advanced Features:
- Chat-based chart generation
- Intelligent sensor extraction
- Automated data summarization
- Anomaly detection
- Root cause analysis
"""

import json
from backend.tools.json_utils import safe_json_dumps
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("llm_orchestrator_enhanced")


class EnhancedLLMOrchestrator:
    """
    Advanced AI orchestrator with powerful features for thermal data analysis
    """
    
    def __init__(self, llm_client: Any = None):
        self.llm = llm_client
        self.domain_knowledge = self._load_domain_knowledge()
    
    def _load_domain_knowledge(self) -> str:
        return """
**Thermal Engineering Domain Knowledge:**
- Heat Transfer Modes: Conduction, Convection, Radiation
- Thermal Resistance: R = ΔT/Q
- Transient vs Steady State behavior
- Automotive thermal systems: Coolant, Oil, Transmission
- Safety Limits: Coolant <115°C, Oil <140°C
- Common Anomalies: Spikes, Inversions, Oscillations
"""
    
    def get_enhanced_chat_response(
        self,
        user_query: str,
        session_context: Dict[str, Any],
        chart_context: Optional[Dict[str, Any]] = None,
        data_analysis: Optional[Dict[str, Any]] = None,
        rag_context: str = ""
    ) -> Dict[str, Any]:
        """
        Generate intelligent response with action detection
        """
        if not self.llm:
            return {"type": "chat", "content": "AI Assistant is unavailable."}
        
        # Detect intent
        intent = self._detect_intent(user_query, session_context, chart_context)
        
        if intent:
            action = intent.get('action')
            
            if action == 'generate_report':
                return {
                    "type": "action",
                    "action": "generate_report",
                    "params": intent,
                    "content": "📊 Generating comprehensive thermal analysis report..."
                }
            
            if action == 'plot_chart':
                return {
                    "type": "action",
                    "action": "plot_chart",
                    "params": intent,
                    "content": "📈 Creating chart based on your request..."
                }
            
            if action == 'extract_sensors':
                return {
                    "type": "action",
                    "action": "extract_sensors",
                    "params": intent,
                    "content": "⚡ Extracting sensor data..."
                }
            
            if action == 'compare_files':
                return {
                    "type": "action",
                    "action": "compare_files",
                    "params": intent,
                    "content": "⚖️ Comparing files..."
                }
            
            if action == 'analyze_data':
                return {
                    "type": "action",
                    "action": "analyze_data",
                    "params": intent,
                    "content": "🔍 Analyzing data for insights..."
                }
        
        # Default: Intelligent chat response
        return self._generate_chat_response(user_query, session_context, chart_context, data_analysis)
    
    def _detect_intent(
        self,
        query: str,
        session_context: Dict[str, Any],
        chart_context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Intelligent intent detection using keywords and LLM
        """
        query_lower = query.lower()
        
        # Report generation
        if any(word in query_lower for word in ['report', 'generate report', 'create report', 'document']):
            return {"action": "generate_report"}
        
        # Chart generation
        if any(word in query_lower for word in ['plot', 'chart', 'graph', 'visualize', 'show me']):
            return {"action": "plot_chart", "query": query}
        
        # Sensor extraction
        if any(word in query_lower for word in ['extract', 'sensors', 'get data from']):
            return {"action": "extract_sensors", "query": query}
        
        #File comparison
        if any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus', 'difference']):
            return {"action": "compare_files", "query": query}
        
        # Data analysis
        if any(word in query_lower for word in ['analyze', 'analysis', 'insights', 'anomalies', 'trends']):
            return {"action": "analyze_data", "query": query}
        
        return None
    
    def _generate_chat_response(
        self,
        user_query: str,
        session_context: Dict[str, Any],
        chart_context: Optional[Dict[str, Any]],
        data_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate intelligent chat response with context awareness
        """
        try:
            # Build context
            context_info = self._build_context(session_context, chart_context, data_analysis)
            
            # Create prompt
            prompt = f"""You are ThermoSense AI, an expert thermal analysis assistant.

**Domain Knowledge:**
{self.domain_knowledge}

**Current Context:**
{context_info}

**User Query:** {user_query}

Provide a helpful, technical response. Be concise but informative."""

            resp = self.llm.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400
            )
            
            return {
                "type": "chat",
                "content": resp.choices[0].message.content.strip()
            }
        
        except Exception as e:
            logger.error(f"Chat response error: {e}")
            return {
                "type": "chat",
                "content": "I encountered an error. Please try again or use the UI for specific actions."
            }
    
    def _build_context(
        self,
        session_context: Dict[str, Any],
        chart_context: Optional[Dict[str, Any]],
        data_analysis: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build context string for LLM
        """
        parts = []
        
        # Files
        files = session_context.get('files', {})
        if files:
            parts.append(f"Uploaded Files: {', '.join(files.keys())}")
        
        # Current chart
        if chart_context:
            x = chart_context.get('x_column', 'X')
            y = chart_context.get('y_columns', [])
            parts.append(f"Current Chart: {', '.join(y)} vs {x}")
        
        # Data stats
        if data_analysis and 'basic_stats' in data_analysis:
            parts.append("Data statistics available")
        
        return "\n".join(parts) if parts else "No data uploaded yet"
    
    def generate_data_summary(
        self,
        filename: str,
        columns: list,
        sample_data: Dict[str, Any]
    ) -> str:
        """
        Generate intelligent summary when data is uploaded
        """
        if not self.llm:
            return f"Uploaded {filename} with {len(columns)} columns."
        
        try:
            # Analyze column names to understand data type
            prompt = f"""Analyze this uploaded file and provide a brief 2-sentence summary:

Filename: {filename}
Columns ({len(columns)}): {', '.join(columns[:20])}
Sample stats: {safe_json_dumps(sample_data, indent=2)[:500]}

Based on column names, what kind of thermal data is this? What can we analyze?
Be concise and specific."""

            resp = self.llm.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=150
            )
            
            return resp.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return f"✅ Uploaded **{filename}** with {len(columns)} columns. Ready for analysis!"


# Singleton
_enhanced_orchestrator = None

def get_enhanced_orchestrator(llm_client=None):
    global _enhanced_orchestrator
    if _enhanced_orchestrator is None:
        _enhanced_orchestrator = EnhancedLLMOrchestrator(llm_client)
    return _enhanced_orchestrator
