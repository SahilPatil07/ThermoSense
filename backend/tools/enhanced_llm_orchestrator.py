"""
Enhanced LLM Orchestrator with Deep Context Awareness
Provides domain expertise in thermal engineering, test data analysis, and intelligent querying
"""

import json
from backend.tools.json_utils import safe_json_dumps
import re
import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

# fuzzy matching
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ = True
except Exception:
    import difflib
    RAPIDFUZZ = False

logger = logging.getLogger("llm_orchestrator")


class EnhancedLLMOrchestrator:
    """
    Advanced LLM Orchestrator with domain knowledge and deep context awareness
    """
    
    # Domain knowledge base for thermal engineering
    THERMAL_DOMAIN_KNOWLEDGE = """
**Thermal Engineering Fundamentals:**
- Heat Transfer: Conduction (Fourier's Law), Convection (Newton's Law), Radiation (Stefan-Boltzmann)
- Temperature vs Heat: Temperature is potential, heat is energy flow
- Thermal Resistance: R = ΔT/Q (analogous to electrical resistance)
- Thermal Capacitance: Time constant τ = RC
- Steady State vs Transient: Steady state = no temporal change, Transient = time-dependent

**Automotive Thermal Systems:**
- Coolant System: Engine block → Radiator → Pump → Thermostat
- Thermal Runaway: Positive feedback loop where temperature increase accelerates further heating
- Heat Exchanger Effectiveness: ε = (T_hot_in - T_hot_out) / (T_hot_in - T_cold_in)
- Thermal Cycling: Repeated heating/cooling can cause material fatigue
- Ambient Effects: Heat rejection capability decreases with higher ambient temperature

**Common Test Parameters:**
- RPM (Revolutions Per Minute): Engine speed, directly relates to heat generation
- Vehicle Speed: Affects ram air cooling
- Coolant Temperature: Primary indicator of thermal state
- Oil Temperature: Lubrication thermal management
- Intake Air Temperature: Affects combustion efficiency
- Exhaust Gas Temperature: Indicates combustion chamber conditions

**Typical Anomalies:**
- Thermal Spikes: Sudden temperature increases (possible sensor fault or thermal event)
- Temperature Inversions: Outlet hotter than inlet (Check sensor calibration)
- Thermal Lag: Delayed response to load changes (Check coolant flow)
- Oscillations: Cyclic temperature fluctuations (Thermostat hunting or control issues)
- Asymmetric Heating: Uneven temperature distribution (Air pockets or flow blockage)

**Root Cause Analysis Patterns:**
1. Temperature Rise with RPM → Normal combustion heat + friction
2. Temperature Drop with Speed → Increased ram air cooling
3. Temperature Plateau → Thermostat regulation or thermal equilibrium
4. Sharp Spikes → Load transient, sensor glitch, or thermal shock
5. Gradual Degradation → Fouling, wear, or coolant degradation
"""
    
    def __init__(self, llm_client: Any = None, planner: Any = None):
        self.llm = llm_client
        self.planner = planner
        self.domain_knowledge = self.THERMAL_DOMAIN_KNOWLEDGE
    
    def get_enhanced_chat_response(
        self,
        user_query: str,
        session_context: Dict[str, Any],
        chart_context: Optional[Dict[str, Any]] = None,
        data_analysis: Optional[Dict[str, Any]] = None,
        rag_context: str = "",
        chat_history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate deeply context-aware response with domain expertise and intent detection.
        Returns a dictionary with 'type', 'content', and optional 'action'.
        """
        if not self.llm:
            return {"type": "chat", "content": self._fallback_response(user_query, session_context)}
        
        # 0. Check for Inference Loop State
        query_lower = user_query.lower()
        last_bot_msg = ""
        if chat_history:
            for msg in reversed(chat_history):
                if msg.get('role') == 'bot':
                    last_bot_msg = msg.get('content', '').lower()
                    break
        
        # State A: User just generated a chart, bot asked "Do you want to add your inference?"
        if "do you want to add your inference" in last_bot_msg:
            if any(word in query_lower for word in ['yes', 'sure', 'ok', 'yeah']) or len(user_query.split()) > 5:
                # If it's a long message, assume it's the inference itself
                if len(user_query.split()) > 5:
                    return {
                        "type": "chat",
                        "content": "That's a valuable insight! In which section of the report would you like to add this information? (e.g., Discussion, Conclusion, Analysis and Results)",
                        "action": "prompt_inference_section",
                        "inference_content": user_query
                    }
                else:
                    return {
                        "type": "chat",
                        "content": "Great! Please type your inference or observation about the data."
                    }
            elif any(word in query_lower for word in ['no', 'skip', 'not now']):
                return {"type": "chat", "content": "No problem. Let me know if you need anything else!"}

        # State B: Bot asked for section, user is providing it
        if "in which section of the report would you like to add this information" in last_bot_msg:
            # Try to find a valid section in the query
            valid_sections = ["Objectives", "Requirements", "Test Objects", "Analysis and Results", "Discussion", "Recommendation", "Conclusion"]
            matched_section = None
            for s in valid_sections:
                if s.lower() in query_lower:
                    matched_section = s
                    break
            
            if matched_section:
                # Find the inference content from history
                inference_content = ""
                for msg in reversed(chat_history):
                    if msg.get('role') == 'user' and len(msg.get('content', '').split()) > 5:
                        inference_content = msg.get('content')
                        break
                
                return {
                    "type": "action",
                    "action": "save_inference",
                    "params": {
                        "content": inference_content,
                        "section": matched_section
                    },
                    "content": f"I've added your inference to the **{matched_section}** section of the report."
                }
            else:
                return {
                    "type": "chat",
                    "content": "I didn't recognize that section. Please choose from: Discussion, Conclusion, Analysis and Results, or Recommendation."
                }

        # 1. Detect Intent (Action vs Chat) - now with chart context
        intent = self._detect_intent(user_query, session_context, chart_context)
        
        # Unified Agentic Flow: If planner is available and intent suggests multi-step, use planner
        if self.planner and intent and intent.get('action') in ['plot_chart', 'extract_sensors', 'generate_report']:
            logger.info(f"Using AgentPlanner for action: {intent.get('action')}")
            return {
                "type": "agent_plan",
                "intent": intent,
                "query": user_query
            }

        if intent and intent.get('action') == 'plot_chart':
            return {
                "type": "action",
                "action": "plot_chart",
                "params": intent,
                "content": "I'm generating the chart based on your request..."
            }
        
        if intent and intent.get('action') == 'preview_data':
            return {
                "type": "action",
                "action": "preview_data",
                "params": intent,
                "content": "Showing data preview..."
            }
        
        if intent and intent.get('action') == 'extract_sensors':
            return {
                "type": "action",
                "action": "extract_sensors",
                "params": intent,
                "content": "Extracting sensors from your file..."
            }
        
        if intent and intent.get('action') == 'delete_file':
            return {
                "type": "action",
                "action": "delete_file",
                "params": intent,
                "content": "Deleting file..."
            }
            
        if intent and intent.get('action') == 'generate_report':
            return {
                "type": "action",
                "action": "generate_report",
                "params": intent,
                "content": "Generating comprehensive thermal analysis report..."
            }
        
        # 2. Build comprehensive context
        context_prompt = self._build_context_prompt(session_context, chart_context, data_analysis, rag_context)
        
        # 3. Create expert system prompt
        system_message = f"""You are ThermoSense AI, an expert thermal analysis system for automotive testing.

**Your Expertise:**
- Thermal Engineering (Heat Transfer, Thermodynamics)
- Automotive Test Data Analysis
- Statistical Analysis and Root Cause Analysis
- Anomaly Detection and Pattern Recognition

**Your Role:**
- Provide deep, actionable insights, not just surface statistics
- Explain physical phenomena behind the data
- Identify root causes of thermal behavior
- Detect anomalies and their implications
- Answer specific questions about data/charts with precision
- Use the provided RAG context to ground your answers in project documentation

{self.domain_knowledge}

**Context for this conversation:**
{context_prompt}

**Guidelines:**
1. Be technical but clear
2. Relate data patterns to physical phenomena
3. Provide root cause analysis when relevant
4. Highlight safety/performance concerns
5. Use data-driven conclusions
6. If asked specific questions (e.g., "where is X when Y>Z"), query the actual data
7. If RAG context is provided, cite it as a source
8. IMPORTANT: If the user asks to generate a report, do NOT write the report in the chat. The system handles report generation. Just confirm you are ready to help or ask for clarification.
"""
        
        # Enhance user query with context
        enhanced_query = self._enhance_user_query(user_query, chart_context, data_analysis)
        
        # 4. Prepare messages with history
        messages = [{"role": "system", "content": system_message}]
        
        if chat_history:
            # Add last 6 messages for context (3 turns)
            for msg in chat_history[-6:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Add current query
        messages.append({"role": "user", "content": enhanced_query})
        
        try:
            resp = self.llm.chat.completions.create(
                model="llama3.2",
                messages=messages,
                temperature=0.4, # Reduced for speed/consistency
                max_tokens=500   # Reduced from 800
            )
            response_text = resp.choices[0].message.content.strip()
            
            # Convert markdown headers to bold for cleaner UI
            response_text = self._format_response_for_ui(response_text)
            
            return {"type": "chat", "content": response_text}
        except ImportError as e:
            logger.warning(f"LLM library import issue (using fallback): {str(e)[:100]}")
            return {"type": "chat", "content": self._fallback_response(user_query, session_context)}
        except AttributeError as e:
            logger.warning(f"LLM API mismatch (using fallback): {str(e)[:100]}")
            return {"type": "chat", "content": self._fallback_response(user_query, session_context)}
        except Exception as e:
            logger.error(f"LLM error: {str(e)[:100]}")
            return {"type": "chat", "content": self._fallback_response(user_query, session_context)}

    def _detect_intent(self, query: str, session_context: Dict[str, Any], chart_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Detect user intent for: plotting, filtering, previewing, extracting sensors, deleting files.
        """
        import re  # Import at function level to avoid scope issues
        
        query_lower = query.lower()
        
        # PRIORITY FALLBACK: Direct keyword-based detection for report generation
        # Check this FIRST to ensure we never miss a report request due to LLM
        is_report_request = any(word in query_lower for word in ['report', 'pdf', 'docx', 'document'])
        if is_report_request and "generate" in query_lower:
            logger.info("Fallback: Detected report generation request (Priority)")
            return {
                "action": "generate_report"
            }
        
        # Detect intent categories
        is_filter_request = any(kw in query_lower for kw in ['above', 'below', 'greater', 'less', 'positive', 'negative', 'only', 'where', 'filter'])
        is_plot_request = any(word in query_lower for word in ['plot', 'graph', 'chart', 'visualize'])
        # Expanded preview detection - catch "show data", "first 5", "top 10", "head", "preview"
        is_preview_request = (
            any(word in query_lower for word in ['rows', 'lines', 'preview', 'head', 'top', 'first']) or
            ('show' in query_lower and any(word in query_lower for word in ['data', 'column', 'columns', 'file']))
        )
        is_extract_request = any(word in query_lower for word in ['extract', 'harvest', 'get sensors', 'pull sensors', 'batch'])
        is_delete_request = any(word in query_lower for word in ['delete', 'remove', 'discard']) and any(word in query_lower for word in ['file', 'upload'])
        is_report_request = any(word in query_lower for word in ['report', 'pdf', 'docx', 'document'])
        
        if is_filter_request or is_plot_request or is_preview_request or is_extract_request or is_delete_request or is_report_request:
            try:
                files = list(session_context.get('files', {}).keys())
                file_list = ", ".join(files) if files else "No files uploaded"
                cols_info = ""
                if files:
                    first_file = files[0]
                    cols = session_context['files'][first_file].get('columns', [])
                    cols_info = f"Available columns: {', '.join(cols[:15])}"
                
                chart_info = ""
                if chart_context:
                    x_col = chart_context.get('x_column', '')
                    y_cols = chart_context.get('y_columns', [])
                    chart_info = f"Current Chart: X={x_col}, Y={', '.join(y_cols)}"

                prompt = f"""You are an intent parser for a thermal engineering tool. Extract structured intent from this user query.

**Query:** "{query}"
**Available Files:** {file_list}
{cols_info}
{chart_info}

**Return ONLY valid JSON for ONE of these actions:**

1. PLOT/FILTER chart (Prioritize this if user asks to "make", "plot", "show", "generate" a chart/graph):
{{"action": "plot_chart", "x_column": "Time", "y_columns": ["Col"], "chart_type": "line", "pandas_query": "Col > 0", "reason": "Visualizing temperature trends"}}

2. PREVIEW data (show first N rows):
{{"action": "preview_data", "count": 10}}

3. EXTRACT SENSORS (batch harvest from Excel sheets):
{{"action": "extract_sensors", "sensors": ["T_Inlet", "T_Outlet"], "source_file": "data.xlsx"}}

4. DELETE FILE:
{{"action": "delete_file", "filename": "file_to_delete.csv"}}

5. GENERAL CHAT (Only if NO specific action is requested):
{{"action": null}}

6. GENERATE REPORT:
{{"action": "generate_report"}}

JSON:"""
                
                resp = self.llm.chat.completions.create(
                    model="llama3.2",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=150 # Reduced from 250
                )
                content = resp.choices[0].message.content.strip()
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    
                    # If it's a plot request, try to get recommendations if columns are missing
                    if parsed.get("action") == "plot_chart" and not parsed.get("y_columns"):
                        # Fallback to some numeric columns if available
                        if files:
                            first_file = files[0]
                            num_cols = session_context['files'][first_file].get('numeric_columns', [])
                            if num_cols:
                                parsed["y_columns"] = num_cols[:3]
                                parsed["x_column"] = session_context['files'][first_file].get('time_column') or num_cols[0]
                    
                    logger.info(f"Detected intent: {parsed}")
                    return parsed
            except Exception as e:
                logger.warning(f"Intent detection failed: {e}")
        
        # FALLBACK: Direct keyword-based detection for preview_data
        # LLM is unreliable at returning preview_data action, so we handle it manually
        if is_preview_request:
            # Extract count from query (e.g., "first 5", "top 10")
            count_match = re.search(r'(\d+)', query)
            count = int(count_match.group(1)) if count_match else 5
            
            logger.info(f"Fallback: Detected preview request with count={count}")
            return {
                "action": "preview_data",
                "count": count
            }
            
        # FALLBACK: Force plot_chart if keywords are present but LLM returned None or null action
        if is_plot_request:
            logger.info("Fallback: Detected plot request (keywords present), forcing plot_chart")
            # Try to guess columns if possible, otherwise let the tool handle defaults
            return {
                "action": "plot_chart",
                "chart_type": "line" # Default
            }
        
        return None
    
    def _build_context_prompt(
        self,
        session_context: Dict[str, Any],
        chart_context: Optional[Dict[str, Any]],
        data_analysis: Optional[Dict[str, Any]],
        rag_context: str = ""
    ) -> str:
        """
        Build comprehensive context string for LLM
        """
        parts = []
        
        # RAG Context
        if rag_context:
            parts.append("**Relevant Documentation (RAG):**")
            parts.append(rag_context)
            parts.append("")
        
        # Session files
        files = session_context.get('files', {})
        if files:
            parts.append("**Available Data:**")
            for fname, fctx in list(files.items())[:3]:
                parts.append(f"  - {fname}: {fctx.get('row_count', 0)} rows")
                parts.append(f"    Columns: {', '.join(fctx.get('columns', [])[:8])}")
        
        # Latest chart
        if chart_context:
            parts.append("\n**Current Chart:**")
            parts.append(f"  - Type: {chart_context.get('chart_type', 'unknown')}")
            parts.append(f"  - X-axis: {chart_context.get('x_column', 'N/A')}")
            parts.append(f"  - Y-axis: {', '.join(chart_context.get('y_columns', []))}")
            
            # Add summary statistics
            if 'df_summary' in chart_context:
                summary = chart_context['df_summary']
                parts.append(f"  - Data points: {summary.get('total_points', 0)}")
        
        # Advanced analysis
        if data_analysis:
            parts.append("\n**Analysis Results:**")
            
            # Key statistics
            if 'basic_stats' in data_analysis:
                for col, stats in list(data_analysis['basic_stats'].items())[:2]:
                    parts.append(f"  - {col}: mean={stats.get('mean', 0):.2f}, range=[{stats.get('min', 0):.2f}, {stats.get('max', 0):.2f}]")
            
            # Anomalies
            if 'anomalies' in data_analysis:
                total_anomalies = sum(a.get('count', 0) for a in data_analysis['anomalies'].values())
                if total_anomalies > 0:
                    parts.append(f"  - ⚠️ {total_anomalies} anomalies detected across all columns")
            
            # Trends
            if 'trends' in data_analysis:
                for col, trend in list(data_analysis['trends'].items())[:2]:
                    if trend.get('is_significant'):
                        parts.append(f"  - {col}: {trend['direction']} trend (strength: {trend['strength']})")
            
            # Recommendations
            if 'recommendations' in data_analysis and data_analysis['recommendations']:
                parts.append("  - Key Recommendations:")
                for rec in data_analysis['recommendations'][:3]:
                    parts.append(f"    • {rec}")
        
        # Recent queries
        queries = session_context.get('queries', [])
        if queries:
            parts.append("\n**Recent Questions:**")
            for q in queries[-3:]:
                parts.append(f"  - {q['query'][:80]}")
        
        return "\n".join(parts)
    
    def _enhance_user_query(
        self,
        user_query: str,
        chart_context: Optional[Dict[str, Any]],
        data_analysis: Optional[Dict[str, Any]]
    ) -> str:
        """
        Enhance user query with relevant context
        """
        query_lower = user_query.lower()
        
        enhancements = [user_query]
        
        # If asking about specific values/conditions
        if any(word in query_lower for word in ['where', 'when', 'which', 'what points']):
            if chart_context and data_analysis:
                enhancements.append("\n**Note: To answer this question, analyze the following data:**")
                
                # Add basic stats for reference
                if 'basic_stats' in data_analysis:
                    enhancements.append(f"Statistics: {safe_json_dumps(data_analysis['basic_stats'], indent=2)}")
                
                # Add critical points
                if 'critical_points' in data_analysis:
                    enhancements.append("Critical Points:")
                    for col, cp in data_analysis['critical_points'].items():
                        enhancements.append(f"  {col}: max at index {cp.get('max_point', {}).get('index', 'N/A')}, min at index {cp.get('min_point', {}).get('index', 'N/A')}")
        
        # If asking about trends
        if any(word in query_lower for word in ['trend', 'increasing', 'decreasing', 'change']):
            if data_analysis and 'trends' in data_analysis:
                enhancements.append(f"\nTrend Analysis: {safe_json_dumps(data_analysis['trends'], indent=2)}")
        
        # If asking about anomalies/problems
        if any(word in query_lower for word in ['anomaly', 'problem', 'issue', 'wrong', 'spike']):
            if data_analysis and 'anomalies' in data_analysis:
                enhancements.append(f"\nAnomaly Report: {safe_json_dumps(data_analysis['anomalies'], indent=2)}")
        
        return "\n".join(enhancements)
    
    def _fallback_response(self, user_query: str, session_context: Dict[str, Any]) -> str:
        """
        Intelligent fallback when LLM is unavailable
        """
        query_lower = user_query.lower()
        
        # Pattern-based responses
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm ThermoSense AI, your thermal analysis expert. I can help you analyze test data, generate charts, detect anomalies, and provide engineering insights. What would you like to explore?"
        
        if 'help' in query_lower:
            return """**I can help you with:**
- Chart Generation: "Plot temperature vs time"
- Heatmap Analysis: "Generate correlation heatmap"
- Data Queries: "Where is temperature above 80?"
- Anomaly Detection: "Find anomalies in coolant temperature"
- Root Cause Analysis: "Why is temperature increasing?"
- Engineering Insights: Ask about any thermal phenomena

Just ask me a question about your data!"""
        
        files = session_context.get('files', {})
        charts = session_context.get('charts', {})
        
        return f"""I'm currently operating in basic mode (LLM unavailable).

**Your Session:**
- Files uploaded: {len(files)}
- Charts generated: {len(charts)}

For advanced analysis and insights, please ensure the LLM service is running.
Meanwhile, you can:
- Upload data files
- Generate charts using the UI
- View basic statistics"""
    
    def analyze_chart_deeply(
        self,
        chart_context: Dict[str, Any],
        data_analysis: Dict[str, Any]
    ) -> str:
        """
        Generate deep analysis of a chart with engineering insights
        """
        if not self.llm:
            return self._basic_chart_summary(chart_context, data_analysis)
        
        # Build analysis request
        prompt = f"""Analyze this thermal test chart as a senior thermal engineer.

**Chart Details:**
- Type: {chart_context.get('chart_type', 'unknown')}
- X-axis: {chart_context.get('x_column', 'N/A')}
- Y-axis: {', '.join(chart_context.get('y_columns', []))}
- Data points: {chart_context.get('df_summary', {}).get('total_points', 0)}

**Statistical Analysis:**
{safe_json_dumps(data_analysis.get('basic_stats', {}), indent=2)}

**Trends:**
{safe_json_dumps(data_analysis.get('trends', {}), indent=2)}

**Anomalies:**
{safe_json_dumps(data_analysis.get('anomalies', {}), indent=2)}

**Critical Points:**
{safe_json_dumps(data_analysis.get('critical_points', {}), indent=2)}

**Patterns:**
{safe_json_dumps(data_analysis.get('patterns', {}), indent=2)}

{self.domain_knowledge}

**Provide:**
1. **Physical Interpretation**: What do these patterns mean in thermal engineering terms?
# 2. **Root Cause Analysis**: Why is the data behaving this way? (DISABLED)
# 3. **Anomaly Assessment**: Are the detected anomalies concerning? (DISABLED)
# 4. **Performance Evaluation**: Is the thermal system performing as expected? (DISABLED)
# 5. **Recommendations**: Specific, actionable next steps. (DISABLED)

Keep it concise (100-150 words) and focus ONLY on physical interpretation of the visible patterns. Do NOT speculate on root causes or truck models.
"""
        
        try:
            resp = self.llm.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, # Reduced
                max_tokens=400   # Reduced from 600
            )
            return resp.choices[0].message.content.strip()
        except (ImportError, AttributeError) as e:
            logger.warning(f"LLM issue in deep analysis (using fallback): {str(e)[:100]}")
            return self._basic_chart_summary(chart_context, data_analysis)
        except Exception as e:
            logger.error(f"Deep analysis error: {str(e)[:100]}")
            return self._basic_chart_summary(chart_context, data_analysis)
    
    def _basic_chart_summary(self, chart_context: Dict[str, Any], data_analysis: Dict[str, Any]) -> str:
        """
        Basic chart summary without LLM
        """
        parts = []
        parts.append(f"**{chart_context.get('chart_type', 'Chart').title()} Analysis:**")
        parts.append(f"Showing {', '.join(chart_context.get('y_columns', []))} vs {chart_context.get('x_column', 'X')}")
        
        # Basic stats
        if 'basic_stats' in data_analysis:
            parts.append("\n**Key Statistics:**")
            for col, stats in data_analysis['basic_stats'].items():
                parts.append(f"- {col}: Range [{stats.get('min', 0):.2f}, {stats.get('max', 0):.2f}], Mean: {stats.get('mean', 0):.2f}")
        
        # Trends
        if 'trends' in data_analysis:
            parts.append("\n**Trends:**")
            for col, trend in data_analysis['trends'].items():
                if trend.get('is_significant'):
                    parts.append(f"- {col}: {trend['direction']} ({trend['strength']} strength)")
        
        # Anomalies
        if 'anomalies' in data_analysis:
            total = sum(a.get('count', 0) for a in data_analysis['anomalies'].values())
            if total > 0:
                parts.append(f"\n**Anomalies:** {total} detected - review recommended")
        
        # Recommendations (DISABLED)
        # if 'recommendations' in data_analysis:
        #     parts.append("\n**Recommendations:**")
        #     for rec in data_analysis['recommendations'][:5]:
        #         parts.append(f"- {rec}")
        
        return "\n".join(parts)


    def analyze_comparison(self, comparison_results: Dict[str, Any]) -> str:
        """
        Generate deep comparative analysis of multiple datasets
        """
        if not self.llm:
            return self._basic_comparison_summary(comparison_results)
            
        # Prepare context from results
        datasets = comparison_results.get('dataset_names', [])
        columns = comparison_results.get('columns_compared', [])
        stats = comparison_results.get('statistics', {})
        diffs = comparison_results.get('differences', {})
        rankings = comparison_results.get('rankings', {})
        
        prompt = f"""Perform a deep comparative analysis of these {len(datasets)} thermal test datasets.

**Datasets:** {', '.join(datasets)}
**Parameters Compared:** {', '.join(columns)}

**Statistical Overview:**
{safe_json_dumps(stats, indent=2)}

**Differences & Deltas:**
{safe_json_dumps(diffs, indent=2)}

**Rankings:**
{safe_json_dumps(rankings, indent=2)}

{self.domain_knowledge}

**Provide a professional engineering assessment covering:**
1.  **Performance Comparison**: Which dataset represents better performance? Why?
2.  **Trend Analysis**: How do the datasets differ in behavior (stability, response time, peaks)?
3.  **Root Cause Hypotheses**: What physical changes or test conditions could explain these differences?
4.  **Critical Deltas**: Highlight the most significant differences and their engineering implications.
5.  **Conclusion**: A definitive summary of the comparison.

Format as a structured technical narrative (approx 200-300 words). Use markdown formatting."""

        try:
            resp = self.llm.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Comparative analysis error: {str(e)[:100]}")
            return self._basic_comparison_summary(comparison_results)

    def _basic_comparison_summary(self, results: Dict[str, Any]) -> str:
        """Fallback summary for comparison"""
        summary = []
        summary.append(f"Comparative analysis of {len(results.get('dataset_names', []))} datasets.")
        
        for col, rankings in results.get('rankings', {}).items():
            if rankings:
                best = rankings[0]
                worst = rankings[-1]
                stats = results['statistics'][col]
                diff = results['differences'][col]
                max_delta = diff.get('max_delta', 0)
                
                summary.append(f"\n**{col}**:")
                summary.append(f"- Highest: {best} (Mean: {stats[best]['mean']:.2f})")
                summary.append(f"- Lowest: {worst} (Mean: {stats[worst]['mean']:.2f})")
                summary.append(f"- Variation: {max_delta:.2f}")
        
        return "\n".join(summary)

    def map_sensors_intelligently(self, requested_sensors: List[str], available_columns: List[str]) -> Dict[str, str]:
        """
        Use LLM to map requested sensor names to available columns in a file.
        Returns a dictionary: {requested_sensor: matched_column}
        """
        if not self.llm or not requested_sensors or not available_columns:
            return {}

        prompt = f"""You are an expert thermal data engineer. Map the requested sensor names to the most likely matching columns from the available list.

**Requested Sensors:**
{json.dumps(requested_sensors, indent=2)}

**Available Columns in File:**
{json.dumps(available_columns, indent=2)}

**Guidelines:**
1. Use domain knowledge (e.g., 'T_RamAir' might be 'Ram Air Temp' or 'T_Amb').
2. Handle common abbreviations (T = Temp, P = Power/Pressure, Spd = Speed).
3. Look for semantic matches even if the names are very different.
4. If no reasonable match exists for a sensor, do NOT map it.
5. Return ONLY a JSON object where keys are requested sensors and values are the matched columns.

**Example Output:**
{{
  "T_Inlet": "Coolant_Temp_In",
  "P_EM": "Total_Electric_Power"
}}

JSON:"""

        try:
            resp = self.llm.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            content = resp.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                mapping = json.loads(match.group(0))
                # Validate that mapped columns actually exist
                valid_mapping = {k: v for k, v in mapping.items() if v in available_columns}
                logger.info(f"Intelligent mapping result: {valid_mapping}")
                return valid_mapping
        except Exception as e:
            logger.error(f"Intelligent mapping failed: {e}")
        
        return {}
    
    def get_proactive_insight(self, analysis: Dict[str, Any], filename: str) -> str:
        """
        Generate a short, proactive insight based on data analysis.
        """
        if not self.llm:
            return f"I've analyzed **{filename}**. It has {analysis.get('basic_stats', {}).get('row_count', 'many')} rows. What would you like to see?"

        prompt = f"""You are ThermoSense AI. I've just uploaded a file: {filename}.
Here is the automated analysis summary:
{json.dumps(analysis, indent=2)}

**Task:**
Generate a very short (1-2 sentences), engaging "Instant Insight" to show the user you've already started working.
Focus on the most interesting finding (e.g., a specific trend, an anomaly, or a key statistic).
Be proactive and suggest a logical next step (e.g., "Should we plot the temperature spike at 300s?").

**Guidelines:**
1. Be professional but enthusiastic.
2. Use markdown for emphasis.
3. Keep it under 40 words.
4. Do NOT say "I've analyzed the file" - just give the insight.

Insight:"""

        try:
            resp = self.llm.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Proactive insight generation failed: {e}")
            return f"I've finished a quick scan of **{filename}**. I noticed some interesting patterns in the sensors. What should we visualize first?"

    def _format_response_for_ui(self, text: str) -> str:
        """
        Convert markdown headers to bold text for cleaner UI
        Converts:
        ### Header -> **Header**
        ## Header -> **Header**
        # Header -> **Header**
        """
        import re
        
        # Replace markdown headers with bold
        # Match lines starting with # (1-6 hashes) followed by space and text
        text = re.sub(r'^###\s+(.+)$', r'**\1**', text, flags=re.MULTILINE)
        text = re.sub(r'^##\s+(.+)$', r'**\1**', text, flags=re.MULTILINE)
        text = re.sub(r'^#\s+(.+)$', r'**\1**', text, flags=re.MULTILINE)
        
        return text

    def get_chart_recommendations(self, y_columns: List[str]) -> List[Dict[str, Any]]:
        """
        Suggest chart types based on the nature of the signals using LLM analysis.
        """
        if not self.llm:
            return [{"type": "line", "reason": "Default for time-series", "confidence": 0.5}]
            
        prompt = f"""
Analyze these data signal names and recommend the 2 best chart types for visualizing them.
Signals: {', '.join(y_columns)}

Return a JSON list of objects with:
- "type": (line, bar, scatter, area, histogram, heatmap, pie)
- "reason": (short explanation)
- "confidence": (0.0 to 1.0)

Example: [{"type": "line", "reason": "Temperature trends over time", "confidence": 0.9}]
"""
        try:
            resp = self.llm.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            response = resp.choices[0].message.content
            
            # Simple JSON extraction
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            logger.error(f"Error getting chart recommendations: {e}")
            
        return [{"type": "line", "reason": "Standard time-series visualization", "confidence": 0.5}]


# Singleton instance
_enhanced_orchestrator = None

def get_enhanced_orchestrator(llm_client=None) -> EnhancedLLMOrchestrator:
    global _enhanced_orchestrator
    if _enhanced_orchestrator is None:
        _enhanced_orchestrator = EnhancedLLMOrchestrator(llm_client)
    return _enhanced_orchestrator
