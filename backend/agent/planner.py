"""
Agent Planner - Converts user intent to structured execution plan
Uses Instructor for Pydantic-validated LLM outputs
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
import logging
import uuid

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    """Types of execution steps"""
    SUMMARIZE = "summarize"
    CHART = "chart"
    COMPARE = "compare"
    EXTRACT = "extract"
    ANALYZE = "analyze"
    REPORT = "report"
    QUERY = "query"


class ExecutionStep(BaseModel):
    """Single step in execution plan"""
    step_number: int = Field(..., description="Step sequence number")
    step_type: StepType = Field(..., description="Type of operation")
    tool_name: str = Field(..., description="Tool to execute")
    description: str = Field(..., description="What this step does")
    parameters: Dict[str, Any] = Field(..., description="Tool parameters")
    depends_on: List[int] = Field(default_factory=list, description="Step dependencies")
    expected_output: str = Field(..., description="Expected output description")
    
    class Config:
        schema_extra = {
            "example": {
                "step_number": 1,
                "step_type": "chart",
                "tool_name": "generate_chart",
                "description": "Generate temperature trend chart",
                "parameters": {
                    "filename": "thermal_data.csv",
                    "x_column": "Time",
                    "y_columns": ["Temp_1", "Temp_2"]
                },
                "depends_on": [],
                "expected_output": "Interactive chart showing temperature trends"
            }
        }


class ExecutionPlan(BaseModel):
    """Complete execution plan for user request"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique plan identifier")
    intent: str = Field(..., description="User's original intent")
    steps: List[ExecutionStep] = Field(..., description="Ordered execution steps")
    expected_artifacts: List[str] = Field(..., description="Expected outputs (charts, reports, etc)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in plan")
    reasoning: str = Field(..., description="Why this plan was chosen")
    
    class Config:
        schema_extra = {
            "example": {
                "intent": "Show me temperature trends and generate a report",
                "steps": [
                    {
                        "step_number": 1,
                        "step_type": "chart",
                        "tool_name": "generate_chart",
                        "description": "Generate temperature chart",
                        "parameters": {},
                        "depends_on": [],
                        "expected_output": "Temperature trend chart"
                    }
                ],
                "expected_artifacts": ["chart", "report"],
                "confidence": 0.9,
                "reasoning": "User wants visualization followed by report generation"
            }
        }


class AgentPlanner:
    """
    Converts user intent to structured execution plan
    Uses LLM with Instructor for reliable structured outputs
    """
    
    def __init__(self, llm_client=None, tool_registry=None):
        """
        Initialize planner
        
        Args:
            llm_client: OpenAI-compatible LLM client
            tool_registry: Tool registry for available tools
        """
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        
        # Try to import instructor
        try:
            import instructor
            if llm_client:
                self.instructor_client = instructor.from_openai(llm_client)
                logger.info("Instructor client initialized")
            else:
                self.instructor_client = None
                logger.warning("No LLM client provided, planner will use fallback")
        except ImportError:
            self.instructor_client = None
            logger.warning("Instructor not installed, using fallback planner")
    
    def create_plan(
        self,
        user_message: str,
        session_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """
        Create execution plan from user message
        
        Args:
            user_message: User's request
            session_context: Current session state (files, charts, etc)
        
        Returns:
            Structured execution plan
        """
        if self.instructor_client:
            return self._create_plan_with_llm(user_message, session_context)
        else:
            return self._create_plan_fallback(user_message, session_context)
    
    def _create_plan_with_llm(
        self,
        user_message: str,
        session_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """Create plan using LLM with Instructor"""
        try:
            # Get available tools
            available_tools = []
            if self.tool_registry:
                available_tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "category": tool.category
                    }
                    for tool in self.tool_registry.list_tools()
                ]
            
            # Build context
            context_str = self._build_context_string(session_context)
            
            # Create prompt
            system_prompt = f"""You are a thermal analysis planning agent. Convert user requests into structured execution plans.

Available tools:
{self._format_tools(available_tools)}

Current session context:
{context_str}

Create a step-by-step plan to fulfill the user's request. Each step should:
1. Use an available tool
2. Have clear parameters
3. Specify dependencies on previous steps
4. Describe expected output

Be specific and actionable. If the request is ambiguous, make reasonable assumptions."""

            # Call LLM with Instructor
            plan = self.instructor_client.chat.completions.create(
                model="llama3.2",  # Or whatever model is configured
                response_model=ExecutionPlan,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_retries=2
            )
            
            logger.info(f"Created plan with {len(plan.steps)} steps (confidence: {plan.confidence})")
            return plan
            
        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            return self._create_plan_fallback(user_message, session_context)
    
    def _create_plan_fallback(
        self,
        user_message: str,
        session_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """Fallback rule-based planning"""
        msg_lower = user_message.lower()
        steps = []
        expected_artifacts = []
        
        # Get actual file metadata for intelligent planning
        time_column = session_context.get("time_column")
        numeric_columns = session_context.get("numeric_columns", [])
        all_columns = session_context.get("columns", [])
        
        # Detect intent
        if any(word in msg_lower for word in ['chart', 'plot', 'graph', 'visualize', 'show']):
            # Auto-detect time column if not provided
            if not time_column and all_columns:
                # Try to detect from column names
                time_keywords = ['time', 'timestamp', 'tme', 'date', 'datetime', 'elapsed']
                for col in all_columns:
                    if any(keyword in col.lower() for keyword in time_keywords):
                        time_column = col
                        break
            
            # Default to first column if still no time column found
            if not time_column and all_columns:
                time_column = all_columns[0]
                logger.warning(f"No time column detected, using first column: {time_column}")
            
            # Chart generation with actual column metadata
            steps.append(ExecutionStep(
                step_number=1,
                step_type=StepType.CHART,
                tool_name="generate_chart",
                description="Generate chart from selected data",
                parameters={
                    "session_id": session_context.get("session_id", "default"),
                    "filename": session_context.get("files", [""])[0] if session_context.get("files") else "",
                    "x_column": time_column or "Time",  # Fallback to "Time" only if no columns available
                    "y_columns": numeric_columns[:5] if numeric_columns else [],  # Use first 5 numeric columns
                    "chart_type": "line"
                },
                depends_on=[],
                expected_output="Interactive chart visualization"
            ))
            expected_artifacts.append("chart")
        
        
        if any(word in msg_lower for word in ['compare', 'comparison', 'difference']):
            # Comparison
            steps.append(ExecutionStep(
                step_number=len(steps) + 1,
                step_type=StepType.COMPARE,
                tool_name="compare_runs",
                description="Compare multiple datasets",
                parameters={
                    "session_id": session_context.get("session_id", "default"),
                    "files": session_context.get("files", []),
                    "column": "Temperature"
                },
                depends_on=[],
                expected_output="Comparison chart and statistics"
            ))
            expected_artifacts.append("comparison")
        
        if any(word in msg_lower for word in ['report', 'document', 'summary']):
            # Report generation
            steps.append(ExecutionStep(
                step_number=len(steps) + 1,
                step_type=StepType.REPORT,
                tool_name="generate_report",
                description="Generate thermal analysis report",
                parameters={
                    "session_id": session_context.get("session_id", "default"),
                    "include_charts": True,
                    "include_analysis": True
                },
                depends_on=list(range(1, len(steps) + 1)),
                expected_output="Professional DOCX report"
            ))
            expected_artifacts.append("report")
        
        if any(word in msg_lower for word in ['extract', 'sensor', 'column']):
            # Sensor extraction
            steps.append(ExecutionStep(
                step_number=len(steps) + 1,
                step_type=StepType.EXTRACT,
                tool_name="extract_sensors",
                description="Extract specific sensors from files",
                parameters={
                    "session_id": session_context.get("session_id", "default"),
                    "files": session_context.get("files", []),
                    "sensors": [],
                    "output_filename": "extracted_data.csv"
                },
                depends_on=[],
                expected_output="CSV file with extracted sensors"
            ))
            expected_artifacts.append("extracted_data")
        
        # Default: summarize if no specific intent detected
        if not steps:
            steps.append(ExecutionStep(
                step_number=1,
                step_type=StepType.SUMMARIZE,
                tool_name="summarize_file",
                description="Summarize uploaded data file",
                parameters={
                    "session_id": session_context.get("session_id", "default"),
                    "filename": session_context.get("files", [""])[0] if session_context.get("files") else ""
                },
                depends_on=[],
                expected_output="Data summary with statistics"
            ))
            expected_artifacts.append("summary")
        
        return ExecutionPlan(
            intent=user_message,
            steps=steps,
            expected_artifacts=expected_artifacts,
            confidence=0.6,  # Lower confidence for rule-based
            reasoning="Rule-based plan created from keyword matching"
        )
    
    def _build_context_string(self, context: Dict[str, Any]) -> str:
        """Build context string for LLM"""
        parts = []
        
        if context.get("files"):
            parts.append(f"Uploaded files: {', '.join(context['files'])}")
        
        if context.get("charts"):
            parts.append(f"Generated charts: {len(context['charts'])}")
        
        if context.get("columns"):
            parts.append(f"Available columns: {', '.join(context['columns'][:10])}")
        
        return "\n".join(parts) if parts else "No context available"
    
    def _format_tools(self, tools: List[Dict]) -> str:
        """Format tools for prompt"""
        if not tools:
            return "No tools available"
        
        return "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in tools
        ])
