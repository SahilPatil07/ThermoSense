"""
Agent execution endpoint
Provides agentic workflow: User intent → Plan → Execute → Verify → Response
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/agent", tags=["agent"])


class AgentExecuteRequest(BaseModel):
    """Request for agent execution"""
    message: str
    session_id: str = "default"
    auto_approve: bool = False  # Auto-approve plan without user confirmation


class AgentExecuteResponse(BaseModel):
    """Response from agent execution"""
    success: bool
    plan: Optional[Dict[str, Any]] = None
    execution_result: Optional[Dict[str, Any]] = None
    response: str
    artifacts: list = []


# Global agent components (will be set by main.py)
agent_planner = None
agent_router = None
agent_registry = None


def set_agent_components(planner, router, registry):
    """Set agent components (called from main.py)"""
    global agent_planner, agent_router, agent_registry
    agent_planner = planner
    agent_router = router
    agent_registry = registry


@router.post("/execute", response_model=AgentExecuteResponse)
async def execute_agent_workflow(req: AgentExecuteRequest):
    """
    Execute complete agentic workflow
    """
    from backend.observability.tracing import trace_span
    from backend.observability.logger import set_context
    
    set_context(session_id=req.session_id)
    
    with trace_span("agent_workflow", {"message": req.message}):
        try:
            print(f"DEBUG: execute_agent_workflow. Planner: {agent_planner is not None}, Router: {agent_router is not None}")
            if not all([agent_planner, agent_router, agent_registry]):
                return JSONResponse({
                    "success": False,
                    "response": "Agent platform not initialized",
                    "plan": None,
                    "execution_result": None,
                    "artifacts": []
                }, status_code=500)

            
            logger.info(f"Agent execution request: {req.message}")
            
            # Step 1: Create execution plan
            from backend.tools.storage import SessionStorage
            storage = SessionStorage()
            
            # Build session context with file metadata
            uploaded_files = storage.get_uploads(req.session_id)
            session_context = {
                "session_id": req.session_id,
                "files": uploaded_files,
                "charts": [],  # TODO: Get from storage
            }
            
            # Get file metadata if files are available
            if uploaded_files:
                from backend.tools.sensor_harvester import SensorHarvester
                harvester = SensorHarvester()
                
                # Get metadata from first file (primary file)
                first_file = uploaded_files[0]
                file_path = storage.find_file(first_file)
                
                if file_path and file_path.exists():
                    try:
                        columns, time_column, numeric_columns = harvester.get_file_columns(str(file_path))
                        session_context["columns"] = columns
                        session_context["time_column"] = time_column
                        session_context["numeric_columns"] = numeric_columns
                        logger.info(f"Loaded file metadata: {len(columns)} columns, time_column={time_column}")
                    except Exception as e:
                        logger.warning(f"Could not load file metadata: {e}")
                        session_context["columns"] = []
                        session_context["time_column"] = None
                        session_context["numeric_columns"] = []
            
            
            logger.info("Creating execution plan...")
            plan = agent_planner.create_plan(
                user_message=req.message,
                session_context=session_context
            )
            
            logger.info(f"Plan created with {len(plan.steps)} steps (confidence: {plan.confidence})")
            
            # Step 2: Execute plan
            if req.auto_approve or plan.confidence > 0.7:
                logger.info("Executing plan...")
                execution_result = agent_router.execute_plan(
                    plan=plan,
                    session_id=req.session_id
                )
                
                # Extract artifacts
                artifacts = []
                for output in execution_result.outputs:
                    if "chart_url" in output:
                        artifacts.append({
                            "type": "chart",
                            "url": output["chart_url"],
                            "id": output.get("chart_id")
                        })
                    elif "report_path" in output:
                        artifacts.append({
                            "type": "report",
                            "path": output["report_path"]
                        })
                
                return AgentExecuteResponse(
                    success=execution_result.success,
                    plan=plan.dict(),
                    execution_result=execution_result.dict(),
                    response=execution_result.final_response,
                    artifacts=artifacts
                )
            else:
                # Return plan for user approval
                return AgentExecuteResponse(
                    success=True,
                    plan=plan.dict(),
                    execution_result=None,
                    response=f"Plan created with {len(plan.steps)} steps. Confidence: {plan.confidence:.0%}. Please review and approve.",
                    artifacts=[]
                )
        
        except Exception as e:
            logger.exception("Agent execution failed")
            return JSONResponse({
                "success": False,
                "response": f"Agent execution failed: {str(e)}",
                "plan": None,
                "execution_result": None,
                "artifacts": []
            }, status_code=500)



@router.get("/plan/{session_id}")
async def get_latest_plan(session_id: str):
    """Get latest execution plan for session"""
    # TODO: Store plans in database
    return JSONResponse({
        "success": False,
        "message": "Plan storage not yet implemented"
    })


@router.get("/tools")
async def list_available_tools():
    """List all available tools"""
    if not agent_registry:
        return JSONResponse({
            "success": False,
            "tools": []
        })
    
    tools = agent_registry.list_tools()
    
    return JSONResponse({
        "success": True,
        "tools": [
            {
                "name": tool.name,
                "category": tool.category,
                "description": tool.description,
                "timeout": tool.timeout,
                "requires_llm": tool.requires_llm
            }
            for tool in tools
        ]
    })
