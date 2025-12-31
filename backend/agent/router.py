"""
Tool Router - Selects appropriate tools based on plan
Routes execution to registered tools
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from backend.agent.planner import ExecutionPlan, ExecutionStep
from backend.agent.verifier import OutputVerifier, ValidationResult
from backend.agent.tool_registry import ToolRegistry
import logging

logger = logging.getLogger(__name__)


class ExecutionResult(BaseModel):
    """Result of executing a plan"""
    success: bool
    steps_completed: int
    steps_failed: int
    outputs: List[Dict[str, Any]]
    errors: List[str]
    final_response: str
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "steps_completed": 3,
                "steps_failed": 0,
                "outputs": [{"chart_id": "abc123"}],
                "errors": [],
                "final_response": "Generated 3 charts successfully"
            }
        }


class ToolRouter:
    """
    Routes execution to appropriate tools
    Manages dependencies and error handling
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        verifier: Optional[OutputVerifier] = None
    ):
        """
        Initialize router
        
        Args:
            tool_registry: Registry of available tools
            verifier: Output verifier (optional)
        """
        self.tool_registry = tool_registry
        self.verifier = verifier or OutputVerifier()
    
    def execute_plan(
        self,
        plan: ExecutionPlan,
        session_id: str
    ) -> ExecutionResult:
        """
        Execute a complete plan
        
        Args:
            plan: Execution plan to run
            session_id: Session identifier
        
        Returns:
            Execution result with outputs
        """
        from backend.observability.tracing import trace_span
        from backend.observability.metrics import metrics, time_execution
        from backend.observability.logger import set_context
        
        set_context(session_id=session_id)
        
        with trace_span("execute_plan", metadata={"plan_id": str(plan.id), "steps": len(plan.steps)}):
            logger.info(f"Executing plan with {len(plan.steps)} steps")
            
            outputs = []
            errors = []
            steps_completed = 0
            steps_failed = 0
            
            # Store step outputs for dependencies
            step_results: Dict[int, Dict[str, Any]] = {}
            
            # Execute steps in order
            for step in sorted(plan.steps, key=lambda s: s.step_number):
                try:
                    # Check dependencies
                    if not self._check_dependencies(step, step_results):
                        error_msg = f"Step {step.step_number} dependencies not met"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        steps_failed += 1
                        continue
                    
                    # Inject session_id if not present
                    if "session_id" not in step.parameters:
                        step.parameters["session_id"] = session_id
                    
                    # Execute step
                    with trace_span(f"execute_step:{step.tool_name}", metadata={"step": step.step_number}):
                        logger.info(f"Executing step {step.step_number}: {step.tool_name}")
                        with time_execution("tool_execution_seconds", labels={"tool": step.tool_name}):
                            result = self._execute_step(step, step_results)
                    
                    # Verify output
                    validation = self.verifier.verify_output(
                        tool_name=step.tool_name,
                        output=result,
                        session_id=session_id
                    )
                    
                    if validation.valid:
                        step_results[step.step_number] = result
                        outputs.append(result)
                        steps_completed += 1
                        logger.info(f"Step {step.step_number} completed successfully")
                        metrics.increment("agent_step_success_total", labels={"tool": step.tool_name})
                    else:
                        error_msg = f"Step {step.step_number} validation failed: {', '.join(validation.errors)}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        steps_failed += 1
                        metrics.increment("agent_step_failure_total", labels={"tool": step.tool_name, "reason": "validation"})
                    
                except Exception as e:
                    error_msg = f"Step {step.step_number} execution failed: {str(e)}"
                    logger.exception(error_msg)
                    errors.append(error_msg)
                    steps_failed += 1
                    metrics.increment("agent_step_failure_total", labels={"tool": step.tool_name, "reason": "exception"})
            
            # Generate final response
            final_response = self._generate_final_response(
                plan, steps_completed, steps_failed, outputs, errors
            )
            
            success = steps_failed == 0 and steps_completed > 0
            
            if success:
                metrics.increment("agent_plan_success_total")
            else:
                metrics.increment("agent_plan_failure_total")
                
            return ExecutionResult(
                success=success,
                steps_completed=steps_completed,
                steps_failed=steps_failed,
                outputs=outputs,
                errors=errors,
                final_response=final_response
            )
    
    def _execute_step(
        self,
        step: ExecutionStep,
        previous_results: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute a single step
        
        Args:
            step: Step to execute
            previous_results: Results from previous steps
        
        Returns:
            Step output
        """
        # Resolve parameters from dependencies
        resolved_params = self._resolve_parameters(
            step.parameters,
            step.depends_on,
            previous_results
        )
        
        # Execute tool
        result = self.tool_registry.execute(
            name=step.tool_name,
            params=resolved_params,
            validate_input=True,
            validate_output=True
        )
        
        return result
    
    def _check_dependencies(
        self,
        step: ExecutionStep,
        step_results: Dict[int, Dict[str, Any]]
    ) -> bool:
        """Check if step dependencies are satisfied"""
        for dep_step in step.depends_on:
            if dep_step not in step_results:
                return False
        return True
    
    def _resolve_parameters(
        self,
        params: Dict[str, Any],
        dependencies: List[int],
        previous_results: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Resolve parameters from previous step outputs
        
        Args:
            params: Step parameters
            dependencies: Step dependencies
            previous_results: Previous step results
        
        Returns:
            Resolved parameters
        """
        resolved = params.copy()
        
        # Look for parameter references like "${step_1.output_file}"
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("${"):
                # Extract reference
                ref = value[2:-1]  # Remove ${ and }
                parts = ref.split(".")
                
                if len(parts) == 2:
                    step_ref, field = parts
                    step_num = int(step_ref.split("_")[1])
                    
                    if step_num in previous_results:
                        resolved[key] = previous_results[step_num].get(field, value)
        
        return resolved
    
    def _generate_final_response(
        self,
        plan: ExecutionPlan,
        steps_completed: int,
        steps_failed: int,
        outputs: List[Dict[str, Any]],
        errors: List[str]
    ) -> str:
        """Generate human-readable final response"""
        if steps_failed == 0 and steps_completed > 0:
            # Success
            artifacts = []
            for output in outputs:
                if "chart_url" in output:
                    artifacts.append("chart")
                elif "report_path" in output:
                    artifacts.append("report")
                elif "output_file" in output:
                    artifacts.append("extracted data")
            
            if artifacts:
                return f"✅ Successfully completed {steps_completed} steps. Generated: {', '.join(artifacts)}"
            else:
                return f"✅ Successfully completed {steps_completed} steps"
        
        elif steps_completed > 0 and steps_failed > 0:
            # Partial success
            return f"⚠️ Completed {steps_completed} steps, but {steps_failed} failed. Errors: {'; '.join(errors[:2])}"
        
        else:
            # Complete failure
            return f"❌ Execution failed. {'; '.join(errors[:2])}"


# Import BaseModel for ExecutionResult
from pydantic import BaseModel
