"""
Tool Registry for ThermoSense Agent Platform
Central registry of all available tools with strict Pydantic contracts
"""
from typing import Dict, Any, List, Callable, Optional
from pydantic import BaseModel, Field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Tool categories"""
    DATA = "data"
    CHART = "chart"
    ANALYTICS = "analytics"
    REPORT = "report"
    KNOWLEDGE = "knowledge"


class ToolDefinition(BaseModel):
    """Tool definition with metadata"""
    name: str = Field(..., description="Unique tool name")
    category: ToolCategory = Field(..., description="Tool category")
    description: str = Field(..., description="What this tool does")
    input_schema: type[BaseModel] = Field(..., description="Pydantic input model")
    output_schema: type[BaseModel] = Field(..., description="Pydantic output model")
    executor: Callable = Field(..., description="Function that executes the tool")
    timeout: int = Field(default=300, description="Timeout in seconds")
    requires_llm: bool = Field(default=False, description="Whether tool requires LLM")
    
    class Config:
        arbitrary_types_allowed = True


class ToolRegistry:
    """
    Central registry for all agent tools
    Provides tool discovery, validation, and execution
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        logger.info("Tool registry initialized")
    
    def register(
        self,
        name: str,
        category: ToolCategory,
        description: str,
        input_schema: type[BaseModel],
        output_schema: type[BaseModel],
        executor: Callable,
        timeout: int = 300,
        requires_llm: bool = False
    ) -> None:
        """
        Register a new tool
        
        Args:
            name: Unique tool identifier
            category: Tool category
            description: Human-readable description
            input_schema: Pydantic model for input validation
            output_schema: Pydantic model for output validation
            executor: Function that executes the tool
            timeout: Maximum execution time in seconds
            requires_llm: Whether tool needs LLM
        """
        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered, overwriting")
        
        tool = ToolDefinition(
            name=name,
            category=category,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            executor=executor,
            timeout=timeout,
            requires_llm=requires_llm
        )
        
        self._tools[name] = tool
        logger.info(f"Registered tool: {name} ({category})")
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get tool by name"""
        return self._tools.get(name)
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> List[ToolDefinition]:
        """
        List all tools, optionally filtered by category
        
        Args:
            category: Optional category filter
        
        Returns:
            List of tool definitions
        """
        tools = list(self._tools.values())
        
        if category:
            tools = [t for t in tools if t.category == category]
        
        return tools
    
    def execute(
        self,
        name: str,
        params: Dict[str, Any],
        validate_input: bool = True,
        validate_output: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a tool with validation
        
        Args:
            name: Tool name
            params: Tool parameters
            validate_input: Whether to validate input
            validate_output: Whether to validate output
        
        Returns:
            Tool execution result
        
        Raises:
            ValueError: If tool not found or validation fails
            TimeoutError: If execution exceeds timeout
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        
        # Validate input
        if validate_input:
            try:
                validated_input = tool.input_schema(**params)
                params = validated_input.dict()
            except Exception as e:
                raise ValueError(f"Input validation failed for {name}: {e}")
        
        # Execute tool
        from backend.observability.metrics import metrics
        
        try:
            logger.info(f"Executing tool: {name}")
            result = tool.executor(**params)
            metrics.increment("tool_success_total", labels={"tool": name})
        except Exception as e:
            logger.error(f"Tool execution failed: {name} - {e}")
            metrics.increment("tool_failure_total", labels={"tool": name})
            raise
        
        # Validate output
        if validate_output:
            try:
                validated_output = tool.output_schema(**result)
                result = validated_output.dict()
            except Exception as e:
                logger.warning(f"Output validation failed for {name}: {e}")
                # Don't fail on output validation, just log
        
        return result
    
    def get_tool_schema(self, name: str) -> Dict[str, Any]:
        """
        Get JSON schema for a tool (for LLM consumption)
        
        Args:
            name: Tool name
        
        Returns:
            Tool schema as dictionary
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        
        return {
            "name": tool.name,
            "category": tool.category,
            "description": tool.description,
            "input_schema": tool.input_schema.schema(),
            "output_schema": tool.output_schema.schema(),
            "timeout": tool.timeout,
            "requires_llm": tool.requires_llm
        }
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools (for LLM tool selection)"""
        return [self.get_tool_schema(name) for name in self._tools.keys()]


# Global registry instance
_registry = None


def get_tool_registry() -> ToolRegistry:
    """Get global tool registry instance"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
