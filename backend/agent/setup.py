"""
Agent Platform Setup
Initializes all agent components and registers tools
"""
from backend.agent.tool_registry import ToolRegistry, ToolCategory
from backend.agent.planner import AgentPlanner
from backend.agent.router import ToolRouter
from backend.agent.verifier import OutputVerifier
from backend.agent.tool_implementations import ToolImplementations
from backend.agent.tool_contracts import (
    SummarizeFileInput, SummarizeFileOutput,
    GenerateChartInput, GenerateChartOutput,
    CompareRunsInput, CompareRunsOutput,
    ExtractSensorsInput, ExtractSensorsOutput,
    GenerateReportInput, GenerateReportOutput
)
import logging

logger = logging.getLogger(__name__)


def initialize_agent_platform(
    storage,
    sensor_harvester,
    plotly_generator,
    comparative_analyzer,
    report_generator=None,
    llm_client=None
):
    """
    Initialize the complete agent platform
    
    Args:
        storage: SessionStorage instance
        sensor_harvester: SensorHarvester instance
        plotly_generator: PlotlyChartGenerator instance
        comparative_analyzer: ComparativeAnalyzer instance
        report_generator: ThermalReportGenerator instance (optional)
        llm_client: LLM client for planner (optional)
    
    Returns:
        Tuple of (registry, planner, router, verifier, implementations)
    """
    logger.info("Initializing agent platform...")
    
    # Create core components
    registry = ToolRegistry()
    verifier = OutputVerifier(workspace_dir="workspace")
    planner = AgentPlanner(llm_client=llm_client, tool_registry=registry)
    router = ToolRouter(tool_registry=registry, verifier=verifier)
    
    # Create tool implementations
    implementations = ToolImplementations(
        storage=storage,
        sensor_harvester=sensor_harvester,
        plotly_generator=plotly_generator,
        comparative_analyzer=comparative_analyzer,
        report_generator=report_generator
    )
    
    # Register Phase 2 tools
    logger.info("Registering Phase 2 tools...")
    
    registry.register(
        name="summarize_file",
        category=ToolCategory.DATA,
        description="Summarize data file with column information and statistics",
        input_schema=SummarizeFileInput,
        output_schema=SummarizeFileOutput,
        executor=implementations.summarize_file,
        timeout=60,
        requires_llm=False
    )
    
    registry.register(
        name="generate_chart",
        category=ToolCategory.CHART,
        description="Generate interactive chart from data file",
        input_schema=GenerateChartInput,
        output_schema=GenerateChartOutput,
        executor=implementations.generate_chart,
        timeout=120,
        requires_llm=False
    )
    
    registry.register(
        name="compare_runs",
        category=ToolCategory.CHART,
        description="Compare multiple data files and generate comparison chart",
        input_schema=CompareRunsInput,
        output_schema=CompareRunsOutput,
        executor=implementations.compare_runs,
        timeout=180,
        requires_llm=False
    )
    
    registry.register(
        name="extract_sensors",
        category=ToolCategory.DATA,
        description="Extract specific sensors from multiple files",
        input_schema=ExtractSensorsInput,
        output_schema=ExtractSensorsOutput,
        executor=implementations.extract_sensors,
        timeout=300,
        requires_llm=False
    )
    
    if report_generator:
        registry.register(
            name="generate_report",
            category=ToolCategory.REPORT,
            description="Generate professional thermal analysis report",
            input_schema=GenerateReportInput,
            output_schema=GenerateReportOutput,
            executor=implementations.generate_report,
            timeout=300,
            requires_llm=True
        )
    
    # Register Phase 3 analytics tools
    try:
        from backend.analytics.tool_implementations import AnalyticsToolImplementations
        from backend.analytics.contracts import (
            DetectChangePointsInput, DetectChangePointsOutput,
            AnalyzeTimeSeriesInput, AnalyzeTimeSeriesOutput,
            DetectAnomaliesInput, DetectAnomaliesOutput,
            StatisticalAnalysisInput, StatisticalAnalysisOutput
        )
        
        logger.info("Registering Phase 3 analytics tools...")
        
        analytics_impl = AnalyticsToolImplementations(storage=storage)
        
        registry.register(
            name="detect_change_points",
            category=ToolCategory.ANALYTICS,
            description="Detect change points in time series data using ruptures",
            input_schema=DetectChangePointsInput,
            output_schema=DetectChangePointsOutput,
            executor=analytics_impl.detect_change_points,
            timeout=120,
            requires_llm=False
        )
        
        registry.register(
            name="analyze_time_series",
            category=ToolCategory.ANALYTICS,
            description="Analyze time series with decomposition, forecasting, or ACF",
            input_schema=AnalyzeTimeSeriesInput,
            output_schema=AnalyzeTimeSeriesOutput,
            executor=analytics_impl.analyze_time_series,
            timeout=120,
            requires_llm=False
        )
        
        registry.register(
            name="detect_anomalies",
            category=ToolCategory.ANALYTICS,
            description="Detect anomalies using IsolationForest, LOF, or One-Class SVM",
            input_schema=DetectAnomaliesInput,
            output_schema=DetectAnomaliesOutput,
            executor=analytics_impl.detect_anomalies,
            timeout=120,
            requires_llm=False
        )
        
        registry.register(
            name="statistical_analysis",
            category=ToolCategory.ANALYTICS,
            description="Perform statistical analysis including descriptive stats and correlations",
            input_schema=StatisticalAnalysisInput,
            output_schema=StatisticalAnalysisOutput,
            executor=analytics_impl.statistical_analysis,
            timeout=60,
            requires_llm=False
        )
        
        logger.info("✅ Registered 4 analytics tools")
        
    except ImportError as e:
        logger.warning(f"Analytics tools not available: {e}")
    
    logger.info(f"✅ Registered {len(registry.list_tools())} total tools")
    
    return registry, planner, router, verifier, implementations
