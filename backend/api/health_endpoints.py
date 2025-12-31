from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
import psutil
import os
import time
from typing import Dict, Any
import logging

from backend.observability.metrics import metrics
from backend.observability.tracing import tracer

router = APIRouter(prefix="/api/observability", tags=["observability"])
health_router = APIRouter(prefix="/api/health", tags=["health"])

logger = logging.getLogger(__name__)



@health_router.get("")
async def health_check():
    """Basic health check"""
    return {"status": "UP", "timestamp": time.time()}

@health_router.get("/details")
async def health_details():
    """Detailed health check with resource usage"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Check if celery/redis is reachable (optional, based on your setup)
    # This is a placeholder for actual service checks
    
    return {
        "status": "UP",
        "timestamp": time.time(),
        "resources": {
            "cpu_percent": cpu_percent,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "free": disk.free,
                "percent": disk.percent
            }
        },
        "pid": os.getpid()
    }

@router.get("/metrics")
async def get_metrics(format: str = "json"):
    """Get system metrics"""
    if format == "prometheus":
        from fastapi.responses import Response
        return Response(content=metrics.to_prometheus(), media_type="text/plain")
    
    return metrics.get_metrics()

@router.get("/traces")
async def get_traces():
    """Get recent traces"""
    return {"traces": tracer.get_traces()}
