"""
Celery task status endpoints for FastAPI
Provides real-time task monitoring and progress tracking
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
from celery.result import AsyncResult
from backend.celery_app import app as celery_app
import logging

logger = logging.getLogger(__name__)

# Create router for task endpoints
router = APIRouter(prefix="/api/tasks", tags=["tasks"])


class TaskStatusResponse(BaseModel):
    """Task status response model"""
    task_id: str
    status: str  # PENDING, STARTED, PROGRESS, SUCCESS, FAILURE
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get status of a Celery task
    
    Args:
        task_id: Celery task ID
    
    Returns:
        Task status with result or progress information
    """
    try:
        task_result = AsyncResult(task_id, app=celery_app)
        
        response = {
            "task_id": task_id,
            "status": task_result.state,
            "result": None,
            "error": None,
            "progress": None
        }
        
        if task_result.state == "PENDING":
            response["result"] = "Task is waiting to be executed"
        
        elif task_result.state == "STARTED":
            response["result"] = "Task has started"
        
        elif task_result.state == "PROGRESS":
            # Get progress information from task meta
            response["progress"] = task_result.info
            response["result"] = task_result.info.get("status", "Processing...")
        
        elif task_result.state == "SUCCESS":
            response["result"] = task_result.result
        
        elif task_result.state == "FAILURE":
            response["error"] = str(task_result.info)
            response["result"] = "Task failed"
        
        else:
            response["result"] = str(task_result.info)
        
        return JSONResponse(response)
    
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a running Celery task
    
    Args:
        task_id: Celery task ID
    
    Returns:
        Cancellation status
    """
    try:
        task_result = AsyncResult(task_id, app=celery_app)
        task_result.revoke(terminate=True)
        
        return JSONResponse({
            "success": True,
            "task_id": task_id,
            "message": "Task cancellation requested"
        })
    
    except Exception as e:
        logger.error(f"Error cancelling task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active")
async def get_active_tasks():
    """
    Get list of active tasks
    
    Returns:
        List of currently running tasks
    """
    try:
        # Get active tasks from Celery
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()
        
        if not active_tasks:
            return JSONResponse({
                "active_tasks": [],
                "count": 0
            })
        
        # Flatten tasks from all workers
        all_tasks = []
        for worker, tasks in active_tasks.items():
            for task in tasks:
                all_tasks.append({
                    "task_id": task.get("id"),
                    "name": task.get("name"),
                    "worker": worker,
                    "args": task.get("args"),
                    "kwargs": task.get("kwargs")
                })
        
        return JSONResponse({
            "active_tasks": all_tasks,
            "count": len(all_tasks)
        })
    
    except Exception as e:
        logger.error(f"Error getting active tasks: {e}")
        # Return empty list if Celery is not running
        return JSONResponse({
            "active_tasks": [],
            "count": 0,
            "error": "Celery worker not available"
        })


@router.get("/stats")
async def get_task_stats():
    """
    Get Celery task statistics
    
    Returns:
        Statistics about task queues and workers
    """
    try:
        inspect = celery_app.control.inspect()
        
        stats = {
            "active": inspect.active() or {},
            "scheduled": inspect.scheduled() or {},
            "reserved": inspect.reserved() or {},
            "stats": inspect.stats() or {}
        }
        
        # Calculate totals
        total_active = sum(len(tasks) for tasks in stats["active"].values())
        total_scheduled = sum(len(tasks) for tasks in stats["scheduled"].values())
        total_reserved = sum(len(tasks) for tasks in stats["reserved"].values())
        
        return JSONResponse({
            "workers": list(stats["active"].keys()),
            "total_active": total_active,
            "total_scheduled": total_scheduled,
            "total_reserved": total_reserved,
            "details": stats
        })
    
    except Exception as e:
        logger.error(f"Error getting task stats: {e}")
        return JSONResponse({
            "workers": [],
            "total_active": 0,
            "total_scheduled": 0,
            "total_reserved": 0,
            "error": "Celery worker not available"
        })
