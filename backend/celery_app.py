"""
Celery application configuration for ThermoSense
Handles background tasks: chart generation, report generation, analytics, indexing
"""
import os
from celery import Celery
from kombu import Queue, Exchange

# Redis broker URL from environment or default
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
app = Celery(
    "thermosense",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "backend.tasks.chart_tasks",
        "backend.tasks.report_tasks",
        "backend.tasks.analytics_tasks",
        "backend.tasks.indexing_tasks",
    ]
)

# Celery configuration
app.conf.update(
    # Task routing
    task_routes={
        "backend.tasks.chart_tasks.*": {"queue": "charts"},
        "backend.tasks.report_tasks.*": {"queue": "reports"},
        "backend.tasks.analytics_tasks.*": {"queue": "analytics"},
        "backend.tasks.indexing_tasks.*": {"queue": "indexing"},
    },
    
    # Define queues
    task_queues=(
        Queue("default", Exchange("default"), routing_key="default"),
        Queue("charts", Exchange("charts"), routing_key="charts"),
        Queue("reports", Exchange("reports"), routing_key="reports"),
        Queue("analytics", Exchange("analytics"), routing_key="analytics"),
        Queue("indexing", Exchange("indexing"), routing_key="indexing"),
    ),
    
    # Task execution settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Retry settings
    task_acks_late=True,  # Acknowledge task after completion
    task_reject_on_worker_lost=True,
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_backend_transport_options={
        "master_name": "mymaster",
    },
    
    # Worker settings
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks to prevent memory leaks
    
    # Task time limits
    task_soft_time_limit=300,  # 5 minutes soft limit
    task_time_limit=600,  # 10 minutes hard limit
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Optional: Celery Beat schedule for periodic tasks
app.conf.beat_schedule = {
    # Example: Clean up old task results every hour
    "cleanup-old-results": {
        "task": "backend.tasks.maintenance_tasks.cleanup_old_results",
        "schedule": 3600.0,  # Every hour
    },
}

if __name__ == "__main__":
    app.start()
