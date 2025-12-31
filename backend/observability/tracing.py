import time
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading

@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List['Span'] = field(default_factory=list)

    def finish(self):
        self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration": (self.end_time - self.start_time) if self.end_time else None,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children]
        }

class Tracer:
    """Simple internal tracer"""
    def __init__(self):
        self._active_spans = threading.local()
        self._completed_traces = [] # Keep last N traces in memory
        self._max_traces = 100
        self._lock = threading.Lock()

    def _get_stack(self):
        if not hasattr(self._active_spans, 'stack'):
            self._active_spans.stack = []
        return self._active_spans.stack

    def start_span(self, name: str, trace_id: str = None, metadata: Dict[str, Any] = None) -> Span:
        stack = self._get_stack()
        parent = stack[-1] if stack else None
        
        if parent:
            trace_id = parent.trace_id
            parent_id = parent.span_id
        else:
            trace_id = trace_id or str(uuid.uuid4())
            parent_id = None
            
        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=str(uuid.uuid4())[:8],
            parent_id=parent_id,
            metadata=metadata or {}
        )
        
        if parent:
            parent.children.append(span)
            
        stack.append(span)
        return span

    def end_span(self):
        stack = self._get_stack()
        if not stack:
            return
            
        span = stack.pop()
        span.finish()
        
        # If this was a root span, save the trace
        if not span.parent_id:
            with self._lock:
                self._completed_traces.append(span.to_dict())
                if len(self._completed_traces) > self._max_traces:
                    self._completed_traces.pop(0)


    def get_traces(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._completed_traces)

# Global tracer instance
tracer = Tracer()

class trace_span:
    """Context manager for tracing"""
    def __init__(self, name: str, metadata: Dict[str, Any] = None):
        self.name = name
        self.metadata = metadata
        
    def __enter__(self):
        return tracer.start_span(self.name, metadata=self.metadata)
        
    def __exit__(self, *args):
        tracer.end_span()
