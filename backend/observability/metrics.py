import time
import threading
from typing import Dict, List, Any
from collections import defaultdict

class MetricsRegistry:
    """In-memory metrics registry"""
    def __init__(self):
        self._counters = defaultdict(int)
        self._histograms = defaultdict(list)
        self._lock = threading.Lock()

    def increment(self, name: str, labels: Dict[str, str] = None):
        key = self._get_key(name, labels)
        with self._lock:
            self._counters[key] += 1

    def observe(self, name: str, value: float, labels: Dict[str, str] = None):
        key = self._get_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)

    def _get_key(self, name: str, labels: Dict[str, str] = None) -> str:
        if not labels:
            return name
        label_str = ",".join([f'{k}="{v}"' for k, v in sorted(labels.items())])
        return f"{name}{{{label_str}}}"

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "histograms": {k: {
                    "count": len(v),
                    "sum": sum(v),
                    "avg": sum(v) / len(v) if v else 0,
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0
                } for k, v in self._histograms.items()}
            }

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        with self._lock:
            for name, value in self._counters.items():
                lines.append(f"{name} {value}")
            for name, stats in self._histograms.items():
                lines.append(f"{name}_count {stats['count']}")
                lines.append(f"{name}_sum {stats['sum']}")
        return "\n".join(lines)

# Global metrics registry
metrics = MetricsRegistry()

def time_execution(name: str, labels: Dict[str, str] = None):
    """Decorator or context manager to time execution"""
    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *args):
            duration = time.time() - self.start
            metrics.observe(name, duration, labels)
            
    return Timer()
