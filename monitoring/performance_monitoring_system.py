"""
Performance Monitoring System - Phase 3
APM tools pro deep insights a monitoring
- Custom dashboards pro research metrics
- Alert automation na základě research KPIs
- Real-time performance tracking
- Resource utilization monitoring
"""

import asyncio
import time
import logging
import json
import psutil
import tracemalloc
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import weakref

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Typy metrik pro monitoring"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertSeverity(Enum):
    """Úrovně alertů"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Základní metrika"""
    name: str
    metric_type: MetricType
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""

@dataclass
class Alert:
    """Alert definice"""
    id: str
    name: str
    condition: str
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    last_triggered: Optional[float] = None
    trigger_count: int = 0

@dataclass
class SystemMetrics:
    """Systémové metriky"""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_usage_percent: float
    network_io_bytes: Dict[str, int]
    process_count: int
    load_average: float
    timestamp: float

class MetricsCollector:
    """Sběr systémových a aplikačních metrik"""

    def __init__(self):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.custom_metrics: Dict[str, Metric] = {}
        self.collection_interval = 10  # seconds
        self.is_collecting = False
        self._collection_task = None

    async def start_collection(self):
        """Spustí continuous metrics collection"""
        if self.is_collecting:
            return

        self.is_collecting = True
        tracemalloc.start()
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Metrics collection started")

    async def stop_collection(self):
        """Zastaví metrics collection"""
        self.is_collecting = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        tracemalloc.stop()
        logger.info("Metrics collection stopped")

    async def _collection_loop(self):
        """Hlavní smyčka pro sběr metrik"""
        while self.is_collecting:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self._store_system_metrics(system_metrics)

                # Collect memory profiling
                memory_metrics = self._collect_memory_metrics()
                self._store_memory_metrics(memory_metrics)

                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5)  # Short delay on error

    def _collect_system_metrics(self) -> SystemMetrics:
        """Sbírá systémové metriky"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent

            # Network metrics
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv
            }

            # Process metrics
            process_count = len(psutil.pids())

            # Load average (Unix-like systems)
            try:
                load_average = psutil.getloadavg()[0]
            except (AttributeError, OSError):
                load_average = 0.0  # Windows fallback

            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                disk_usage_percent=disk_usage_percent,
                network_io_bytes=network_io,
                process_count=process_count,
                load_average=load_average,
                timestamp=time.time()
            )

        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
            return SystemMetrics(0, 0, 0, 0, {}, 0, 0, time.time())

    def _collect_memory_metrics(self) -> Dict[str, Any]:
        """Sbírá memory profiling metriky"""
        try:
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()

                return {
                    'current_memory_mb': current / (1024**2),
                    'peak_memory_mb': peak / (1024**2),
                    'timestamp': time.time()
                }
            else:
                return {'current_memory_mb': 0, 'peak_memory_mb': 0, 'timestamp': time.time()}

        except Exception as e:
            logger.error(f"Memory metrics collection failed: {e}")
            return {'current_memory_mb': 0, 'peak_memory_mb': 0, 'timestamp': time.time()}

    def _store_system_metrics(self, metrics: SystemMetrics):
        """Uloží systémové metriky"""
        timestamp = metrics.timestamp

        # Store individual metrics
        self.metrics_history['cpu_percent'].append((timestamp, metrics.cpu_percent))
        self.metrics_history['memory_percent'].append((timestamp, metrics.memory_percent))
        self.metrics_history['memory_used_gb'].append((timestamp, metrics.memory_used_gb))
        self.metrics_history['disk_usage_percent'].append((timestamp, metrics.disk_usage_percent))
        self.metrics_history['process_count'].append((timestamp, metrics.process_count))
        self.metrics_history['load_average'].append((timestamp, metrics.load_average))

        # Network metrics
        for key, value in metrics.network_io_bytes.items():
            self.metrics_history[f'network_{key}'].append((timestamp, value))

    def _store_memory_metrics(self, metrics: Dict[str, Any]):
        """Uloží memory metriky"""
        timestamp = metrics['timestamp']

        self.metrics_history['memory_current_mb'].append((timestamp, metrics['current_memory_mb']))
        self.metrics_history['memory_peak_mb'].append((timestamp, metrics['peak_memory_mb']))

    def record_custom_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Dict[str, str] = None,
        unit: str = ""
    ):
        """Zaznamenává custom metriku"""

        metric = Metric(
            name=name,
            metric_type=metric_type,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            unit=unit
        )

        self.custom_metrics[name] = metric
        self.metrics_history[name].append((metric.timestamp, metric.value))

    def get_metric_history(
        self,
        metric_name: str,
        time_range_minutes: int = 60
    ) -> List[Tuple[float, float]]:
        """Získá historii metriky"""

        if metric_name not in self.metrics_history:
            return []

        cutoff_time = time.time() - (time_range_minutes * 60)

        return [
            (timestamp, value)
            for timestamp, value in self.metrics_history[metric_name]
            if timestamp >= cutoff_time
        ]

class AlertManager:
    """Manager pro alerting systém"""

    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable] = []
        self.alert_history: deque = deque(maxlen=1000)
        self._setup_default_alerts()

    def _setup_default_alerts(self):
        """Nastaví default alerts pro research system"""

        # System resource alerts
        self.add_alert(
            "high_cpu",
            "CPU Usage High",
            "cpu_percent > 80",
            80.0,
            AlertSeverity.WARNING
        )

        self.add_alert(
            "high_memory",
            "Memory Usage High",
            "memory_percent > 85",
            85.0,
            AlertSeverity.WARNING
        )

        self.add_alert(
            "critical_memory",
            "Memory Usage Critical",
            "memory_percent > 95",
            95.0,
            AlertSeverity.CRITICAL
        )

        # Research-specific alerts
        self.add_alert(
            "low_cache_hit_ratio",
            "Cache Hit Ratio Low",
            "cache_hit_ratio < 0.3",
            0.3,
            AlertSeverity.WARNING
        )

        self.add_alert(
            "slow_response_time",
            "Response Time Slow",
            "avg_response_time > 5000",
            5000.0,
            AlertSeverity.WARNING
        )

        self.add_alert(
            "high_error_rate",
            "Error Rate High",
            "error_rate > 0.1",
            0.1,
            AlertSeverity.ERROR
        )

    def add_alert(
        self,
        alert_id: str,
        name: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity
    ):
        """Přidá nový alert"""

        alert = Alert(
            id=alert_id,
            name=name,
            condition=condition,
            threshold=threshold,
            severity=severity
        )

        self.alerts[alert_id] = alert

    def add_alert_handler(self, handler: Callable[[Alert, float], None]):
        """Přidá handler pro alerts"""
        self.alert_handlers.append(handler)

    async def check_alerts(self, metrics: Dict[str, float]):
        """Kontroluje alert conditions"""

        for alert in self.alerts.values():
            if not alert.enabled:
                continue

            try:
                # Simple condition evaluation
                should_trigger = self._evaluate_condition(alert.condition, metrics)

                if should_trigger:
                    await self._trigger_alert(alert, metrics)

            except Exception as e:
                logger.error(f"Alert evaluation failed for {alert.id}: {e}")

    def _evaluate_condition(self, condition: str, metrics: Dict[str, float]) -> bool:
        """Vyhodnotí alert condition"""

        # Simple condition parser (can be extended)
        try:
            # Replace metric names with values
            for metric_name, value in metrics.items():
                condition = condition.replace(metric_name, str(value))

            # Evaluate the condition
            return eval(condition)

        except Exception as e:
            logger.error(f"Condition evaluation failed: {condition}, {e}")
            return False

    async def _trigger_alert(self, alert: Alert, metrics: Dict[str, float]):
        """Spustí alert"""

        current_time = time.time()

        # Check if alert was recently triggered (avoid spam)
        if (alert.last_triggered and
            current_time - alert.last_triggered < 300):  # 5 minutes cooldown
            return

        alert.last_triggered = current_time
        alert.trigger_count += 1

        # Create alert event
        alert_event = {
            'alert_id': alert.id,
            'alert_name': alert.name,
            'severity': alert.severity.value,
            'condition': alert.condition,
            'threshold': alert.threshold,
            'triggered_at': current_time,
            'metrics': metrics.copy()
        }

        self.alert_history.append(alert_event)

        # Call handlers
        for handler in self.alert_handlers:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, handler, alert, metrics.get('value', 0.0)
                )
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        logger.warning(f"ALERT TRIGGERED: {alert.name} - {alert.condition}")

class ResearchMetricsDashboard:
    """Dashboard pro research-specific metriky"""

    def __init__(self):
        self.research_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'ml_optimizations': 0,
            'rate_limit_blocks': 0,
            'token_savings_bytes': 0,
            'papers_indexed': 0,
            'semantic_searches': 0,
            'citations_managed': 0,
            'projects_created': 0,
            'annotations_added': 0
        }

        self.performance_targets = {
            'cache_hit_ratio': 0.7,  # 70% target
            'avg_response_time_ms': 2000,  # 2s target
            'error_rate': 0.05,  # 5% max
            'success_rate': 0.95  # 95% min
        }

    def update_research_metric(self, metric_name: str, value: float):
        """Aktualizuje research metriku"""
        if metric_name in self.research_metrics:
            self.research_metrics[metric_name] = value
        else:
            logger.warning(f"Unknown research metric: {metric_name}")

    def increment_research_metric(self, metric_name: str, increment: float = 1.0):
        """Inkrementuje research metriku"""
        if metric_name in self.research_metrics:
            self.research_metrics[metric_name] += increment
        else:
            logger.warning(f"Unknown research metric: {metric_name}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Získá data pro dashboard"""

        # Calculate derived metrics
        total_cache_ops = self.research_metrics['cache_hits'] + self.research_metrics['cache_misses']
        cache_hit_ratio = (
            self.research_metrics['cache_hits'] / max(total_cache_ops, 1)
        )

        success_rate = (
            self.research_metrics['successful_queries'] /
            max(self.research_metrics['total_queries'], 1)
        )

        return {
            'research_metrics': self.research_metrics.copy(),
            'derived_metrics': {
                'cache_hit_ratio': cache_hit_ratio,
                'success_rate': success_rate,
                'optimization_ratio': (
                    self.research_metrics['ml_optimizations'] /
                    max(self.research_metrics['total_queries'], 1)
                )
            },
            'performance_targets': self.performance_targets.copy(),
            'target_compliance': {
                'cache_hit_ratio': cache_hit_ratio >= self.performance_targets['cache_hit_ratio'],
                'success_rate': success_rate >= self.performance_targets['success_rate']
            },
            'timestamp': time.time()
        }

class PerformanceMonitor:
    """Hlavní performance monitoring systém"""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.dashboard = ResearchMetricsDashboard()
        self.is_monitoring = False
        self._monitoring_task = None
        self._setup_alert_handlers()

    def _setup_alert_handlers(self):
        """Nastaví alert handlers"""

        def log_alert_handler(alert: Alert, value: float):
            logger.warning(f"ALERT: {alert.name} - Value: {value}, Threshold: {alert.threshold}")

        def email_alert_handler(alert: Alert, value: float):
            # Placeholder for email notifications
            if alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
                logger.critical(f"CRITICAL ALERT: {alert.name} requires immediate attention!")

        self.alert_manager.add_alert_handler(log_alert_handler)
        self.alert_manager.add_alert_handler(email_alert_handler)

    async def start_monitoring(self):
        """Spustí complete monitoring"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        await self.metrics_collector.start_collection()
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("Performance monitoring started")

    async def stop_monitoring(self):
        """Zastaví monitoring"""
        self.is_monitoring = False
        await self.metrics_collector.stop_collection()

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self):
        """Hlavní monitoring smyčka"""
        while self.is_monitoring:
            try:
                # Get current metrics
                current_metrics = self._get_current_metrics()

                # Check alerts
                await self.alert_manager.check_alerts(current_metrics)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)

    def _get_current_metrics(self) -> Dict[str, float]:
        """Získá aktuální metriky pro alert checking"""

        # Get latest system metrics
        metrics = {}

        for metric_name in ['cpu_percent', 'memory_percent', 'load_average']:
            history = self.metrics_collector.get_metric_history(metric_name, 5)
            if history:
                metrics[metric_name] = history[-1][1]  # Latest value

        # Add research metrics
        dashboard_data = self.dashboard.get_dashboard_data()
        metrics.update(dashboard_data['derived_metrics'])

        return metrics

    def record_research_event(self, event_type: str, **kwargs):
        """Zaznamenává research event"""

        if event_type == 'query_executed':
            self.dashboard.increment_research_metric('total_queries')
            if kwargs.get('success', False):
                self.dashboard.increment_research_metric('successful_queries')

        elif event_type == 'cache_hit':
            self.dashboard.increment_research_metric('cache_hits')

        elif event_type == 'cache_miss':
            self.dashboard.increment_research_metric('cache_misses')

        elif event_type == 'ml_optimization':
            self.dashboard.increment_research_metric('ml_optimizations')

        elif event_type == 'rate_limit_block':
            self.dashboard.increment_research_metric('rate_limit_blocks')

        elif event_type == 'token_savings':
            bytes_saved = kwargs.get('bytes_saved', 0)
            self.dashboard.increment_research_metric('token_savings_bytes', bytes_saved)

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Získá komprehensivní performance report"""

        return {
            'system_metrics': {
                name: self.metrics_collector.get_metric_history(name, 60)[-10:]  # Last 10 points
                for name in ['cpu_percent', 'memory_percent', 'load_average']
            },
            'research_dashboard': self.dashboard.get_dashboard_data(),
            'recent_alerts': list(self.alert_manager.alert_history)[-10:],
            'alert_summary': {
                alert_id: {
                    'name': alert.name,
                    'enabled': alert.enabled,
                    'trigger_count': alert.trigger_count,
                    'last_triggered': alert.last_triggered
                }
                for alert_id, alert in self.alert_manager.alerts.items()
            },
            'monitoring_status': {
                'is_active': self.is_monitoring,
                'collection_interval': self.metrics_collector.collection_interval,
                'metrics_tracked': len(self.metrics_collector.metrics_history)
            },
            'timestamp': time.time()
        }

# Factory funkce
async def create_performance_monitor() -> PerformanceMonitor:
    """Factory pro vytvoření performance monitoru"""
    monitor = PerformanceMonitor()
    logger.info("Performance Monitor initialized")
    return monitor
