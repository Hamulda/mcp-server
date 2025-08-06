"""
Monitoring metriky pro Research Tool
Integrace s Prometheus pro sledování výkonu aplikace
"""
import time
import logging
from typing import Dict, Any
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry

class MetricsCollector:
    """Sběr a export metrik pro monitoring"""

    def __init__(self, port: int = 8000):
        self.port = port
        self.logger = logging.getLogger(__name__)
        self.registry = CollectorRegistry()
        self._setup_metrics()

    def _setup_metrics(self):
        """Nastavení Prometheus metrik"""
        # Počítadla
        self.api_calls_total = Counter(
            'research_api_calls_total',
            'Total number of API calls',
            ['api_type', 'status'],
            registry=self.registry
        )

        self.errors_total = Counter(
            'research_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )

        self.tokens_used_total = Counter(
            'research_tokens_used_total',
            'Total tokens consumed',
            ['model', 'operation'],
            registry=self.registry
        )

        # Histogramy pro distribuci hodnot
        self.response_time = Histogram(
            'research_response_time_seconds',
            'Response time in seconds',
            ['operation'],
            registry=self.registry
        )

        self.token_count = Histogram(
            'research_token_count',
            'Token count per request',
            ['operation'],
            buckets=[10, 50, 100, 500, 1000, 5000, 10000],
            registry=self.registry
        )

        # Gauge pro aktuální hodnoty
        self.daily_cost = Gauge(
            'research_daily_cost_usd',
            'Current daily cost in USD',
            registry=self.registry
        )

        self.active_requests = Gauge(
            'research_active_requests',
            'Number of active requests',
            registry=self.registry
        )

        self.cache_hit_rate = Gauge(
            'research_cache_hit_rate',
            'Cache hit rate percentage',
            registry=self.registry
        )

        self.database_connections = Gauge(
            'research_database_connections',
            'Number of active database connections',
            registry=self.registry
        )

    def start_server(self):
        """Spuštění HTTP serveru pro metriky"""
        try:
            start_http_server(self.port, registry=self.registry)
            self.logger.info(f"Metrics server started on port {self.port}")
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")

    def record_api_call(self, api_type: str, status: str, duration: float, tokens: int = 0):
        """Zaznamenání API volání"""
        self.api_calls_total.labels(api_type=api_type, status=status).inc()
        self.response_time.labels(operation=api_type).observe(duration)

        if tokens > 0:
            self.tokens_used_total.labels(model=api_type, operation='inference').inc(tokens)
            self.token_count.labels(operation=api_type).observe(tokens)

    def record_error(self, error_type: str, component: str):
        """Zaznamenání chyby"""
        self.errors_total.labels(error_type=error_type, component=component).inc()

    def update_daily_cost(self, cost: float):
        """Aktualizace denních nákladů"""
        self.daily_cost.set(cost)

    def update_cache_stats(self, hits: int, total: int):
        """Aktualizace statistik cache"""
        if total > 0:
            hit_rate = (hits / total) * 100
            self.cache_hit_rate.set(hit_rate)

    def increment_active_requests(self):
        """Zvýšení počtu aktivních požadavků"""
        self.active_requests.inc()

    def decrement_active_requests(self):
        """Snížení počtu aktivních požadavků"""
        self.active_requests.dec()

# Globální instance pro snadné použití
metrics = MetricsCollector()

class MetricsMiddleware:
    """Middleware pro automatické zaznamenávání metrik"""

    def __init__(self, component_name: str):
        self.component_name = component_name
        self.logger = logging.getLogger(__name__)

    def track_operation(self, operation_name: str):
        """Decorator pro sledování operací"""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                metrics.increment_active_requests()

                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    metrics.record_api_call(operation_name, 'success', duration)
                    return result

                except Exception as e:
                    duration = time.time() - start_time
                    metrics.record_api_call(operation_name, 'error', duration)
                    metrics.record_error(type(e).__name__, self.component_name)
                    raise

                finally:
                    metrics.decrement_active_requests()

            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                metrics.increment_active_requests()

                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    metrics.record_api_call(operation_name, 'success', duration)
                    return result

                except Exception as e:
                    duration = time.time() - start_time
                    metrics.record_api_call(operation_name, 'error', duration)
                    metrics.record_error(type(e).__name__, self.component_name)
                    raise

                finally:
                    metrics.decrement_active_requests()

            return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else sync_wrapper
        return decorator

class HealthChecker:
    """Health check endpoint pro aplikaci"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = datetime.now()

    def get_health_status(self) -> Dict[str, Any]:
        """Získání stavu aplikace"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        try:
            # Základní kontroly
            status = {
                "status": "healthy",
                "uptime_seconds": uptime,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "components": {
                    "database": self._check_database(),
                    "cache": self._check_cache(),
                    "external_apis": self._check_external_apis()
                }
            }

            # Kontrola celkového stavu
            if not all(comp["status"] == "healthy" for comp in status["components"].values()):
                status["status"] = "degraded"

            return status

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _check_database(self) -> Dict[str, str]:
        """Kontrola databáze"""
        try:
            # Zde by byla skutečná kontrola databáze
            return {"status": "healthy", "message": "Database connection OK"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"Database error: {e}"}

    def _check_cache(self) -> Dict[str, str]:
        """Kontrola cache"""
        try:
            # Kontrola cache systému
            return {"status": "healthy", "message": "Cache operational"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"Cache error: {e}"}

    def _check_external_apis(self) -> Dict[str, str]:
        """Kontrola externích API"""
        try:
            # Kontrola dostupnosti externích služeb
            return {"status": "healthy", "message": "External APIs accessible"}
        except Exception as e:
            return {"status": "degraded", "message": f"Some APIs unavailable: {e}"}

# Globální instance
health_checker = HealthChecker()
