"""
Advanced Monitoring System - Enterprise-grade monitoring
Rozšířené metriky, zdraví systému, performance tracking
"""

import time
import psutil
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import threading

logger = logging.getLogger(__name__)

# Prometheus metriky
REGISTRY = CollectorRegistry()

# Request metriky
REQUEST_COUNT = Counter(
    'biohack_requests_total',
    'Total number of requests',
    ['endpoint', 'method', 'status'],
    registry=REGISTRY
)

REQUEST_DURATION = Histogram(
    'biohack_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint', 'method'],
    registry=REGISTRY
)

# Cache metriky
CACHE_HITS = Counter(
    'biohack_cache_hits_total',
    'Total cache hits',
    ['cache_type'],
    registry=REGISTRY
)

CACHE_MISSES = Counter(
    'biohack_cache_misses_total',
    'Total cache misses',
    ['cache_type'],
    registry=REGISTRY
)

# Systémové metriky
SYSTEM_CPU_USAGE = Gauge(
    'biohack_system_cpu_percent',
    'System CPU usage percentage',
    registry=REGISTRY
)

SYSTEM_MEMORY_USAGE = Gauge(
    'biohack_system_memory_percent',
    'System memory usage percentage',
    registry=REGISTRY
)

ACTIVE_CONNECTIONS = Gauge(
    'biohack_active_connections',
    'Number of active connections',
    registry=REGISTRY
)

# AI/Research metriky
RESEARCH_QUERIES = Counter(
    'biohack_research_queries_total',
    'Total research queries',
    ['research_type', 'success'],
    registry=REGISTRY
)

AI_MODEL_USAGE = Counter(
    'biohack_ai_model_calls_total',
    'AI model usage',
    ['model_name', 'operation'],
    registry=REGISTRY
)

@dataclass
class HealthCheck:
    """Struktura pro health check výsledky"""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    response_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

class AdvancedMonitoringSystem:
    """
    Pokročilý monitoring systém pro biohacking research tool
    """

    def __init__(self):
        self.start_time = datetime.now()
        self.health_checks: List[HealthCheck] = []
        self.performance_metrics = {
            "requests_per_minute": 0,
            "avg_response_time": 0.0,
            "error_rate": 0.0,
            "cache_hit_rate": 0.0
        }

        # Čítače pro kalkulace
        self._request_times = []
        self._error_count = 0
        self._total_requests = 0

        # Background monitoring
        self._monitoring_task = None
        self._lock = threading.Lock()

    async def start_monitoring(self):
        """Spustí background monitoring"""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(
                self._background_monitoring()
            )
            logger.info("📊 Advanced monitoring started")

    async def stop_monitoring(self):
        """Zastaví background monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("📊 Monitoring stopped")

    async def _background_monitoring(self):
        """Background úloha pro monitoring systému"""
        while True:
            try:
                await self._update_system_metrics()
                await self._health_check_all_components()
                await asyncio.sleep(30)  # Update každých 30 sekund

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                await asyncio.sleep(60)  # Delší pauza při chybě

    async def _update_system_metrics(self):
        """Aktualizuje systémové metriky"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_CPU_USAGE.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(memory.percent)

            # Disk usage pro cache
            disk_usage = psutil.disk_usage('/')

            logger.debug(f"📊 System metrics - CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%")

        except Exception as e:
            logger.error(f"❌ Failed to update system metrics: {e}")

    async def _health_check_all_components(self):
        """Provede health check všech komponent"""
        components_to_check = [
            ("database", self._health_check_database),
            ("cache", self._health_check_cache),
            ("ai_models", self._health_check_ai),
            ("semantic_search", self._health_check_semantic_search),
            ("scrapers", self._health_check_scrapers)
        ]

        current_health = []

        for component_name, check_func in components_to_check:
            try:
                health_result = await check_func()
                current_health.append(health_result)

            except Exception as e:
                current_health.append(HealthCheck(
                    component=component_name,
                    status="unhealthy",
                    message=f"Health check failed: {e}",
                    response_time=0.0
                ))

        with self._lock:
            self.health_checks = current_health

    async def _health_check_database(self) -> HealthCheck:
        """Health check pro databázi"""
        start_time = time.time()

        try:
            # Pokus o připojení k databázi
            from unified_config import get_config
            config = get_config()

            if config.database.type == "sqlite":
                # SQLite check
                import sqlite3
                conn = sqlite3.connect(":memory:")
                conn.execute("SELECT 1")
                conn.close()

                return HealthCheck(
                    component="database",
                    status="healthy",
                    message="SQLite accessible",
                    response_time=time.time() - start_time
                )
            else:
                return HealthCheck(
                    component="database",
                    status="degraded",
                    message="Database type not fully tested",
                    response_time=time.time() - start_time
                )

        except Exception as e:
            return HealthCheck(
                component="database",
                status="unhealthy",
                message=f"Database error: {e}",
                response_time=time.time() - start_time
            )

    async def _health_check_cache(self) -> HealthCheck:
        """Health check pro cache systém"""
        start_time = time.time()

        try:
            from cache.redis_cache_system import get_redis_cache
            cache = await get_redis_cache()

            # Test cache operace
            test_key = "health_check_test"
            test_value = {"timestamp": datetime.now().isoformat()}

            await cache.set(test_key, test_value, ttl=60)
            retrieved = await cache.get(test_key)
            await cache.delete(test_key)

            if retrieved:
                return HealthCheck(
                    component="cache",
                    status="healthy",
                    message="Redis cache operational",
                    response_time=time.time() - start_time
                )
            else:
                return HealthCheck(
                    component="cache",
                    status="degraded",
                    message="Cache retrieval failed",
                    response_time=time.time() - start_time
                )

        except Exception as e:
            return HealthCheck(
                component="cache",
                status="degraded",
                message=f"Cache unavailable, using fallback: {e}",
                response_time=time.time() - start_time
            )

    async def _health_check_ai(self) -> HealthCheck:
        """Health check pro AI komponenty"""
        start_time = time.time()

        try:
            from ai.local_ai_adapter import M1OptimizedOllamaClient

            # Test AI dostupnosti
            async with M1OptimizedOllamaClient() as ai_client:
                test_response = await ai_client.simple_query(
                    "Test health check",
                    max_tokens=10
                )

                if test_response:
                    return HealthCheck(
                        component="ai_models",
                        status="healthy",
                        message="AI models responsive",
                        response_time=time.time() - start_time
                    )
                else:
                    return HealthCheck(
                        component="ai_models",
                        status="degraded",
                        message="AI models slow or unresponsive",
                        response_time=time.time() - start_time
                    )

        except Exception as e:
            return HealthCheck(
                component="ai_models",
                status="unhealthy",
                message=f"AI unavailable: {e}",
                response_time=time.time() - start_time
            )

    async def _health_check_semantic_search(self) -> HealthCheck:
        """Health check pro sémantické vyhledávání"""
        start_time = time.time()

        try:
            from ai.semantic_search_system import get_semantic_search

            semantic_search = await get_semantic_search()
            stats = await semantic_search.get_collection_stats()

            if "error" not in stats:
                return HealthCheck(
                    component="semantic_search",
                    status="healthy",
                    message=f"Semantic search active, {stats.get('total_documents', 0)} docs indexed",
                    response_time=time.time() - start_time,
                    details=stats
                )
            else:
                return HealthCheck(
                    component="semantic_search",
                    status="degraded",
                    message=f"Semantic search issues: {stats['error']}",
                    response_time=time.time() - start_time
                )

        except Exception as e:
            return HealthCheck(
                component="semantic_search",
                status="unhealthy",
                message=f"Semantic search unavailable: {e}",
                response_time=time.time() - start_time
            )

    async def _health_check_scrapers(self) -> HealthCheck:
        """Health check pro scraping komponenty"""
        start_time = time.time()

        try:
            from academic_scraper import create_scraping_orchestrator

            # Test vytvoření orchestrátoru
            orchestrator = await create_scraping_orchestrator(
                max_concurrent=1,
                timeout=5
            )

            if orchestrator:
                return HealthCheck(
                    component="scrapers",
                    status="healthy",
                    message="Scraping orchestrator ready",
                    response_time=time.time() - start_time
                )
            else:
                return HealthCheck(
                    component="scrapers",
                    status="degraded",
                    message="Scraper initialization issues",
                    response_time=time.time() - start_time
                )

        except Exception as e:
            return HealthCheck(
                component="scrapers",
                status="unhealthy",
                message=f"Scrapers unavailable: {e}",
                response_time=time.time() - start_time
            )

    def record_request(self, endpoint: str, method: str, status_code: int, duration: float):
        """Zaznamenává request metriky"""
        status = "success" if 200 <= status_code < 400 else "error"

        # Prometheus metriky
        REQUEST_COUNT.labels(
            endpoint=endpoint,
            method=method,
            status=status
        ).inc()

        REQUEST_DURATION.labels(
            endpoint=endpoint,
            method=method
        ).observe(duration)

        # Interní statistiky
        with self._lock:
            self._request_times.append(duration)
            self._total_requests += 1

            if status == "error":
                self._error_count += 1

            # Udržování sliding window (posledních 1000 requestů)
            if len(self._request_times) > 1000:
                self._request_times = self._request_times[-1000:]

    def record_cache_operation(self, cache_type: str, hit: bool):
        """Zaznamenává cache operace"""
        if hit:
            CACHE_HITS.labels(cache_type=cache_type).inc()
        else:
            CACHE_MISSES.labels(cache_type=cache_type).inc()

    def record_research_query(self, research_type: str, success: bool):
        """Zaznamenává research dotazy"""
        status = "success" if success else "error"
        RESEARCH_QUERIES.labels(
            research_type=research_type,
            success=status
        ).inc()

    def record_ai_usage(self, model_name: str, operation: str):
        """Zaznamenává AI usage"""
        AI_MODEL_USAGE.labels(
            model_name=model_name,
            operation=operation
        ).inc()

    def get_health_status(self) -> Dict[str, Any]:
        """Vrací celkový health status"""
        with self._lock:
            healthy_count = sum(1 for hc in self.health_checks if hc.status == "healthy")
            total_components = len(self.health_checks)

            if total_components == 0:
                overall_status = "unknown"
            elif healthy_count == total_components:
                overall_status = "healthy"
            elif healthy_count > total_components // 2:
                overall_status = "degraded"
            else:
                overall_status = "unhealthy"

            return {
                "overall_status": overall_status,
                "healthy_components": healthy_count,
                "total_components": total_components,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "components": [
                    {
                        "name": hc.component,
                        "status": hc.status,
                        "message": hc.message,
                        "response_time": hc.response_time,
                        "last_check": hc.timestamp.isoformat()
                    }
                    for hc in self.health_checks
                ],
                "performance_metrics": self._calculate_performance_metrics()
            }

    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Kalkuluje performance metriky"""
        if not self._request_times:
            return {
                "avg_response_time": 0.0,
                "error_rate": 0.0,
                "requests_per_minute": 0.0
            }

        # Průměrný response time
        avg_response_time = sum(self._request_times) / len(self._request_times)

        # Error rate
        error_rate = (self._error_count / self._total_requests) * 100 if self._total_requests > 0 else 0.0

        # Requests per minute (odhad na základě posledních requestů)
        uptime_minutes = (datetime.now() - self.start_time).total_seconds() / 60
        requests_per_minute = self._total_requests / uptime_minutes if uptime_minutes > 0 else 0.0

        return {
            "avg_response_time": avg_response_time,
            "error_rate": error_rate,
            "requests_per_minute": requests_per_minute
        }

    def get_prometheus_metrics(self) -> str:
        """Vrací Prometheus metriky"""
        return generate_latest(REGISTRY).decode('utf-8')

# Global monitoring instance
_monitoring_system = None

def get_monitoring_system() -> AdvancedMonitoringSystem:
    """Singleton pro monitoring systém"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = AdvancedMonitoringSystem()
    return _monitoring_system

async def start_monitoring():
    """Spustí monitoring systém"""
    monitoring = get_monitoring_system()
    await monitoring.start_monitoring()

async def stop_monitoring():
    """Zastaví monitoring systém"""
    monitoring = get_monitoring_system()
    await monitoring.stop_monitoring()
