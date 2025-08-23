"""
Advanced Connection Pool Optimizer - Phase 1 Fixed
Pokročilé connection pooling optimalizace pro M1 MacBook Air
- Intelligent connection management
- Adaptive pool sizing
- M1 network stack optimizations
- Circuit breaker pattern
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """Stavy connection pool"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"

@dataclass
class PoolConfig:
    """Konfigurace connection pool"""
    min_connections: int = 5
    max_connections: int = 50
    connection_timeout: float = 30.0
    read_timeout: float = 60.0
    total_timeout: float = 120.0
    keepalive_timeout: float = 300.0
    enable_m1_optimizations: bool = True
    adaptive_sizing: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0

@dataclass
class ConnectionMetrics:
    """Metriky connection performance"""
    active_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    current_pool_size: int = 0
    circuit_breaker_trips: int = 0
    adaptive_adjustments: int = 0

class CircuitBreaker:
    """Circuit breaker pro connection failure handling"""

    def __init__(self, failure_threshold: int, timeout: float):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = ConnectionState.HEALTHY

    def record_success(self):
        """Zaznamenává úspěšný request"""
        self.failure_count = 0
        if self.state == ConnectionState.RECOVERING:
            self.state = ConnectionState.HEALTHY

    def record_failure(self):
        """Zaznamenává neúspěšný request"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = ConnectionState.FAILED

    def can_execute(self) -> bool:
        """Kontroluje, zda může být request vykonán"""
        if self.state == ConnectionState.HEALTHY:
            return True

        if self.state == ConnectionState.FAILED:
            # Zkus recovery po timeout
            if time.time() - self.last_failure_time > self.timeout:
                self.state = ConnectionState.RECOVERING
                return True
            return False

        # RECOVERING state
        return True

class AdaptiveConnectionPool:
    """
    Adaptivní connection pool s M1 optimalizacemi
    - Dynamické přizpůsobení velikosti poolu
    - Circuit breaker pattern
    - M1 network stack optimizations
    """

    def __init__(self, config: PoolConfig):
        self.config = config
        self.metrics = ConnectionMetrics()
        self.circuit_breaker = CircuitBreaker(
            config.circuit_breaker_threshold,
            config.circuit_breaker_timeout
        )
        self.connector = None
        self.session = None
        self._pool_adjustment_lock = asyncio.Lock()
        self._last_adjustment_time = time.time()

    async def initialize(self):
        """Inicializuje connection pool s M1 optimalizacemi"""
        connector_kwargs = {
            'limit': self.config.max_connections,
            'limit_per_host': self.config.max_connections,
            'keepalive_timeout': self.config.keepalive_timeout,
            'enable_cleanup_closed': True,
        }

        # M1 specific optimizations
        if self.config.enable_m1_optimizations:
            connector_kwargs.update({
                'use_dns_cache': True,
                'ttl_dns_cache': 300,
                'family': 0,  # Allow both IPv4 and IPv6
                'happy_eyeballs_delay': 0.25,  # Faster connection attempts
            })

        self.connector = aiohttp.TCPConnector(**connector_kwargs)

        timeout = aiohttp.ClientTimeout(
            total=self.config.total_timeout,
            connect=self.config.connection_timeout,
            sock_read=self.config.read_timeout
        )

        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout
        )

        self.metrics.current_pool_size = self.config.min_connections
        logger.info(f"Connection pool initialized with {self.metrics.current_pool_size} connections")

    @asynccontextmanager
    async def get_session(self):
        """Context manager pro získání session s metrics tracking"""
        if not self.circuit_breaker.can_execute():
            raise ConnectionError("Circuit breaker is open")

        start_time = time.time()
        self.metrics.active_connections += 1
        self.metrics.total_requests += 1

        try:
            yield self.session

            # Record success
            response_time = time.time() - start_time
            self._update_response_time(response_time)
            self.circuit_breaker.record_success()
            self.metrics.successful_requests += 1

            # Adaptive pool sizing
            if self.config.adaptive_sizing:
                await self._adjust_pool_size()

        except Exception as e:
            self.circuit_breaker.record_failure()
            self.metrics.failed_requests += 1
            logger.error(f"Connection pool error: {e}")
            raise
        finally:
            self.metrics.active_connections -= 1

    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Provede HTTP request s pool management"""
        async with self.get_session() as session:
            return await session.request(method, url, **kwargs)

    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """HTTP GET s pool management"""
        return await self.request('GET', url, **kwargs)

    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """HTTP POST s pool management"""
        return await self.request('POST', url, **kwargs)

    async def _adjust_pool_size(self):
        """Adaptivní úprava velikosti connection poolu"""
        async with self._pool_adjustment_lock:
            current_time = time.time()

            # Adjust only every 30 seconds
            if current_time - self._last_adjustment_time < 30:
                return

            self._last_adjustment_time = current_time

            # Calculate success rate
            total_requests = self.metrics.total_requests
            if total_requests < 10:  # Need enough data
                return

            success_rate = self.metrics.successful_requests / total_requests
            avg_active = self.metrics.active_connections
            current_size = self.metrics.current_pool_size

            # Decision logic
            should_increase = (
                success_rate > 0.95 and
                avg_active > current_size * 0.8 and
                current_size < self.config.max_connections
            )

            should_decrease = (
                success_rate < 0.9 or
                avg_active < current_size * 0.3
            ) and current_size > self.config.min_connections

            if should_increase:
                new_size = min(current_size + 5, self.config.max_connections)
                await self._resize_pool(new_size)
                logger.info(f"Pool size increased to {new_size}")

            elif should_decrease:
                new_size = max(current_size - 3, self.config.min_connections)
                await self._resize_pool(new_size)
                logger.info(f"Pool size decreased to {new_size}")

    async def _resize_pool(self, new_size: int):
        """Změní velikost connection poolu"""
        # Close current connector
        if self.connector:
            await self.connector.close()

        if self.session:
            await self.session.close()

        # Create new connector with updated size
        old_size = self.metrics.current_pool_size
        self.config.max_connections = new_size
        await self.initialize()

        self.metrics.current_pool_size = new_size
        self.metrics.adaptive_adjustments += 1

        logger.info(f"Pool resized from {old_size} to {new_size} connections")

    def _update_response_time(self, response_time: float):
        """Aktualizuje průměrný response time"""
        if self.metrics.avg_response_time == 0.0:
            self.metrics.avg_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.avg_response_time = (
                (1 - alpha) * self.metrics.avg_response_time +
                alpha * response_time
            )

    async def get_pool_stats(self) -> Dict[str, Any]:
        """Získá detailní statistiky pool performance"""
        total_requests = self.metrics.total_requests
        success_rate = (
            self.metrics.successful_requests / total_requests
            if total_requests > 0 else 0.0
        )

        return {
            'pool_config': {
                'current_size': self.metrics.current_pool_size,
                'min_connections': self.config.min_connections,
                'max_connections': self.config.max_connections,
                'adaptive_sizing': self.config.adaptive_sizing,
                'm1_optimizations': self.config.enable_m1_optimizations
            },
            'performance_metrics': {
                'active_connections': self.metrics.active_connections,
                'total_requests': self.metrics.total_requests,
                'success_rate': success_rate,
                'avg_response_time_ms': self.metrics.avg_response_time * 1000,
                'adaptive_adjustments': self.metrics.adaptive_adjustments
            },
            'circuit_breaker': {
                'state': self.circuit_breaker.state.value,
                'failure_count': self.circuit_breaker.failure_count,
                'trips': self.metrics.circuit_breaker_trips
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check connection poolu"""
        try:
            # Test connection
            start_time = time.time()
            async with self.get_session() as session:
                # Simple health check - ping httpbin
                async with session.get('http://httpbin.org/status/200', timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    status_ok = resp.status == 200

            ping_time = (time.time() - start_time) * 1000

            return {
                'status': 'healthy' if status_ok else 'degraded',
                'ping_time_ms': ping_time,
                'circuit_breaker_state': self.circuit_breaker.state.value,
                'pool_utilization': self.metrics.active_connections / max(self.metrics.current_pool_size, 1)
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'circuit_breaker_state': self.circuit_breaker.state.value
            }

    async def close(self):
        """Uzavře connection pool"""
        if self.session:
            await self.session.close()

        if self.connector:
            await self.connector.close()

        logger.info("Connection pool closed")

# Factory funkce pro easy setup
async def create_connection_pool(
    min_connections: int = 5,
    max_connections: int = 50,
    enable_m1_optimizations: bool = True,
    **kwargs
) -> AdaptiveConnectionPool:
    """Factory funkce pro vytvoření optimized connection poolu"""

    config = PoolConfig(
        min_connections=min_connections,
        max_connections=max_connections,
        enable_m1_optimizations=enable_m1_optimizations,
        **kwargs
    )

    pool = AdaptiveConnectionPool(config)
    await pool.initialize()

    # Test pool functionality
    health = await pool.health_check()
    if health['status'] == 'unhealthy':
        logger.warning(f"Connection pool health check failed: {health.get('error')}")
    else:
        logger.info(f"Connection pool ready - status: {health['status']}")

    return pool
