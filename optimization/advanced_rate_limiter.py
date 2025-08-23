"""
Advanced Rate Limiter s adaptive throttling a circuit breaker pattern
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"      # Normální provoz
    OPEN = "open"          # Circuit otevřen - blokuje requesty
    HALF_OPEN = "half_open"  # Testovací stav

class AdaptiveRateLimiter:
    """Pokročilý rate limiter s adaptivním chováním"""

    def __init__(self):
        # Rate limiting per source
        self.source_limits = {
            "pubmed": {"requests_per_minute": 10, "burst": 5},
            "openalex": {"requests_per_minute": 20, "burst": 10},
            "wikipedia": {"requests_per_minute": 60, "burst": 20},
            "arxiv": {"requests_per_minute": 15, "burst": 8},
            "default": {"requests_per_minute": 30, "burst": 10}
        }

        # Tracking per source
        self.request_history = defaultdict(deque)
        self.response_times = defaultdict(deque)
        self.error_counts = defaultdict(int)
        self.success_counts = defaultdict(int)

        # Circuit breaker states
        self.circuit_states = defaultdict(lambda: CircuitState.CLOSED)
        self.circuit_failure_counts = defaultdict(int)
        self.circuit_last_failure = defaultdict(float)

        # Adaptive parameters
        self.failure_threshold = 5
        self.timeout_duration = 60  # seconds
        self.recovery_timeout = 30  # seconds

        # User-based quotas
        self.user_quotas = defaultdict(lambda: {"daily": 1000, "hourly": 100})
        self.user_usage = defaultdict(lambda: {"daily": 0, "hourly": 0, "last_reset": time.time()})

    async def check_rate_limit(self, source: str, user_id: str = None) -> Dict[str, Any]:
        """Kontroluje rate limit pro zdroj a uživatele"""
        current_time = time.time()

        # 1. Zkontroluj circuit breaker
        circuit_check = await self._check_circuit_breaker(source)
        if not circuit_check["allowed"]:
            return circuit_check

        # 2. Zkontroluj source rate limit
        source_check = await self._check_source_limit(source, current_time)
        if not source_check["allowed"]:
            return source_check

        # 3. Zkontroluj user quota (pokud je user_id poskytnut)
        if user_id:
            user_check = await self._check_user_quota(user_id, current_time)
            if not user_check["allowed"]:
                return user_check

        # 4. Adaptivní úprava na základě response times
        adaptive_delay = await self._calculate_adaptive_delay(source)

        return {
            "allowed": True,
            "source": source,
            "adaptive_delay": adaptive_delay,
            "remaining_requests": self._get_remaining_requests(source, current_time),
            "reset_time": self._get_reset_time(source, current_time)
        }

    async def _check_circuit_breaker(self, source: str) -> Dict[str, Any]:
        """Kontroluje circuit breaker stav"""
        state = self.circuit_states[source]
        current_time = time.time()

        if state == CircuitState.OPEN:
            # Zkontroluj, zda můžeme přejít do HALF_OPEN
            if current_time - self.circuit_last_failure[source] > self.recovery_timeout:
                self.circuit_states[source] = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker {source}: OPEN -> HALF_OPEN")
                return {"allowed": True, "circuit_state": "half_open"}
            else:
                return {
                    "allowed": False,
                    "reason": "circuit_breaker_open",
                    "circuit_state": "open",
                    "retry_after": self.recovery_timeout - (current_time - self.circuit_last_failure[source])
                }

        return {"allowed": True, "circuit_state": state.value}

    async def _check_source_limit(self, source: str, current_time: float) -> Dict[str, Any]:
        """Kontroluje rate limit pro konkrétní zdroj"""
        limits = self.source_limits.get(source, self.source_limits["default"])
        requests_per_minute = limits["requests_per_minute"]
        burst = limits["burst"]

        # Vyčisti staré requesty (starší než 1 minuta)
        minute_ago = current_time - 60
        history = self.request_history[source]

        while history and history[0] < minute_ago:
            history.popleft()

        current_requests = len(history)

        # Zkontroluj per-minute limit PŘED burst limitem
        if current_requests >= requests_per_minute:
            return {
                "allowed": False,
                "reason": "rate_limit_exceeded",
                "limit": requests_per_minute,
                "current": current_requests,
                "retry_after": 60 - (current_time - history[0]) if history else 60
            }

        # Zkontroluj burst limit
        if current_requests >= burst:
            recent_requests = sum(1 for req_time in history if req_time > current_time - 10)
            if recent_requests >= burst:
                return {
                    "allowed": False,
                    "reason": "burst_limit_exceeded",
                    "limit": burst,
                    "current": recent_requests,
                    "retry_after": 10
                }

        return {"allowed": True}

    async def _check_user_quota(self, user_id: str, current_time: float) -> Dict[str, Any]:
        """Kontroluje uživatelské kvóty"""
        usage = self.user_usage[user_id]
        quota = self.user_quotas[user_id]

        # Reset počítadel pokud je potřeba
        if current_time - usage["last_reset"] > 3600:  # Hodinový reset
            usage["hourly"] = 0
            usage["last_reset"] = current_time

        if current_time - usage["last_reset"] > 86400:  # Denní reset
            usage["daily"] = 0

        # Zkontroluj limity
        if usage["hourly"] >= quota["hourly"]:
            return {
                "allowed": False,
                "reason": "hourly_quota_exceeded",
                "quota": quota["hourly"],
                "used": usage["hourly"],
                "retry_after": 3600 - (current_time - usage["last_reset"])
            }

        if usage["daily"] >= quota["daily"]:
            return {
                "allowed": False,
                "reason": "daily_quota_exceeded",
                "quota": quota["daily"],
                "used": usage["daily"],
                "retry_after": 86400 - (current_time - usage["last_reset"])
            }

        return {"allowed": True}

    async def _calculate_adaptive_delay(self, source: str) -> float:
        """Vypočítá adaptivní delay na základě response times"""
        response_times = self.response_times[source]

        if len(response_times) < 5:
            return 0.0

        # Vypočítaj průměrný response time za posledních 10 requestů
        recent_times = list(response_times)[-10:]
        avg_response_time = statistics.mean(recent_times)

        # Adaptivní delay - delší delay pro pomalejší API
        if avg_response_time > 5.0:  # Pomalé API
            return min(2.0, avg_response_time * 0.2)
        elif avg_response_time > 2.0:  # Středně rychlé API
            return min(1.0, avg_response_time * 0.1)
        else:  # Rychlé API
            return 0.1

    def _get_remaining_requests(self, source: str, current_time: float) -> int:
        """Vrátí počet zbývajících requestů"""
        limits = self.source_limits.get(source, self.source_limits["default"])
        requests_per_minute = limits["requests_per_minute"]

        minute_ago = current_time - 60
        history = self.request_history[source]
        current_requests = sum(1 for req_time in history if req_time > minute_ago)

        return max(0, requests_per_minute - current_requests)

    def _get_reset_time(self, source: str, current_time: float) -> float:
        """Vrátí čas do resetu rate limitu"""
        history = self.request_history[source]
        if not history:
            return 0.0

        oldest_request = min(history)
        return max(0.0, 60 - (current_time - oldest_request))

    async def record_request(self, source: str, user_id: str = None):
        """Zaznamenaj request"""
        current_time = time.time()
        self.request_history[source].append(current_time)

        if user_id:
            self.user_usage[user_id]["hourly"] += 1
            self.user_usage[user_id]["daily"] += 1

        # Omeź velikost historie
        if len(self.request_history[source]) > 100:
            self.request_history[source].popleft()

    async def record_response(self, source: str, response_time: float, success: bool):
        """Zaznamenaj response a aktualizuj circuit breaker"""
        # Zaznamenaj response time
        self.response_times[source].append(response_time)
        if len(self.response_times[source]) > 50:
            self.response_times[source].popleft()

        # Aktualizuj circuit breaker
        if success:
            self.success_counts[source] += 1
            self.circuit_failure_counts[source] = 0

            # Přechod z HALF_OPEN do CLOSED
            if self.circuit_states[source] == CircuitState.HALF_OPEN:
                self.circuit_states[source] = CircuitState.CLOSED
                logger.info(f"Circuit breaker {source}: HALF_OPEN -> CLOSED")
        else:
            self.error_counts[source] += 1
            self.circuit_failure_counts[source] += 1

            # Přechod do OPEN stavu
            if self.circuit_failure_counts[source] >= self.failure_threshold:
                self.circuit_states[source] = CircuitState.OPEN
                self.circuit_last_failure[source] = time.time()
                logger.warning(f"Circuit breaker {source}: -> OPEN (failures: {self.circuit_failure_counts[source]})")

    async def get_statistics(self) -> Dict[str, Any]:
        """Vrátí statistiky rate limiteru"""
        stats = {}

        for source in self.source_limits.keys():
            if source == "default":
                continue

            current_time = time.time()
            minute_ago = current_time - 60

            recent_requests = sum(
                1 for req_time in self.request_history[source]
                if req_time > minute_ago
            )

            avg_response_time = 0.0
            if self.response_times[source]:
                avg_response_time = statistics.mean(list(self.response_times[source])[-10:])

            stats[source] = {
                "requests_last_minute": recent_requests,
                "circuit_state": self.circuit_states[source].value,
                "success_count": self.success_counts[source],
                "error_count": self.error_counts[source],
                "avg_response_time": round(avg_response_time, 2),
                "remaining_requests": self._get_remaining_requests(source, current_time)
            }

        return {
            "sources": stats,
            "total_users": len(self.user_usage),
            "active_circuits": len([s for s in self.circuit_states.values() if s != CircuitState.CLOSED])
        }

    def set_user_quota(self, user_id: str, daily: int, hourly: int):
        """Nastaví kvótu pro uživatele"""
        self.user_quotas[user_id] = {"daily": daily, "hourly": hourly}

    def update_source_limits(self, source: str, requests_per_minute: int, burst: int):
        """Aktualizuje limity pro zdroj"""
        self.source_limits[source] = {
            "requests_per_minute": requests_per_minute,
            "burst": burst
        }

# Globální instance
rate_limiter = AdaptiveRateLimiter()

# Decorator pro automatické rate limiting
def rate_limited(source: str):
    """Decorator pro automatické rate limiting"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            user_id = kwargs.get('user_id')

            # Zkontroluj rate limit
            check = await rate_limiter.check_rate_limit(source, user_id)
            if not check["allowed"]:
                raise Exception(f"Rate limit exceeded: {check}")

            # Adaptivní delay
            if check.get("adaptive_delay", 0) > 0:
                await asyncio.sleep(check["adaptive_delay"])

            # Zaznamenaj request
            await rate_limiter.record_request(source, user_id)

            # Spusť funkci a změř čas
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                response_time = time.time() - start_time
                await rate_limiter.record_response(source, response_time, True)
                return result
            except Exception as e:
                response_time = time.time() - start_time
                await rate_limiter.record_response(source, response_time, False)
                raise

        return wrapper
    return decorator
