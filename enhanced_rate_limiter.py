"""
Intelligent Rate Limiter s adaptivním throttlingem pro academic APIs
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import redis
import json
import logging

@dataclass
class RateLimitConfig:
    """Konfigurace rate limitingu pro různé zdroje"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    adaptive_scaling: bool = True
    priority_multiplier: float = 1.0

@dataclass
class UserQuota:
    """Uživatelské kvóty a limity"""
    daily_limit: int = 1000
    hourly_limit: int = 100
    priority_level: int = 1  # 1=basic, 2=premium, 3=enterprise
    used_today: int = 0
    used_this_hour: int = 0
    last_reset: datetime = field(default_factory=datetime.now)

class IntelligentRateLimiter:
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        self.local_cache = defaultdict(dict)

        # Konfigurace pro různé academic APIs
        self.api_configs = {
            'pubmed': RateLimitConfig(requests_per_minute=10, requests_per_hour=200),
            'openalex': RateLimitConfig(requests_per_minute=100, requests_per_hour=2000),
            'wikipedia': RateLimitConfig(requests_per_minute=200, requests_per_hour=5000),
            'semantic_scholar': RateLimitConfig(requests_per_minute=20, requests_per_hour=500),
            'crossref': RateLimitConfig(requests_per_minute=50, requests_per_hour=1000),
            'arxiv': RateLimitConfig(requests_per_minute=30, requests_per_hour=600),
        }

        # Circuit breaker states
        self.circuit_breakers = defaultdict(lambda: {
            'failures': 0,
            'last_failure': None,
            'state': 'closed'  # closed, open, half-open
        })

        self.response_times = defaultdict(deque)  # Pro adaptivní throttling

    async def check_rate_limit(self, user_id: str, api_source: str,
                             priority_level: int = 1) -> Dict[str, Any]:
        """Zkontroluje rate limit pro uživatele a API zdroj"""

        # Získej konfiguraci pro API
        config = self.api_configs.get(api_source, RateLimitConfig())

        # Kontrola circuit breaker
        if self._is_circuit_open(api_source):
            return {
                'allowed': False,
                'reason': 'circuit_breaker_open',
                'retry_after': 60
            }

        # Kontrola uživatelských kvót
        user_quota = await self._get_user_quota(user_id)
        if not self._check_user_quota(user_quota, priority_level):
            return {
                'allowed': False,
                'reason': 'user_quota_exceeded',
                'quota_resets': self._get_quota_reset_time(user_quota)
            }

        # Kontrola API limitů
        current_minute = int(time.time() // 60)
        current_hour = int(time.time() // 3600)

        # Redis klíče
        minute_key = f"rl:{api_source}:minute:{current_minute}"
        hour_key = f"rl:{api_source}:hour:{current_hour}"
        user_minute_key = f"rl:user:{user_id}:{api_source}:minute:{current_minute}"

        # Zkontroluj limity v Redis
        pipe = self.redis_client.pipeline()
        pipe.get(minute_key)
        pipe.get(hour_key)
        pipe.get(user_minute_key)
        results = pipe.execute()

        minute_count = int(results[0] or 0)
        hour_count = int(results[1] or 0)
        user_minute_count = int(results[2] or 0)

        # Aplikuj priority multiplier
        effective_minute_limit = int(config.requests_per_minute * config.priority_multiplier * priority_level)
        effective_hour_limit = int(config.requests_per_hour * config.priority_multiplier * priority_level)

        # Kontrola limitů
        if minute_count >= effective_minute_limit:
            return {
                'allowed': False,
                'reason': 'minute_limit_exceeded',
                'retry_after': 60 - (time.time() % 60)
            }

        if hour_count >= effective_hour_limit:
            return {
                'allowed': False,
                'reason': 'hour_limit_exceeded',
                'retry_after': 3600 - (time.time() % 3600)
            }

        # Burst protection
        if user_minute_count >= config.burst_limit:
            return {
                'allowed': False,
                'reason': 'burst_limit_exceeded',
                'retry_after': 60 - (time.time() % 60)
            }

        return {
            'allowed': True,
            'remaining_minute': effective_minute_limit - minute_count,
            'remaining_hour': effective_hour_limit - hour_count
        }

    async def record_request(self, user_id: str, api_source: str,
                           response_time: float = None, success: bool = True):
        """Zaznamenej request pro rate limiting a monitoring"""

        current_minute = int(time.time() // 60)
        current_hour = int(time.time() // 3600)

        # Redis klíče
        minute_key = f"rl:{api_source}:minute:{current_minute}"
        hour_key = f"rl:{api_source}:hour:{current_hour}"
        user_minute_key = f"rl:user:{user_id}:{api_source}:minute:{current_minute}"

        # Increment counters s expirací
        pipe = self.redis_client.pipeline()
        pipe.incr(minute_key)
        pipe.expire(minute_key, 120)  # 2 minuty buffer
        pipe.incr(hour_key)
        pipe.expire(hour_key, 7200)  # 2 hodiny buffer
        pipe.incr(user_minute_key)
        pipe.expire(user_minute_key, 120)
        pipe.execute()

        # Aktualizuj user quota
        await self._update_user_quota(user_id)

        # Record response time pro adaptivní throttling
        if response_time is not None:
            self.response_times[api_source].append(response_time)
            if len(self.response_times[api_source]) > 100:
                self.response_times[api_source].popleft()

        # Circuit breaker logic
        if not success:
            self.circuit_breakers[api_source]['failures'] += 1
            self.circuit_breakers[api_source]['last_failure'] = time.time()

            if self.circuit_breakers[api_source]['failures'] >= 5:
                self.circuit_breakers[api_source]['state'] = 'open'
        else:
            # Reset failures on success
            self.circuit_breakers[api_source]['failures'] = 0
            if self.circuit_breakers[api_source]['state'] == 'half-open':
                self.circuit_breakers[api_source]['state'] = 'closed'

    def get_adaptive_delay(self, api_source: str) -> float:
        """Získej adaptivní delay na základě response times"""
        if api_source not in self.response_times:
            return 0.0

        times = list(self.response_times[api_source])
        if not times:
            return 0.0

        # Pokud jsou response times vysoké, přidej delay
        avg_response = sum(times) / len(times)
        if avg_response > 2.0:  # Více než 2 sekundy
            return min(avg_response * 0.5, 5.0)  # Max 5 sekund delay

        return 0.0

    def _is_circuit_open(self, api_source: str) -> bool:
        """Zkontroluj jestli je circuit breaker otevřený"""
        breaker = self.circuit_breakers[api_source]

        if breaker['state'] == 'closed':
            return False

        if breaker['state'] == 'open':
            # Zkus half-open po 60 sekundách
            if time.time() - breaker['last_failure'] > 60:
                breaker['state'] = 'half-open'
                return False
            return True

        return False  # half-open state

    async def _get_user_quota(self, user_id: str) -> UserQuota:
        """Získej uživatelské kvóty z Redis/cache"""
        quota_key = f"quota:user:{user_id}"

        try:
            quota_data = self.redis_client.get(quota_key)
            if quota_data:
                data = json.loads(quota_data)
                return UserQuota(**data)
        except:
            pass

        # Default quota pro nové uživatele
        return UserQuota()

    async def _update_user_quota(self, user_id: str):
        """Aktualizuj uživatelské kvóty"""
        quota = await self._get_user_quota(user_id)

        now = datetime.now()

        # Reset daily counter
        if (now - quota.last_reset).days >= 1:
            quota.used_today = 0
            quota.used_this_hour = 0
            quota.last_reset = now

        # Reset hourly counter
        elif (now - quota.last_reset).total_seconds() >= 3600:
            quota.used_this_hour = 0

        quota.used_today += 1
        quota.used_this_hour += 1

        # Ulož do Redis
        quota_key = f"quota:user:{user_id}"
        quota_data = {
            'daily_limit': quota.daily_limit,
            'hourly_limit': quota.hourly_limit,
            'priority_level': quota.priority_level,
            'used_today': quota.used_today,
            'used_this_hour': quota.used_this_hour,
            'last_reset': quota.last_reset.isoformat()
        }

        self.redis_client.setex(quota_key, 86400, json.dumps(quota_data))

    def _check_user_quota(self, quota: UserQuota, priority_level: int) -> bool:
        """Zkontroluj jestli uživatel nepřekročil kvóty"""
        effective_daily = quota.daily_limit * priority_level
        effective_hourly = quota.hourly_limit * priority_level

        return (quota.used_today < effective_daily and
                quota.used_this_hour < effective_hourly)

    def _get_quota_reset_time(self, quota: UserQuota) -> Dict[str, str]:
        """Získej časy resetu kvót"""
        now = datetime.now()
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        next_day = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

        return {
            'hourly_reset': next_hour.isoformat(),
            'daily_reset': next_day.isoformat()
        }

# Global instance
intelligent_limiter = IntelligentRateLimiter()
