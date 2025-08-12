import time
import threading
from typing import Any, Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class InMemoryCache:
    def __init__(self, default_ttl: int = 600):
        self._store = {}
        self._lock = threading.Lock()
        self.default_ttl = default_ttl

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        expire_at = time.time() + (ttl or self.default_ttl)
        with self._lock:
            self._store[key] = (value, expire_at)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            value, expire_at = item
            if time.time() > expire_at:
                del self._store[key]
                return None
            return value

    def delete(self, key: str):
        with self._lock:
            if key in self._store:
                del self._store[key]

    def clear(self):
        with self._lock:
            self._store.clear()

class RedisCache:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, ttl: int = 86400):
        self.ttl = ttl
        self.client = redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)

    def set(self, key: str, value: Any, ttl: int = None) -> None:
        import json
        self.client.setex(key, ttl or self.ttl, json.dumps(value))

    def get(self, key: str) -> Optional[Any]:
        import json
        val = self.client.get(key)
        if val is not None:
            return json.loads(val)
        return None

    def delete(self, key: str) -> None:
        self.client.delete(key)

    def clear(self) -> None:
        self.client.flushdb()

class CacheManager:
    def __init__(self, backend: Any = None, redis_config: dict = None):
        self.hits = 0
        self.misses = 0
        if REDIS_AVAILABLE and redis_config:
            self.backend = RedisCache(**redis_config)
        else:
            self.backend = backend or InMemoryCache()

    def set(self, key: str, value: Any, ttl: int = None) -> None:
        self.backend.set(key, value, ttl)

    def get(self, key: str) -> Optional[Any]:
        val = self.backend.get(key)
        if val is not None:
            self.hits += 1
        else:
            self.misses += 1
        return val

    def delete(self, key: str) -> None:
        self.backend.delete(key)

    def clear(self) -> None:
        self.backend.clear()

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total if total > 0 else 0.0
        }
