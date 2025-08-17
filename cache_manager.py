import time
import threading
import hashlib
import json
from typing import Any, Optional, Dict, List
from collections import OrderedDict
import pickle
import os
from pathlib import Path

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class LRUCache:
    """Optimalizovaný LRU cache s TTL a size limits"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def _evict_expired(self):
        """Odstraní expirované entries"""
        current_time = time.time()
        expired_keys = []

        for key, (value, expire_at, access_count) in self._cache.items():
            if current_time > expire_at:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]

    def _evict_lru(self):
        """Odstraní nejméně používané entries když překročíme limit"""
        while len(self._cache) >= self.max_size:
            # Remove least recently used item
            self._cache.popitem(last=False)

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        with self._lock:
            self._evict_expired()
            expire_at = time.time() + (ttl or self.default_ttl)

            if key in self._cache:
                # Update existing entry and move to end
                self._cache.move_to_end(key)
            else:
                # Add new entry
                self._evict_lru()

            self._cache[key] = (value, expire_at, 1)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            self._evict_expired()

            if key not in self._cache:
                self._misses += 1
                return None

            value, expire_at, access_count = self._cache[key]

            if time.time() > expire_at:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used) and increment access count
            self._cache.move_to_end(key)
            self._cache[key] = (value, expire_at, access_count + 1)
            self._hits += 1
            return value

    def delete(self, key: str):
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Vrátí cache statistiky"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': f"{hit_rate:.2f}%",
                'memory_usage': len(pickle.dumps(self._cache))
            }

class PersistentCache:
    """Persistent cache s disk storage"""

    def __init__(self, cache_dir: str = "cache", default_ttl: int = 86400):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = default_ttl
        self._index_file = self.cache_dir / "index.json"
        self._load_index()

    def _load_index(self):
        """Načte index souborů"""
        try:
            if self._index_file.exists():
                with open(self._index_file, 'r') as f:
                    self._index = json.load(f)
            else:
                self._index = {}
        except Exception:
            self._index = {}

    def _save_index(self):
        """Uloží index souborů"""
        try:
            with open(self._index_file, 'w') as f:
                json.dump(self._index, f, indent=2)
        except Exception:
            pass

    def _get_cache_path(self, key: str) -> Path:
        """Generuje cestu k cache souboru"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        try:
            expire_at = time.time() + (ttl or self.default_ttl)
            cache_path = self._get_cache_path(key)

            # Save data
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)

            # Update index
            self._index[key] = {
                'file': cache_path.name,
                'expire_at': expire_at,
                'created_at': time.time()
            }
            self._save_index()

        except Exception as e:
            print(f"Error saving to persistent cache: {e}")

    def get(self, key: str) -> Optional[Any]:
        try:
            if key not in self._index:
                return None

            entry = self._index[key]

            # Check expiration
            if time.time() > entry['expire_at']:
                self.delete(key)
                return None

            cache_path = self.cache_dir / entry['file']
            if not cache_path.exists():
                del self._index[key]
                self._save_index()
                return None

            # Load data
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        except Exception as e:
            print(f"Error loading from persistent cache: {e}")
            return None

    def delete(self, key: str):
        try:
            if key in self._index:
                entry = self._index[key]
                cache_path = self.cache_dir / entry['file']
                if cache_path.exists():
                    cache_path.unlink()
                del self._index[key]
                self._save_index()
        except Exception:
            pass

    def cleanup_expired(self):
        """Vyčistí expirované soubory"""
        current_time = time.time()
        expired_keys = []

        for key, entry in self._index.items():
            if current_time > entry['expire_at']:
                expired_keys.append(key)

        for key in expired_keys:
            self.delete(key)

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
        result = self.client.get(key)
        return json.loads(result) if result else None

    def delete(self, key: str):
        self.client.delete(key)

    def clear(self):
        self.client.flushdb()

class CacheManager:
    """Multi-layer cache manager s fallback strategií"""

    def __init__(self, use_redis: bool = False, use_persistent: bool = True,
                 memory_size: int = 1000, default_ttl: int = 3600):

        # Layer 1: In-memory LRU cache (fastest)
        self.memory_cache = LRUCache(max_size=memory_size, default_ttl=default_ttl)

        # Layer 2: Persistent disk cache
        self.persistent_cache = PersistentCache() if use_persistent else None

        # Layer 3: Redis cache (if available)
        self.redis_cache = None
        if use_redis and REDIS_AVAILABLE:
            try:
                self.redis_cache = RedisCache(ttl=default_ttl * 24)  # Redis cache longer
            except Exception:
                print("Redis not available, using fallback caches")

        self.stats_counters = {
            'memory_hits': 0,
            'persistent_hits': 0,
            'redis_hits': 0,
            'total_misses': 0
        }

    def _generate_key(self, key: str) -> str:
        """Normalize cache key"""
        if isinstance(key, str):
            return f"cache:{hashlib.md5(key.encode()).hexdigest()}"
        return f"cache:{hashlib.md5(str(key).encode()).hexdigest()}"

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in all available cache layers"""
        cache_key = self._generate_key(key)

        # Always set in memory cache
        self.memory_cache.set(cache_key, value, ttl)

        # Set in persistent cache if available
        if self.persistent_cache:
            self.persistent_cache.set(cache_key, value, ttl)

        # Set in Redis cache if available
        if self.redis_cache:
            try:
                self.redis_cache.set(cache_key, value, ttl)
            except Exception:
                pass  # Silently fail if Redis is down

    def get(self, key: str) -> Optional[Any]:
        """Get value with fallback strategy: Memory -> Persistent -> Redis"""
        cache_key = self._generate_key(key)

        # Try memory cache first (fastest)
        value = self.memory_cache.get(cache_key)
        if value is not None:
            self.stats_counters['memory_hits'] += 1
            return value

        # Try persistent cache
        if self.persistent_cache:
            value = self.persistent_cache.get(cache_key)
            if value is not None:
                # Promote to memory cache
                self.memory_cache.set(cache_key, value)
                self.stats_counters['persistent_hits'] += 1
                return value

        # Try Redis cache
        if self.redis_cache:
            try:
                value = self.redis_cache.get(cache_key)
                if value is not None:
                    # Promote to both memory and persistent cache
                    self.memory_cache.set(cache_key, value)
                    if self.persistent_cache:
                        self.persistent_cache.set(cache_key, value)
                    self.stats_counters['redis_hits'] += 1
                    return value
            except Exception:
                pass

        # Cache miss
        self.stats_counters['total_misses'] += 1
        return None

    def delete(self, key: str):
        """Delete from all cache layers"""
        cache_key = self._generate_key(key)

        self.memory_cache.delete(cache_key)

        if self.persistent_cache:
            self.persistent_cache.delete(cache_key)

        if self.redis_cache:
            try:
                self.redis_cache.delete(cache_key)
            except Exception:
                pass

    def clear(self):
        """Clear all cache layers"""
        self.memory_cache.clear()

        if self.persistent_cache:
            self.persistent_cache.cleanup_expired()

        if self.redis_cache:
            try:
                self.redis_cache.clear()
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        memory_stats = self.memory_cache.stats()

        total_hits = (self.stats_counters['memory_hits'] +
                     self.stats_counters['persistent_hits'] +
                     self.stats_counters['redis_hits'])
        total_requests = total_hits + self.stats_counters['total_misses']

        overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'memory_cache': memory_stats,
            'layer_stats': self.stats_counters,
            'overall_hit_rate': f"{overall_hit_rate:.2f}%",
            'total_requests': total_requests,
            'persistent_enabled': self.persistent_cache is not None,
            'redis_enabled': self.redis_cache is not None
        }
