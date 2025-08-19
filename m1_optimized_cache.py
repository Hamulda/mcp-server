Memory Optimized Cache Manager pro MacBook Air M1
Inteligentní správa paměti s ohledem na omezené zdroje
"""

import time
import threading
import hashlib
import json
import psutil
from typing import Any, Optional, Dict, List
from collections import OrderedDict
import pickle
import os
from pathlib import Path

class M1OptimizedCache:
    """Cache optimalizovaný pro Apple Silicon M1 s omezenou RAM"""

    def __init__(self, max_memory_mb: int = 256):
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self.max_memory_mb = max_memory_mb
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._current_size_bytes = 0
        self._hits = 0
        self._misses = 0

        # M1 optimalizace
        self._compression_enabled = True
        self._auto_cleanup_threshold = 0.8  # Cleanup při 80% capacity

    def _get_system_memory_pressure(self) -> float:
        """Vrátí memory pressure (0.0 - 1.0)"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except:
            return 0.5  # Conservative fallback

    def _should_aggressive_cleanup(self) -> bool:
        """Rozhoduje o agresivním čištění cache"""
        memory_pressure = self._get_system_memory_pressure()
        cache_pressure = self._current_size_bytes / self.max_memory_bytes

        # Agresivní cleanup pokud:
        # - System memory > 85% NEBO
        # - Cache memory > 80% AND system memory > 70%
        return memory_pressure > 0.85 or (cache_pressure > 0.8 and memory_pressure > 0.7)

    def _estimate_object_size(self, obj: Any) -> int:
        """Rychlý odhad velikosti objektu"""
        try:
            if self._compression_enabled:
                # Simplified estimation pro rychlost
                return len(str(obj)) * 0.7  # Předpoklad 30% komprese
            else:
                return len(pickle.dumps(obj))
        except:
            return len(str(obj))  # Fallback

    def _cleanup_if_needed(self):
        """Inteligentní cleanup s ohledem na M1 memory management"""
        if self._should_aggressive_cleanup():
            # Agresivní cleanup - odstraň 40% nejstarších entries
            items_to_remove = max(1, len(self._cache) // 2.5)
            for _ in range(int(items_to_remove)):
                if self._cache:
                    key, (_, size, _) = self._cache.popitem(last=False)
                    self._current_size_bytes -= size
        elif self._current_size_bytes > self.max_memory_bytes * self._auto_cleanup_threshold:
            # Normální cleanup - odstraň 20% nejstarších entries
            items_to_remove = max(1, len(self._cache) // 5)
            for _ in range(items_to_remove):
                if self._cache:
                    key, (_, size, _) = self._cache.popitem(last=False)
                    self._current_size_bytes -= size

    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Uloží hodnotu s M1 optimalizací"""
        with self._lock:
            current_time = time.time()
            expire_at = current_time + ttl_seconds
            estimated_size = self._estimate_object_size(value)

            # Aggressive pre-cleanup pokud by nový item byl příliš velký
            if estimated_size > self.max_memory_bytes * 0.5:
                # Item je větší než 50% cache - odmítni
                return False

            # Cleanup před přidáním
            self._cleanup_if_needed()

            # Přidej/aktualizuj item
            if key in self._cache:
                old_size = self._cache[key][1]
                self._current_size_bytes -= old_size

            self._cache[key] = (value, estimated_size, expire_at)
            self._current_size_bytes += estimated_size

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            return True

    def get(self, key: str) -> Optional[Any]:
        """Získá hodnotu s TTL kontrolou"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, size, expire_at = self._cache[key]
            current_time = time.time()

            if current_time > expire_at:
                # Expired - odstraň
                del self._cache[key]
                self._current_size_bytes -= size
                self._misses += 1
                return None

            # Hit - move to end
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def clear(self):
        """Vyčistí celý cache"""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Vrátí detailní statistiky"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                'entries': len(self._cache),
                'size_mb': self._current_size_bytes / (1024 * 1024),
                'max_size_mb': self.max_memory_mb,
                'utilization_percent': (self._current_size_bytes / self.max_memory_bytes) * 100,
                'hit_rate_percent': hit_rate,
                'hits': self._hits,
                'misses': self._misses,
                'system_memory_pressure': self._get_system_memory_pressure() * 100,
                'compression_enabled': self._compression_enabled
            }

class PrivateCacheManager:
    """Cache manager optimalizovaný pro soukromé použití na M1"""

    def __init__(self, memory_limit_mb: int = 256, persistent_cache: bool = True):
        # Memory cache
        self.memory_cache = M1OptimizedCache(memory_limit_mb)

        # Persistent cache pro offline použití
        self.persistent_enabled = persistent_cache
        if persistent_cache:
            self.cache_dir = Path.home() / ".research_tool_cache"
            self.cache_dir.mkdir(exist_ok=True)

        self._stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'total_misses': 0
        }

    def _generate_cache_key(self, query: str, domain: str, strategy: str) -> str:
        """Generuje cache klíč"""
        key_string = f"{query.lower().strip()}|{domain}|{strategy}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def _get_disk_path(self, cache_key: str) -> Path:
        """Vrátí cestu k souboru na disku"""
        return self.cache_dir / f"{cache_key}.cache"

    async def get(self, query: str, domain: str, strategy: str) -> Optional[Any]:
        """Získá z cache s fallback strategií"""
        cache_key = self._generate_cache_key(query, domain, strategy)

        # 1. Zkus memory cache
        result = self.memory_cache.get(cache_key)
        if result is not None:
            self._stats['memory_hits'] += 1
            return result

        # 2. Zkus persistent cache
        if self.persistent_enabled:
            try:
                disk_path = self._get_disk_path(cache_key)
                if disk_path.exists():
                    with open(disk_path, 'rb') as f:
                        data = pickle.load(f)

                    # Zkontroluj TTL
                    if time.time() < data.get('expire_at', 0):
                        result = data.get('value')
                        # Promote do memory cache
                        self.memory_cache.set(cache_key, result, 1800)  # 30 min
                        self._stats['disk_hits'] += 1
                        return result
                    else:
                        # Expired - smaž
                        disk_path.unlink()
            except Exception:
                pass  # Ignore disk cache errors

        # Cache miss
        self._stats['total_misses'] += 1
        return None

    async def set(self, query: str, domain: str, strategy: str, value: Any, ttl_seconds: int = 3600):
        """Uloží do cache"""
        cache_key = self._generate_cache_key(query, domain, strategy)

        # 1. Memory cache
        self.memory_cache.set(cache_key, value, ttl_seconds)

        # 2. Persistent cache (pro velké objekty)
        if self.persistent_enabled:
            try:
                disk_path = self._get_disk_path(cache_key)
                cache_data = {
                    'value': value,
                    'expire_at': time.time() + ttl_seconds,
                    'created_at': time.time()
                }

                with open(disk_path, 'wb') as f:
                    pickle.dump(cache_data, f)
            except Exception:
                pass  # Ignore disk cache errors

    def clear_all(self):
        """Vyčistí všechny cache"""
        self.memory_cache.clear()

        if self.persistent_enabled and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                except:
                    pass

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Vrátí kompletní statistiky"""
        memory_stats = self.memory_cache.get_stats()

        disk_stats = {}
        if self.persistent_enabled and self.cache_dir.exists():
            cache_files = list(self.cache_dir.glob("*.cache"))
            total_disk_size = sum(f.stat().st_size for f in cache_files if f.exists())
            disk_stats = {
                'files_count': len(cache_files),
                'total_size_mb': total_disk_size / (1024 * 1024),
                'cache_dir': str(self.cache_dir)
            }

        total_hits = self._stats['memory_hits'] + self._stats['disk_hits']
        total_requests = total_hits + self._stats['total_misses']
        overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'memory_cache': memory_stats,
            'disk_cache': disk_stats,
            'overall_stats': {
                'hit_rate_percent': overall_hit_rate,
                'memory_hits': self._stats['memory_hits'],
                'disk_hits': self._stats['disk_hits'],
                'total_misses': self._stats['total_misses'],
                'total_requests': total_requests
            },
            'system_info': {
                'platform': 'macOS_M1',
                'persistent_cache_enabled': self.persistent_enabled
            }
        }

# Factory pro private použití
def create_private_cache_manager(memory_limit_mb: int = 256) -> PrivateCacheManager:
    """Vytvoří cache manager optimalizovaný pro MacBook Air M1"""
    return PrivateCacheManager(memory_limit_mb=memory_limit_mb, persistent_cache=True)
