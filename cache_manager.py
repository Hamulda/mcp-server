M1 Optimized Cache Manager - Maximálně efektivní caching pro MacBook Air M1
Prioritizuje nízkou spotřebu paměti a rychlý přístup
"""

import time
import threading
import hashlib
import json
import gc
import psutil
from typing import Any, Optional, Dict, List
from collections import OrderedDict
import pickle
import os
from pathlib import Path

class M1OptimizedLRUCache:
    """LRU cache optimalizovaný pro M1 s aktivním memory managementem"""

    def __init__(self, max_size: int = 500, default_ttl: int = 3600):
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

        # M1 specifické optimalizace
        self.memory_threshold = 1.5  # GB - spustí cleanup při nedostatku paměti
        self.auto_gc_enabled = True
        self.compression_enabled = True

    def _check_memory_pressure(self) -> bool:
        """Kontroluje memory pressure a spustí cleanup pokud je potřeba"""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)

            if available_gb < self.memory_threshold:
                # Emergency cleanup
                self._emergency_cleanup()
                return True
        except:
            pass
        return False

    def _emergency_cleanup(self):
        """Emergency cleanup při nízkém RAM"""
        with self._lock:
            # Odstraň 50% nejstarších entries
            items_to_remove = len(self._cache) // 2
            for _ in range(items_to_remove):
                if self._cache:
                    self._cache.popitem(last=False)

            # Force garbage collection
            if self.auto_gc_enabled:
                gc.collect()

    def _evict_expired(self):
        """Odstraní expirované entries s optimalizací"""
        current_time = time.time()
        expired_keys = []

        # Optimalizace: kontroluj max 100 items najednou
        check_count = 0
        for key, (value, expire_at, access_count) in self._cache.items():
            if current_time > expire_at:
                expired_keys.append(key)
            check_count += 1
            if check_count >= 100:  # Limit pro performance
                break

        for key in expired_keys:
            del self._cache[key]

    def _evict_lru(self):
        """Odstraní nejméně používané entries s memory check"""
        self._check_memory_pressure()

        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Uloží hodnotu s M1 optimalizacemi"""
        with self._lock:
            self._evict_expired()
            expire_at = time.time() + (ttl or self.default_ttl)

            # Komprese pro větší objekty
            if self.compression_enabled and self._should_compress(value):
                value = self._compress_value(value)

            if key in self._cache:
                # Update existing
                _, _, access_count = self._cache[key]
                self._cache[key] = (value, expire_at, access_count + 1)
                self._cache.move_to_end(key)
            else:
                # Add new
                self._cache[key] = (value, expire_at, 1)
                self._evict_lru()

    def get(self, key: str) -> Optional[Any]:
        """Získá hodnotu s tracking statistik"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, expire_at, access_count = self._cache[key]
            current_time = time.time()

            if current_time > expire_at:
                del self._cache[key]
                self._misses += 1
                return None

            # Update access info and move to end
            self._cache[key] = (value, expire_at, access_count + 1)
            self._cache.move_to_end(key)
            self._hits += 1

            # Dekomprese pokud je potřeba
            if self.compression_enabled and self._is_compressed(value):
                value = self._decompress_value(value)

            return value

    def _should_compress(self, value: Any) -> bool:
        """Rozhodne, zda komprimovat hodnotu"""
        try:
            # Komprimuj objekty větší než 1KB
            serialized = pickle.dumps(value)
            return len(serialized) > 1024
        except:
            return False

    def _compress_value(self, value: Any) -> Dict:
        """Komprimuje hodnotu"""
        try:
            import gzip
            serialized = pickle.dumps(value)
            compressed = gzip.compress(serialized)
            return {'_compressed': True, 'data': compressed}
        except:
            return value

    def _is_compressed(self, value: Any) -> bool:
        """Kontroluje, zda je hodnota komprimovaná"""
        return isinstance(value, dict) and value.get('_compressed', False)

    def _decompress_value(self, value: Dict) -> Any:
        """Dekomprimuje hodnotu"""
        try:
            import gzip
            decompressed = gzip.decompress(value['data'])
            return pickle.loads(decompressed)
        except:
            return value

    def clear(self):
        """Vyčistí cache"""
        with self._lock:
            self._cache.clear()
            if self.auto_gc_enabled:
                gc.collect()

    def get_stats(self) -> Dict:
        """Vrátí statistiky cache"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': f"{hit_rate:.1f}%",
                'memory_usage_mb': self._estimate_memory_usage()
            }

    def _estimate_memory_usage(self) -> float:
        """Odhadne spotřebu paměti v MB"""
        try:
            total_size = 0
            for key, (value, _, _) in list(self._cache.items())[:10]:  # Sample prvních 10
                total_size += len(pickle.dumps((key, value)))

            avg_size = total_size / min(10, len(self._cache)) if self._cache else 0
            estimated_total = avg_size * len(self._cache)
            return estimated_total / (1024 * 1024)  # Convert to MB
        except:
            return 0.0

class M1OptimizedCacheManager:
    """Hlavní cache manager optimalizovaný pro M1"""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".research_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Memory cache optimalizovaný pro M1
        self.memory_cache = M1OptimizedLRUCache(max_size=300, default_ttl=1800)

        # Disk cache pro persistence
        self.disk_cache_enabled = True
        self.max_disk_cache_size_mb = 500  # 500MB limit

        # Statistics
        self.stats = {
            'memory_hits': 0,
            'memory_misses': 0,
            'disk_hits': 0,
            'disk_misses': 0
        }

    def _get_cache_key(self, key: str) -> str:
        """Vytvoří normalizovaný cache key"""
        return hashlib.md5(key.encode()).hexdigest()

    def _get_disk_path(self, cache_key: str) -> Path:
        """Vrátí cestu k disk cache souboru"""
        return self.cache_dir / f"{cache_key}.cache"

    def get(self, key: str) -> Optional[Any]:
        """Získá hodnotu z cache (memory -> disk)"""
        cache_key = self._get_cache_key(key)

        # 1. Zkus memory cache
        value = self.memory_cache.get(cache_key)
        if value is not None:
            self.stats['memory_hits'] += 1
            return value

        self.stats['memory_misses'] += 1

        # 2. Zkus disk cache
        if self.disk_cache_enabled:
            disk_value = self._get_from_disk(cache_key)
            if disk_value is not None:
                # Vrať do memory cache
                self.memory_cache.set(cache_key, disk_value)
                self.stats['disk_hits'] += 1
                return disk_value

        self.stats['disk_misses'] += 1
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Uloží hodnotu do cache"""
        cache_key = self._get_cache_key(key)

        # Ulož do memory cache
        self.memory_cache.set(cache_key, value, ttl)

        # Ulož na disk
        if self.disk_cache_enabled:
            self._save_to_disk(cache_key, value, ttl)

    def _get_from_disk(self, cache_key: str) -> Optional[Any]:
        """Načte hodnotu z disk cache"""
        try:
            cache_path = self._get_disk_path(cache_key)
            if not cache_path.exists():
                return None

            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            # Kontrola expiry
            if time.time() > data.get('expire_at', 0):
                cache_path.unlink(missing_ok=True)
                return None

            return data.get('value')

        except Exception:
            return None

    def _save_to_disk(self, cache_key: str, value: Any, ttl: Optional[int]):
        """Uloží hodnotu na disk"""
        try:
            # Kontrola velikosti disk cache
            self._cleanup_disk_cache_if_needed()

            cache_path = self._get_disk_path(cache_key)
            expire_at = time.time() + (ttl or 3600)

            data = {
                'value': value,
                'expire_at': expire_at,
                'created_at': time.time()
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)

        except Exception:
            pass  # Disk cache failures are non-critical

    def _cleanup_disk_cache_if_needed(self):
        """Vyčistí disk cache pokud je příliš velká"""
        try:
            total_size = sum(
                f.stat().st_size for f in self.cache_dir.glob("*.cache")
            )

            if total_size > self.max_disk_cache_size_mb * 1024 * 1024:
                # Vymaž nejstarší soubory
                cache_files = sorted(
                    self.cache_dir.glob("*.cache"),
                    key=lambda x: x.stat().st_mtime
                )

                # Vymaž nejstarší 30%
                files_to_remove = len(cache_files) // 3
                for cache_file in cache_files[:files_to_remove]:
                    cache_file.unlink(missing_ok=True)

        except Exception:
            pass

    def clear(self):
        """Vyčistí veškerou cache"""
        self.memory_cache.clear()

        if self.disk_cache_enabled:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink(missing_ok=True)

    def get_stats(self) -> Dict:
        """Vrátí kompletní statistiky"""
        memory_stats = self.memory_cache.get_stats()

        # Disk cache stats
        try:
            disk_files = list(self.cache_dir.glob("*.cache"))
            disk_size_mb = sum(f.stat().st_size for f in disk_files) / (1024 * 1024)
        except:
            disk_files = []
            disk_size_mb = 0

        return {
            **memory_stats,
            'disk_files': len(disk_files),
            'disk_size_mb': f"{disk_size_mb:.1f}",
            'total_memory_hits': self.stats['memory_hits'],
            'total_disk_hits': self.stats['disk_hits'],
            'total_misses': self.stats['memory_misses'] + self.stats['disk_misses']
        }

# Factory function
def create_m1_optimized_cache_manager() -> M1OptimizedCacheManager:
    """Vytvoří optimalizovaný cache manager pro M1"""
    return M1OptimizedCacheManager()

# Global instance
_cache_manager = None

def get_cache_manager() -> M1OptimizedCacheManager:
    """Získá globální cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = create_m1_optimized_cache_manager()
    return _cache_manager
