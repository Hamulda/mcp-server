"""
Unified Cache System - Konsolidovaný cache systém pro M1 MacBook
Sjednocuje cache_manager.py a m1_optimized_cache.py do jednoho optimalizovaného systému
"""

import asyncio
import time
import threading
import hashlib
import json
import gc
import psutil
import pickle
import sqlite3
from typing import Any, Optional, Dict, List, Union
from collections import OrderedDict
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

@dataclass
class CacheEntry:
    """Optimalizovaná cache entry s metadaty"""
    key: str
    data: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: int
    size_bytes: int
    compressed: bool = False

class UnifiedCacheSystem:
    """
    Sjednocený cache systém optimalizovaný pro M1 MacBook
    Kombinuje in-memory LRU cache s persistentním SQLite storage
    """

    def __init__(
        self,
        max_memory_items: int = 1000,
        memory_threshold_gb: float = 1.5,
        cache_dir: Path = None,
        default_ttl: int = 3600,
        enable_persistence: bool = True
    ):
        # Core configuration
        self.max_memory_items = max_memory_items
        self.memory_threshold_gb = memory_threshold_gb
        self.default_ttl = default_ttl
        self.enable_persistence = enable_persistence

        # Cache storage
        self._memory_cache = OrderedDict()
        self._cache_metadata = {}
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memory_cleanups": 0,
            "persistence_saves": 0,
            "persistence_loads": 0
        }

        # Setup cache directory and persistence
        self.cache_dir = cache_dir or Path("cache/unified")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if enable_persistence:
            self._init_persistence()

        # Background maintenance
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes

    def _init_persistence(self):
        """Initialize SQLite persistence layer"""
        self.db_path = self.cache_dir / "cache.db"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER,
                    ttl INTEGER,
                    size_bytes INTEGER,
                    compressed BOOLEAN
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed 
                ON cache_entries(last_accessed)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON cache_entries(created_at)
            """)

    def _generate_cache_key(self, key_data: Union[str, Dict, List]) -> str:
        """Generate consistent cache key from various data types"""
        if isinstance(key_data, str):
            content = key_data
        else:
            content = json.dumps(key_data, sort_keys=True, default=str)

        return hashlib.md5(content.encode()).hexdigest()

    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes"""
        try:
            return len(pickle.dumps(data))
        except:
            return len(str(data).encode())

    def _check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            return available_gb < self.memory_threshold_gb
        except:
            return False

    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage"""
        import gzip
        pickled = pickle.dumps(data)
        return gzip.compress(pickled)

    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data from storage"""
        import gzip
        pickled = gzip.decompress(compressed_data)
        return pickle.loads(pickled)

    async def get(self, key: Union[str, Dict, List], default: Any = None) -> Any:
        """
        Get item from cache with intelligent fallback to persistence
        """
        cache_key = self._generate_cache_key(key)
        current_time = time.time()

        with self._lock:
            # Check memory cache first
            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]

                # Check TTL
                if current_time - entry.created_at > entry.ttl:
                    del self._memory_cache[cache_key]
                    if cache_key in self._cache_metadata:
                        del self._cache_metadata[cache_key]
                    self.stats["misses"] += 1
                else:
                    # Update access statistics
                    entry.last_accessed = current_time
                    entry.access_count += 1

                    # Move to end (most recently used)
                    self._memory_cache.move_to_end(cache_key)

                    self.stats["hits"] += 1
                    return entry.data

            # Try persistence layer
            if self.enable_persistence:
                persistent_data = await self._get_from_persistence(cache_key, current_time)
                if persistent_data is not None:
                    # Load back to memory if there's space
                    if len(self._memory_cache) < self.max_memory_items:
                        await self._store_in_memory(cache_key, persistent_data, current_time)

                    self.stats["hits"] += 1
                    self.stats["persistence_loads"] += 1
                    return persistent_data

            self.stats["misses"] += 1
            return default

    async def set(
        self,
        key: Union[str, Dict, List],
        data: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set item in cache with intelligent storage strategy
        """
        cache_key = self._generate_cache_key(key)
        current_time = time.time()
        actual_ttl = ttl or self.default_ttl

        # Calculate data size
        data_size = self._calculate_size(data)

        with self._lock:
            # Memory pressure check
            if self._check_memory_pressure():
                await self._emergency_cleanup()

            # Store in memory cache
            await self._store_in_memory(cache_key, data, current_time, actual_ttl, data_size)

            # Store in persistence if enabled
            if self.enable_persistence:
                await self._store_in_persistence(cache_key, data, current_time, actual_ttl, data_size)

            # Background maintenance
            if current_time - self._last_cleanup > self._cleanup_interval:
                asyncio.create_task(self._background_cleanup())
                self._last_cleanup = current_time

        return True

    async def _store_in_memory(
        self,
        cache_key: str,
        data: Any,
        current_time: float,
        ttl: int = None,
        data_size: int = None
    ):
        """Store item in memory cache with LRU eviction"""
        if ttl is None:
            ttl = self.default_ttl
        if data_size is None:
            data_size = self._calculate_size(data)

        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            data=data,
            created_at=current_time,
            last_accessed=current_time,
            access_count=1,
            ttl=ttl,
            size_bytes=data_size
        )

        # Handle capacity
        while len(self._memory_cache) >= self.max_memory_items:
            # Remove least recently used
            oldest_key, oldest_entry = self._memory_cache.popitem(last=False)
            if oldest_key in self._cache_metadata:
                del self._cache_metadata[oldest_key]
            self.stats["evictions"] += 1

        # Store entry
        self._memory_cache[cache_key] = entry
        self._cache_metadata[cache_key] = {
            "created_at": current_time,
            "size_bytes": data_size,
            "ttl": ttl
        }

    async def _store_in_persistence(
        self,
        cache_key: str,
        data: Any,
        current_time: float,
        ttl: int,
        data_size: int
    ):
        """Store item in SQLite persistence layer"""
        try:
            # Compress large data
            should_compress = data_size > 1024  # Compress if > 1KB

            if should_compress:
                stored_data = self._compress_data(data)
            else:
                stored_data = pickle.dumps(data)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, data, created_at, last_accessed, access_count, ttl, size_bytes, compressed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache_key, stored_data, current_time, current_time,
                    1, ttl, data_size, should_compress
                ))

            self.stats["persistence_saves"] += 1

        except Exception as e:
            # Persistence failure shouldn't break caching
            pass

    async def _get_from_persistence(self, cache_key: str, current_time: float) -> Any:
        """Retrieve item from SQLite persistence layer"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT data, created_at, ttl, compressed 
                    FROM cache_entries 
                    WHERE key = ?
                """, (cache_key,))

                row = cursor.fetchone()
                if row is None:
                    return None

                stored_data, created_at, ttl, compressed = row

                # Check TTL
                if current_time - created_at > ttl:
                    # Delete expired entry
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (cache_key,))
                    return None

                # Update access time
                conn.execute("""
                    UPDATE cache_entries 
                    SET last_accessed = ?, access_count = access_count + 1 
                    WHERE key = ?
                """, (current_time, cache_key))

                # Decompress if needed
                if compressed:
                    return self._decompress_data(stored_data)
                else:
                    return pickle.loads(stored_data)

        except Exception as e:
            return None

    async def delete(self, key: Union[str, Dict, List]) -> bool:
        """Delete item from both memory and persistence"""
        cache_key = self._generate_cache_key(key)

        with self._lock:
            # Remove from memory
            deleted_memory = cache_key in self._memory_cache
            if deleted_memory:
                del self._memory_cache[cache_key]
                if cache_key in self._cache_metadata:
                    del self._cache_metadata[cache_key]

            # Remove from persistence
            deleted_persistence = False
            if self.enable_persistence:
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.execute(
                            "DELETE FROM cache_entries WHERE key = ?",
                            (cache_key,)
                        )
                        deleted_persistence = cursor.rowcount > 0
                except:
                    pass

            return deleted_memory or deleted_persistence

    async def clear(self) -> int:
        """Clear all cache data"""
        with self._lock:
            count = len(self._memory_cache)
            self._memory_cache.clear()
            self._cache_metadata.clear()

            if self.enable_persistence:
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                        count += cursor.fetchone()[0]
                        conn.execute("DELETE FROM cache_entries")
                except:
                    pass

            # Reset statistics
            self.stats = {key: 0 for key in self.stats}

            return count

    async def _emergency_cleanup(self):
        """Emergency cleanup during memory pressure"""
        current_time = time.time()

        # Remove expired entries first
        expired_keys = []
        for key, entry in self._memory_cache.items():
            if current_time - entry.created_at > entry.ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self._memory_cache[key]
            if key in self._cache_metadata:
                del self._cache_metadata[key]

        # If still over capacity, remove least recently used
        target_size = max(self.max_memory_items // 2, 100)  # Remove half
        while len(self._memory_cache) > target_size:
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
            if oldest_key in self._cache_metadata:
                del self._cache_metadata[oldest_key]
            self.stats["evictions"] += 1

        # Force garbage collection
        gc.collect()
        self.stats["memory_cleanups"] += 1

    async def _background_cleanup(self):
        """Background cleanup of expired entries"""
        current_time = time.time()

        # Clean memory cache
        expired_keys = []
        with self._lock:
            for key, entry in list(self._memory_cache.items()):
                if current_time - entry.created_at > entry.ttl:
                    expired_keys.append(key)

        for key in expired_keys:
            await self.delete(key)

        # Clean persistence layer
        if self.enable_persistence:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        DELETE FROM cache_entries 
                        WHERE ? - created_at > ttl
                    """, (current_time,))
            except:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0

        memory_stats = {
            "items_in_memory": len(self._memory_cache),
            "memory_capacity": self.max_memory_items,
            "memory_usage_percent": len(self._memory_cache) / self.max_memory_items * 100
        }

        # Get persistence stats
        persistence_stats = {"items_in_persistence": 0}
        if self.enable_persistence:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                    persistence_stats["items_in_persistence"] = cursor.fetchone()[0]
            except:
                pass

        return {
            **self.stats,
            "hit_rate_percent": hit_rate,
            "total_requests": total_requests,
            **memory_stats,
            **persistence_stats,
            "system_memory_gb": psutil.virtual_memory().available / (1024**3)
        }

# Global cache instance
_global_cache = None

def get_unified_cache(
    max_memory_items: int = 1000,
    memory_threshold_gb: float = 1.5,
    cache_dir: Path = None,
    default_ttl: int = 3600,
    enable_persistence: bool = True
) -> UnifiedCacheSystem:
    """Get global unified cache instance (singleton pattern)"""
    global _global_cache

    if _global_cache is None:
        _global_cache = UnifiedCacheSystem(
            max_memory_items=max_memory_items,
            memory_threshold_gb=memory_threshold_gb,
            cache_dir=cache_dir,
            default_ttl=default_ttl,
            enable_persistence=enable_persistence
        )

    return _global_cache

# Backward compatibility
def get_cache_manager():
    """Backward compatibility with old cache_manager interface"""
    return get_unified_cache()

# Export
__all__ = [
    'UnifiedCacheSystem',
    'CacheEntry',
    'get_unified_cache',
    'get_cache_manager'
]
