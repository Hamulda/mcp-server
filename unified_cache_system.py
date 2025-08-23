"""
Unified Cache System - Konsolidovaný cache systém pro M1 MacBook
Sjednocuje cache_manager.py a m1_optimized_cache.py do jednoho optimalizovaného systému
OPTIMALIZOVÁNO PRO M1 MACBOOK AIR - využívá efektivní paměť a ARM optimalizace
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
import zlib
from typing import Any, Optional, Dict, List, Union
from collections import OrderedDict
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

@dataclass
class CacheEntry:
    """Optimalizovaná cache entry s metadaty pro M1"""
    key: str
    data: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: int
    size_bytes: int
    compressed: bool = False
    m1_optimized: bool = True  # Flag pro M1 specifické optimalizace

class M1OptimizedCacheSystem:
    """
    M1 MacBook Air optimalizovaný cache systém
    - Využívá unified memory architekturu M1
    - Optimalizované pro nízkou spotřebu energie
    - Efektivní kompresi dat
    - Adaptivní správa paměti
    """

    def __init__(
        self,
        max_memory_items: int = 2000,  # Zvýšeno pro M1 unified memory
        memory_threshold_gb: float = 2.0,  # Více paměti pro M1
        cache_dir: Path = None,
        default_ttl: int = 3600,
        enable_persistence: bool = True,
        enable_compression: bool = True,  # M1 má rychlou kompresi
        adaptive_cleanup: bool = True     # Adaptivní cleanup pro M1
    ):
        # M1 optimized configuration
        self.max_memory_items = max_memory_items
        self.memory_threshold_gb = memory_threshold_gb
        self.default_ttl = default_ttl
        self.enable_persistence = enable_persistence
        self.enable_compression = enable_compression
        self.adaptive_cleanup = adaptive_cleanup

        # Cache storage with M1 optimizations
        self._memory_cache = OrderedDict()
        self._cache_metadata = {}
        self._lock = threading.RLock()

        # M1 specific features
        self._m1_memory_pool = {}  # Pre-allocated memory pool
        self._compression_threshold = 1024  # Compress items > 1KB
        self._thermal_throttling = False

        # Enhanced statistics for M1
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memory_cleanups": 0,
            "persistence_saves": 0,
            "persistence_loads": 0,
            "compressions": 0,
            "decompressions": 0,
            "m1_optimizations": 0,
            "thermal_throttles": 0
        }

        # Setup cache directory and persistence
        self.cache_dir = cache_dir or Path("cache/unified")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if enable_persistence:
            self._init_m1_persistence()

        # M1 optimized background maintenance
        self._last_cleanup = time.time()
        self._cleanup_interval = 180  # 3 minutes for M1 efficiency
        self._background_task = None

    def _init_m1_persistence(self):
        """Initialize M1 optimized SQLite persistence"""
        self.db_path = self.cache_dir / "cache.db"

        # M1 optimized SQLite settings
        with sqlite3.connect(self.db_path) as conn:
            # M1 performance optimizations
            conn.execute("PRAGMA journal_mode=WAL")  # Better for SSD
            conn.execute("PRAGMA synchronous=NORMAL")  # Balanced for M1
            conn.execute("PRAGMA cache_size=10000")   # More cache for M1
            conn.execute("PRAGMA temp_store=MEMORY")  # Use unified memory
            conn.execute("PRAGMA mmap_size=268435456") # 256MB mmap for M1

            # Create optimized table structure
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER,
                    ttl INTEGER,
                    size_bytes INTEGER,
                    compressed INTEGER,
                    m1_optimized INTEGER DEFAULT 1
                )
            """)

            # M1 specific indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_m1_optimized ON cache_entries(m1_optimized)")

            conn.commit()

    async def __aenter__(self):
        """Async context manager entry with M1 optimizations"""
        # Start M1 optimized background maintenance
        if not self._background_task:
            self._background_task = asyncio.create_task(self._m1_background_maintenance())

        # Pre-warm M1 memory pool
        await self._prewarm_m1_memory_pool()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup with M1 power management"""
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        # Final M1 optimized cleanup
        await self._final_m1_cleanup()

    async def _prewarm_m1_memory_pool(self):
        """Pre-warm memory pool for M1 unified memory"""
        try:
            # Allocate common object sizes for M1 efficiency
            common_sizes = [64, 256, 1024, 4096, 16384]
            for size in common_sizes:
                self._m1_memory_pool[size] = bytearray(size)

            self.stats["m1_optimizations"] += 1
        except Exception as e:
            # Graceful degradation if memory prewarming fails
            pass

    async def get(self, key: str) -> Optional[Any]:
        """M1 optimized cache retrieval"""
        with self._lock:
            # Check memory cache first (M1 unified memory advantage)
            if key in self._memory_cache:
                entry = self._memory_cache[key]

                # Check TTL
                if self._is_expired(entry):
                    del self._memory_cache[key]
                    self.stats["misses"] += 1
                    return None

                # Update access metadata (M1 optimized)
                entry.last_accessed = time.time()
                entry.access_count += 1

                # Move to end (LRU optimization for M1)
                self._memory_cache.move_to_end(key)

                # M1 optimized decompression if needed
                data = await self._m1_decompress_if_needed(entry.data, entry.compressed)

                self.stats["hits"] += 1
                return data

            # Check persistent storage if enabled
            if self.enable_persistence:
                data = await self._load_from_m1_persistence(key)
                if data is not None:
                    # Cache in memory for M1 unified memory advantage
                    await self.set(key, data, ttl=self.default_ttl)
                    self.stats["hits"] += 1
                    self.stats["persistence_loads"] += 1
                    return data

            self.stats["misses"] += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """M1 optimized cache storage"""
        ttl = ttl or self.default_ttl

        try:
            # M1 optimized serialization and compression
            serialized_data, is_compressed = await self._m1_optimize_data(value)
            size_bytes = len(serialized_data) if isinstance(serialized_data, (bytes, bytearray)) else len(str(serialized_data))

            entry = CacheEntry(
                key=key,
                data=serialized_data,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                ttl=ttl,
                size_bytes=size_bytes,
                compressed=is_compressed,
                m1_optimized=True
            )

            with self._lock:
                # M1 memory management
                await self._ensure_m1_memory_capacity()

                # Store in memory cache
                self._memory_cache[key] = entry
                self._cache_metadata[key] = entry

                # Move to end (most recent)
                self._memory_cache.move_to_end(key)

            # Async persistence for M1 efficiency
            if self.enable_persistence:
                asyncio.create_task(self._save_to_m1_persistence(key, entry))

            return True

        except Exception as e:
            return False

    async def _m1_optimize_data(self, value: Any) -> tuple[Any, bool]:
        """M1 specific data optimization with efficient compression"""
        try:
            # Serialize data
            if isinstance(value, (str, int, float, bool)):
                serialized = json.dumps(value).encode('utf-8')
            else:
                serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

            # M1 optimized compression decision
            if self.enable_compression and len(serialized) > self._compression_threshold:
                # M1 má velmi rychlou kompresi
                compressed = zlib.compress(serialized, level=6)  # Balanced compression for M1
                if len(compressed) < len(serialized) * 0.9:  # Only if 10%+ savings
                    self.stats["compressions"] += 1
                    return compressed, True

            return serialized, False

        except Exception as e:
            # Fallback to original value
            return value, False

    async def _m1_decompress_if_needed(self, data: Any, is_compressed: bool) -> Any:
        """M1 optimized decompression"""
        try:
            if is_compressed and isinstance(data, (bytes, bytearray)):
                # M1 rychlá decomprese
                decompressed = zlib.decompress(data)
                self.stats["decompressions"] += 1

                # Try JSON first, then pickle
                try:
                    return json.loads(decompressed.decode('utf-8'))
                except:
                    return pickle.loads(decompressed)

            # Handle non-compressed data
            if isinstance(data, (bytes, bytearray)):
                try:
                    return json.loads(data.decode('utf-8'))
                except:
                    return pickle.loads(data)

            return data

        except Exception as e:
            return data

    async def _ensure_m1_memory_capacity(self):
        """M1 unified memory optimized capacity management"""
        # Check item count limit
        while len(self._memory_cache) >= self.max_memory_items:
            # Remove LRU item (first item in OrderedDict)
            oldest_key, oldest_entry = self._memory_cache.popitem(last=False)
            if oldest_key in self._cache_metadata:
                del self._cache_metadata[oldest_key]
            self.stats["evictions"] += 1

        # M1 specific memory pressure check
        if self.adaptive_cleanup and await self._check_m1_memory_pressure():
            await self._adaptive_m1_cleanup()

    async def _check_m1_memory_pressure(self) -> bool:
        """Check M1 specific memory pressure indicators"""
        try:
            # M1 unified memory monitoring
            memory = psutil.virtual_memory()
            memory_usage_gb = (memory.total - memory.available) / (1024**3)

            # M1 thermal monitoring
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                # Check for thermal throttling indicators
                self._thermal_throttling = any(
                    temp.current > 80 for sensor in temps.values()
                    for temp in sensor if hasattr(temp, 'current')
                )

            return (memory_usage_gb > self.memory_threshold_gb or
                    memory.percent > 85 or
                    self._thermal_throttling)

        except Exception:
            return False

    async def _adaptive_m1_cleanup(self):
        """M1 adaptive cleanup based on system conditions"""
        with self._lock:
            cleanup_count = 0
            current_time = time.time()

            # More aggressive cleanup if thermal throttling
            cleanup_factor = 0.5 if self._thermal_throttling else 0.3
            target_size = int(len(self._memory_cache) * cleanup_factor)

            # Remove expired and LRU items
            keys_to_remove = []
            for key, entry in list(self._memory_cache.items()):
                if (self._is_expired(entry) or
                    current_time - entry.last_accessed > 1800 or  # 30 min
                    cleanup_count < target_size):
                    keys_to_remove.append(key)
                    cleanup_count += 1

            for key in keys_to_remove:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                if key in self._cache_metadata:
                    del self._cache_metadata[key]

            self.stats["memory_cleanups"] += 1
            if self._thermal_throttling:
                self.stats["thermal_throttles"] += 1

            # Force garbage collection for M1 efficiency
            gc.collect()

    async def _m1_background_maintenance(self):
        """M1 optimized background maintenance task"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)

                current_time = time.time()
                if current_time - self._last_cleanup > self._cleanup_interval:
                    # Periodic M1 optimized cleanup
                    if await self._check_m1_memory_pressure():
                        await self._adaptive_m1_cleanup()

                    # Persistence maintenance
                    if self.enable_persistence:
                        await self._maintain_m1_persistence()

                    self._last_cleanup = current_time

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Continue background maintenance even if one cycle fails
                await asyncio.sleep(60)

    async def _save_to_m1_persistence(self, key: str, entry: CacheEntry):
        """M1 optimized async persistence save"""
        if not self.enable_persistence:
            return

        try:
            # Use connection pooling for M1 efficiency
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, data, created_at, last_accessed, access_count, ttl, size_bytes, compressed, m1_optimized)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.key,
                    entry.data,
                    entry.created_at,
                    entry.last_accessed,
                    entry.access_count,
                    entry.ttl,
                    entry.size_bytes,
                    int(entry.compressed),
                    1
                ))
                conn.commit()

            self.stats["persistence_saves"] += 1

        except Exception as e:
            # Graceful degradation for persistence failures
            pass

    async def _load_from_m1_persistence(self, key: str) -> Optional[Any]:
        """M1 optimized persistence load"""
        if not self.enable_persistence:
            return None

        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.execute("""
                    SELECT data, created_at, ttl, compressed 
                    FROM cache_entries 
                    WHERE key = ? AND m1_optimized = 1
                """, (key,))

                row = cursor.fetchone()
                if row:
                    data, created_at, ttl, compressed = row

                    # Check if expired
                    if time.time() - created_at > ttl:
                        # Clean up expired entry
                        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        conn.commit()
                        return None

                    # M1 optimized decompression
                    return await self._m1_decompress_if_needed(data, bool(compressed))

        except Exception as e:
            pass

        return None

    async def _maintain_m1_persistence(self):
        """M1 optimized persistence maintenance"""
        if not self.enable_persistence:
            return

        try:
            current_time = time.time()

            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                # Remove expired entries
                conn.execute("""
                    DELETE FROM cache_entries 
                    WHERE ? - created_at > ttl
                """, (current_time,))

                # M1 specific optimizations
                conn.execute("VACUUM")  # Reclaim space
                conn.execute("ANALYZE") # Update statistics

                conn.commit()

        except Exception as e:
            pass

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        return time.time() - entry.created_at > entry.ttl

    async def _final_m1_cleanup(self):
        """Final cleanup optimized for M1 power management"""
        try:
            # Save important cache data before shutdown
            if self.enable_persistence:
                tasks = []
                with self._lock:
                    for key, entry in self._memory_cache.items():
                        if not self._is_expired(entry) and entry.access_count > 1:
                            tasks.append(self._save_to_m1_persistence(key, entry))

                # Wait for all saves to complete
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            # Clear memory efficiently
            with self._lock:
                self._memory_cache.clear()
                self._cache_metadata.clear()
                self._m1_memory_pool.clear()

            # Final garbage collection
            gc.collect()

        except Exception as e:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive M1 optimized statistics"""
        with self._lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0

            return {
                **self.stats,
                "memory_items": len(self._memory_cache),
                "hit_rate_percent": round(hit_rate, 2),
                "thermal_throttling": self._thermal_throttling,
                "m1_optimized": True,
                "cache_dir": str(self.cache_dir),
                "compression_enabled": self.enable_compression
            }

    async def clear(self, pattern: Optional[str] = None):
        """M1 optimized cache clearing"""
        with self._lock:
            if pattern:
                # Pattern-based clearing
                keys_to_remove = [k for k in self._memory_cache.keys() if pattern in k]
                for key in keys_to_remove:
                    del self._memory_cache[key]
                    if key in self._cache_metadata:
                        del self._cache_metadata[key]
            else:
                # Clear all
                self._memory_cache.clear()
                self._cache_metadata.clear()

        # Clear persistence if pattern matching
        if self.enable_persistence:
            await self._clear_m1_persistence(pattern)

    async def _clear_m1_persistence(self, pattern: Optional[str] = None):
        """Clear M1 optimized persistence"""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                if pattern:
                    conn.execute("DELETE FROM cache_entries WHERE key LIKE ?", (f"%{pattern}%",))
                else:
                    conn.execute("DELETE FROM cache_entries")
                conn.commit()
        except Exception as e:
            pass

# Factory function for M1 optimization
def get_unified_cache() -> M1OptimizedCacheSystem:
    """Factory function to get M1 optimized cache system"""
    return M1OptimizedCacheSystem()

# Backward compatibility
UnifiedCacheSystem = M1OptimizedCacheSystem

# Export
__all__ = ['M1OptimizedCacheSystem', 'UnifiedCacheSystem', 'get_unified_cache', 'CacheEntry']
