"""
Agresivní cache systém pro maximální výkon
Optimalizováno pro soukromé použití
"""
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Optional
import logging
from datetime import datetime, timedelta

class HighPerformanceCache:
    """Vysoce výkonný cache systém s persistencí"""
    
    def __init__(self, cache_dir: Path, max_size_mb: int = 500):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.memory_cache = {}  # Hot cache v paměti
        self.access_times = {}  # Pro LRU
        self.logger = logging.getLogger(__name__)
        
    def _get_cache_key(self, key: str) -> str:
        """Rychlé generování cache klíče"""
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Cesta k cache souboru"""
        return self.cache_dir / f"{cache_key}.cache"
    
    def get(self, key: str, default=None) -> Any:
        """Rychlé získání z cache"""
        cache_key = self._get_cache_key(key)
        
        # 1. Zkus memory cache (nejrychlejší)
        if cache_key in self.memory_cache:
            self.access_times[cache_key] = time.time()
            return self.memory_cache[cache_key]
        
        # 2. Zkus disk cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Kontrola expiraci
                if data['expires'] > time.time():
                    # Přidej do memory cache pro příště
                    self.memory_cache[cache_key] = data['value']
                    self.access_times[cache_key] = time.time()
                    return data['value']
                else:
                    # Smaž expirované
                    cache_path.unlink()
            except:
                # Pokud je cache poškozená, smaž ji
                cache_path.unlink(missing_ok=True)
        
        return default
    
    def set(self, key: str, value: Any, ttl_hours: int = 24):
        """Rychlé uložení do cache"""
        cache_key = self._get_cache_key(key)
        expires = time.time() + (ttl_hours * 3600)
        
        # Ulož do memory cache
        self.memory_cache[cache_key] = value
        self.access_times[cache_key] = time.time()
        
        # Ulož na disk asynchronně (neblokuj)
        cache_path = self._get_cache_path(cache_key)
        try:
            data = {'value': value, 'expires': expires}
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Cache write failed: {e}")
        
        # Cleanup pokud je cache příliš velká
        self._cleanup_if_needed()
    
    def _cleanup_if_needed(self):
        """Rychlý cleanup při překročení limitu"""
        # Cleanup memory cache
        if len(self.memory_cache) > 1000:  # Max 1000 items v paměti
            # Odstraň nejstarší
            oldest_keys = sorted(self.access_times.items(), key=lambda x: x[1])[:100]
            for key, _ in oldest_keys:
                self.memory_cache.pop(key, None)
                self.access_times.pop(key, None)
        
        # Cleanup disk cache (méně často)
        if time.time() % 300 < 1:  # Každých 5 minut
            self._cleanup_disk_cache()
    
    def _cleanup_disk_cache(self):
        """Cleanup disk cache"""
        try:
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.cache"))
            
            if total_size > self.max_size_bytes:
                # Seřaď podle access time a smaž nejstarší
                cache_files = [(f, f.stat().st_mtime) for f in self.cache_dir.glob("*.cache")]
                cache_files.sort(key=lambda x: x[1])
                
                for cache_file, _ in cache_files[:len(cache_files)//4]:  # Smaž 25%
                    cache_file.unlink(missing_ok=True)
        except Exception as e:
            self.logger.warning(f"Cache cleanup failed: {e}")

# Globální instance pro celou aplikaci
high_perf_cache = HighPerformanceCache(Path("cache"))

def cache_result(ttl_hours: int = 24):
    """Decorator pro cache výsledků funkcí"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Vytvoř cache klíč z funkce a argumentů
            cache_key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            
            # Zkus cache
            result = high_perf_cache.get(cache_key)
            if result is not None:
                return result
            
            # Spočítej a ulož
            result = await func(*args, **kwargs)
            high_perf_cache.set(cache_key, result, ttl_hours)
            return result
        
        def sync_wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            
            result = high_perf_cache.get(cache_key)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            high_perf_cache.set(cache_key, result, ttl_hours)
            return result
        
        return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else sync_wrapper
    return decorator
