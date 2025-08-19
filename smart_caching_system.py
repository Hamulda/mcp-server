Smart Caching & Predictive Preloading System
Kreativní vylepšení pro dramatické zvýšení rychlosti a snížení spotřeby
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
import pickle
import sqlite3
from datetime import datetime, timedelta

try:
    from unified_config import get_config
    from local_ai_adapter import M1OptimizedOllamaClient
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata"""
    key: str
    data: Any
    timestamp: float
    access_count: int
    last_accessed: float
    source: str
    query_pattern: str
    relevance_score: float = 1.0
    size_bytes: int = 0

class PredictiveQueryAnalyzer:
    """Analyzuje vzory dotazů pro prediktivní načítání"""

    def __init__(self):
        self.query_history: deque = deque(maxlen=1000)
        self.common_patterns: Dict[str, int] = defaultdict(int)
        self.peptide_sequences: Dict[str, List[str]] = defaultdict(list)

    def analyze_query_pattern(self, query: str) -> Dict[str, Any]:
        """Analyzuje vzor dotazu a předpovídá související dotazy"""
        query_lower = query.lower()

        # Detekce peptide patterns
        peptide_keywords = ['bpc', 'tb500', 'gh', 'ghrp', 'cjc', 'ipamorelin', 'thymosin']
        detected_peptides = [kw for kw in peptide_keywords if kw in query_lower]

        # Detekce research patterns
        research_types = {
            'dosage': ['dosage', 'dose', 'mg', 'mcg', 'protocol'],
            'safety': ['side effects', 'safety', 'risks', 'contraindications'],
            'mechanism': ['mechanism', 'how it works', 'pathway', 'receptor'],
            'benefits': ['benefits', 'effects', 'results', 'efficacy'],
            'stacking': ['stack', 'combination', 'synergy', 'together']
        }

        detected_types = []
        for research_type, keywords in research_types.items():
            if any(kw in query_lower for kw in keywords):
                detected_types.append(research_type)

        # Predikce souvisejících dotazů
        predicted_queries = self._predict_related_queries(detected_peptides, detected_types)

        return {
            'peptides': detected_peptides,
            'research_types': detected_types,
            'predicted_queries': predicted_queries,
            'pattern_confidence': len(detected_peptides) + len(detected_types) * 0.5
        }

    def _predict_related_queries(self, peptides: List[str], research_types: List[str]) -> List[str]:
        """Předpovídá související dotazy"""
        predictions = []

        # Pro každý peptide předpovídej nejčastější typy dotazů
        for peptide in peptides:
            if 'dosage' not in research_types:
                predictions.append(f"{peptide} dosage protocol")
            if 'safety' not in research_types:
                predictions.append(f"{peptide} side effects")
            if 'mechanism' not in research_types:
                predictions.append(f"{peptide} mechanism of action")

        # Kombinace peptidů (stacking)
        if len(peptides) == 1 and 'stacking' not in research_types:
            peptide = peptides[0]
            common_stacks = {
                'bpc': ['tb500', 'gh'],
                'tb500': ['bpc', 'ghrp'],
                'ghrp': ['cjc', 'ipamorelin']
            }
            if peptide in common_stacks:
                for stack_partner in common_stacks[peptide]:
                    predictions.append(f"{peptide} {stack_partner} stack")

        return predictions[:5]  # Limit na 5 predikcí

class IntelligentCacheManager:
    """Inteligentní cache manager s prediktivním načítáním"""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("cache/intelligent")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # SQLite pro rychlé queries
        self.db_path = self.cache_dir / "cache_index.db"
        self._init_database()

        # In-memory cache pro nejčastější dotazy
        self.hot_cache: Dict[str, CacheEntry] = {}
        self.max_hot_cache_size = 100

        # Predictive analyzer
        self.query_analyzer = PredictiveQueryAnalyzer()

        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.predictive_hits = 0

    def _init_database(self):
        """Inicializace SQLite databáze pro cache index"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    timestamp REAL,
                    access_count INTEGER,
                    last_accessed REAL,
                    source TEXT,
                    query_pattern TEXT,
                    relevance_score REAL,
                    size_bytes INTEGER,
                    file_path TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_pattern ON cache_entries(query_pattern)
            """)

    def _generate_cache_key(self, query: str, source: str = "", params: Dict = None) -> str:
        """Generuje cache klíč s normalizací"""
        # Normalizace dotazu
        normalized_query = query.lower().strip()
        normalized_query = ' '.join(normalized_query.split())  # Cleanup whitespace

        cache_content = f"{normalized_query}:{source}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.sha256(cache_content.encode()).hexdigest()

    async def get_cached_result(self, query: str, source: str = "", params: Dict = None) -> Optional[Any]:
        """Získá cached výsledek s inteligentní relevance scoring"""
        cache_key = self._generate_cache_key(query, source, params)

        # Check hot cache first
        if cache_key in self.hot_cache:
            entry = self.hot_cache[cache_key]
            entry.access_count += 1
            entry.last_accessed = time.time()
            self.cache_hits += 1
            logger.debug(f"🔥 Hot cache hit for {query[:50]}...")
            return entry.data

        # Check SQLite cache
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT file_path FROM cache_entries WHERE key = ?", (cache_key,)
            )
            result = cursor.fetchone()

            if result:
                file_path = Path(result[0])
                if file_path.exists():
                    try:
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)

                        # Update access stats
                        conn.execute("""
                            UPDATE cache_entries 
                            SET access_count = access_count + 1, last_accessed = ?
                            WHERE key = ?
                        """, (time.time(), cache_key))

                        self.cache_hits += 1
                        logger.debug(f"💾 Disk cache hit for {query[:50]}...")

                        # Promote to hot cache if frequently accessed
                        if len(self.hot_cache) < self.max_hot_cache_size:
                            entry = CacheEntry(
                                key=cache_key,
                                data=data,
                                timestamp=time.time(),
                                access_count=1,
                                last_accessed=time.time(),
                                source=source,
                                query_pattern=query[:100]
                            )
                            self.hot_cache[cache_key] = entry

                        return data
                    except Exception as e:
                        logger.warning(f"Failed to load cached data: {e}")

        self.cache_misses += 1
        return None

    async def cache_result(self, query: str, data: Any, source: str = "", params: Dict = None):
        """Uloží výsledek do cache s inteligentním managementem"""
        cache_key = self._generate_cache_key(query, source, params)

        # Serialize data
        try:
            serialized_data = pickle.dumps(data)
            size_bytes = len(serialized_data)
        except Exception as e:
            logger.error(f"Failed to serialize data for caching: {e}")
            return

        # Save to disk
        file_path = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(file_path, 'wb') as f:
                f.write(serialized_data)
        except Exception as e:
            logger.error(f"Failed to save cache file: {e}")
            return

        # Update SQLite index
        with sqlite3.connect(self.db_path) as conn:
            pattern = self.query_analyzer.analyze_query_pattern(query)

            conn.execute("""
                INSERT OR REPLACE INTO cache_entries 
                (key, timestamp, access_count, last_accessed, source, query_pattern, 
                 relevance_score, size_bytes, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key, time.time(), 0, time.time(), source,
                query[:100], pattern['pattern_confidence'], size_bytes, str(file_path)
            ))

        logger.debug(f"💾 Cached result for {query[:50]}... ({size_bytes} bytes)")

    async def predictive_preload(self, query: str, max_predictions: int = 3):
        """Prediktivní načítání souvisejících dotazů"""
        pattern = self.query_analyzer.analyze_query_pattern(query)

        if pattern['pattern_confidence'] > 1.0:  # Sufficient confidence
            predicted_queries = pattern['predicted_queries'][:max_predictions]

            logger.info(f"🔮 Preloading {len(predicted_queries)} predicted queries")

            # Asynchronně načti předpovězené dotazy na pozadí
            preload_tasks = []
            for predicted_query in predicted_queries:
                task = asyncio.create_task(
                    self._background_preload(predicted_query)
                )
                preload_tasks.append(task)

            # Don't wait for completion - run in background
            asyncio.create_task(self._track_preload_completion(preload_tasks))

    async def _background_preload(self, query: str):
        """Background preloading of predicted queries"""
        try:
            # Check if already cached
            cached = await self.get_cached_result(query)
            if cached:
                return

            # Simulate lightweight research (using fast model)
            if CONFIG_AVAILABLE:
                async with M1OptimizedOllamaClient() as client:
                    result = await client.generate_optimized(
                        query,
                        priority="speed",
                        use_specialized_prompt=True
                    )
                    await self.cache_result(query, result, "preload")
                    logger.debug(f"🔮 Preloaded: {query[:50]}...")

        except Exception as e:
            logger.debug(f"Preload failed for {query}: {e}")

    async def _track_preload_completion(self, tasks: List[asyncio.Task]):
        """Track completion of preload tasks"""
        completed = 0
        for task in tasks:
            try:
                await task
                completed += 1
                self.predictive_hits += 1
            except Exception:
                pass

        if completed > 0:
            logger.info(f"✅ Completed preloading {completed} queries")

    async def cleanup_old_cache(self, max_age_days: int = 7, max_size_gb: float = 1.0):
        """Inteligentní čištění starého cache"""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        max_size_bytes = max_size_gb * 1024 * 1024 * 1024

        with sqlite3.connect(self.db_path) as conn:
            # Get cache stats
            cursor = conn.execute("""
                SELECT SUM(size_bytes) as total_size, COUNT(*) as total_entries
                FROM cache_entries
            """)
            stats = cursor.fetchone()
            total_size, total_entries = stats[0] or 0, stats[1] or 0

            logger.info(f"📊 Cache stats: {total_entries} entries, {total_size/1024/1024:.1f}MB")

            # Remove old entries
            cursor = conn.execute("""
                SELECT key, file_path FROM cache_entries 
                WHERE last_accessed < ? 
                ORDER BY last_accessed ASC
            """, (cutoff_time,))

            old_entries = cursor.fetchall()
            removed_count = 0

            for key, file_path in old_entries:
                try:
                    Path(file_path).unlink(missing_ok=True)
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove cache entry: {e}")

            # Remove least accessed entries if still over size limit
            if total_size > max_size_bytes:
                cursor = conn.execute("""
                    SELECT key, file_path FROM cache_entries 
                    ORDER BY access_count ASC, last_accessed ASC
                """)

                remaining_entries = cursor.fetchall()
                current_size = total_size

                for key, file_path in remaining_entries:
                    if current_size <= max_size_bytes:
                        break

                    try:
                        file_size = Path(file_path).stat().st_size
                        Path(file_path).unlink(missing_ok=True)
                        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        current_size -= file_size
                        removed_count += 1
                    except Exception:
                        pass

            conn.commit()

        logger.info(f"🧹 Cleaned {removed_count} old cache entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Získá statistiky cache"""
        hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(size_bytes) as total_size,
                    AVG(access_count) as avg_access_count,
                    SUM(CASE WHEN last_accessed > ? THEN 1 ELSE 0 END) as recent_entries
                FROM cache_entries
            """, (time.time() - 3600,))  # Last hour

            stats = cursor.fetchone()

        return {
            'cache_hit_rate': hit_rate,
            'total_requests': self.cache_hits + self.cache_misses,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'predictive_hits': self.predictive_hits,
            'total_entries': stats[0] or 0,
            'total_size_mb': (stats[1] or 0) / 1024 / 1024,
            'avg_access_count': stats[2] or 0,
            'recent_entries': stats[3] or 0,
            'hot_cache_size': len(self.hot_cache)
        }

# Global cache instance
_cache_instance = None

def get_intelligent_cache() -> IntelligentCacheManager:
    """Get global intelligent cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = IntelligentCacheManager()
    return _cache_instance

# Convenience functions
async def smart_cache_get(query: str, source: str = "") -> Optional[Any]:
    """Smart cache retrieval with predictive preloading"""
    cache = get_intelligent_cache()
    result = await cache.get_cached_result(query, source)

    if result is None:
        # Trigger predictive preloading for future queries
        await cache.predictive_preload(query)

    return result

async def smart_cache_set(query: str, data: Any, source: str = ""):
    """Smart cache storage"""
    cache = get_intelligent_cache()
    await cache.cache_result(query, data, source)
