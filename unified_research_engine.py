"""
Unified Research Engine - Kompletní řešení spojující všechny komponenty
Řeší všechny identifikované problémy v jedné konzistentní architektuře
"""

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager
import hashlib

# Unified imports s fallback podporou
try:
    from optimized_academic_scraper import scraping_session, ScrapingResult
    OPTIMIZED_SCRAPER_AVAILABLE = True
except ImportError:
    OPTIMIZED_SCRAPER_AVAILABLE = False

try:
    from optimized_database_manager import database_session, QueryRecord, ResultRecord
    OPTIMIZED_DB_AVAILABLE = True
except ImportError:
    OPTIMIZED_DB_AVAILABLE = False

try:
    from unified_config import get_config, ResearchStrategy
    UNIFIED_CONFIG_AVAILABLE = True
except ImportError:
    UNIFIED_CONFIG_AVAILABLE = False
    # Fallback hodnoty
    class ResearchStrategy:
        FAST = "fast"
        BALANCED = "balanced"
        THOROUGH = "thorough"

@dataclass
class ResearchRequest:
    """Unified request model"""
    query: str
    strategy: str = "balanced"  # Změněno na string pro kompatibilitu
    domain: str = "general"
    sources: Optional[List[str]] = None
    max_results: int = 10
    user_id: str = "default"
    budget_limit: Optional[float] = None

@dataclass
class ResearchResult:
    """Unified result model"""
    query_id: str
    success: bool
    sources_found: int
    total_tokens: int
    cost: float
    execution_time: float
    summary: str
    key_findings: List[str]
    detailed_results: List[Dict[str, Any]]
    cached: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class CostController:
    """Unified cost tracking and optimization"""

    def __init__(self):
        if UNIFIED_CONFIG_AVAILABLE:
            self.config = get_config()
            # Fall back to safe defaults when cost config is not present
            try:
                cost_cfg = getattr(self.config, 'cost', None)
                if cost_cfg is not None:
                    self.daily_budget = getattr(cost_cfg, 'daily_limit', 2.0)
                    self.monthly_budget = getattr(cost_cfg, 'monthly_target', 50.0)
                else:
                    self.daily_budget = 2.0
                    self.monthly_budget = 50.0
            except Exception:
                self.daily_budget = 2.0
                self.monthly_budget = 50.0
        else:
            self.daily_budget = 2.0
            self.monthly_budget = 50.0

        self.daily_costs = {}
        self.logger = self._setup_logger()

    def _setup_logger(self):
        import logging
        return logging.getLogger(__name__)

    async def check_budget(self, estimated_cost: float) -> bool:
        """Zkontroluje zda je v rámci rozpočtu"""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        current_daily_cost = self.daily_costs.get(today, 0.0)

        if current_daily_cost + estimated_cost > self.daily_budget:
            self.logger.warning(f"Budget exceeded: {current_daily_cost + estimated_cost:.4f} > {self.daily_budget:.4f}")
            return False

        return True

    async def record_cost(self, cost: float, tokens: int, operation: str):
        """Zaznamená náklady"""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.daily_costs[today] = self.daily_costs.get(today, 0.0) + cost

        self.logger.info(f"Cost recorded: ${cost:.4f} for {operation} ({tokens} tokens)")

class UnifiedResultCache:
    """Adapter nad globálním CacheManagerem pro ukládání ResearchResult"""

    def __init__(self):
        from cache_manager import CacheManager as BaseCacheManager

        if UNIFIED_CONFIG_AVAILABLE:
            self.config = get_config()
            self.enabled = self.config.cache.enabled
            self.ttl_seconds = getattr(self.config.cache, 'ttl_seconds', 3600)
        else:
            self.enabled = True
            self.ttl_seconds = 3600

        # Bez redis_config používá in-memory backend a může ukládat libovolné objekty
        self.backend = BaseCacheManager()

    def _generate_cache_key(self, query: str, domain: str, strategy: str) -> str:
        content = f"{query}:{domain}:{strategy}"
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def _serialize_result(result: 'ResearchResult') -> Dict[str, Any]:
        data = asdict(result)
        # Serializace datetime
        if isinstance(result.timestamp, datetime):
            data['timestamp'] = result.timestamp.isoformat()
        data['_type'] = 'ResearchResult'
        return data

    @staticmethod
    def _deserialize_result(data: Dict[str, Any]) -> 'ResearchResult':
        # Očekáváme _type marker
        ts = data.get('timestamp')
        if isinstance(ts, str):
            try:
                data['timestamp'] = datetime.fromisoformat(ts)
            except Exception:
                data['timestamp'] = datetime.now(timezone.utc)
        # Odstraň interní klíče
        data.pop('_type', None)
        return ResearchResult(**data)

    async def get(self, query: str, domain: str, strategy: str) -> Optional['ResearchResult']:
        if not self.enabled:
            return None
        key = self._generate_cache_key(query, domain, strategy)
        raw = self.backend.get(key)
        if raw is None:
            return None
        if isinstance(raw, ResearchResult):
            return raw
        if isinstance(raw, dict):
            try:
                if raw.get('_type') == 'ResearchResult':
                    return self._deserialize_result(dict(raw))
            except Exception:
                return None
        return None

    async def set(self, query: str, domain: str, strategy: str, result: 'ResearchResult') -> None:
        if not self.enabled:
            return
        key = self._generate_cache_key(query, domain, strategy)
        payload = self._serialize_result(result)
        self.backend.set(key, payload, ttl=self.ttl_seconds)

    def get_stats(self) -> Dict[str, Any]:
        return self.backend.get_stats()

class StrategyOptimizer:
    """Optimalizuje research strategie na základě historických dat"""

    def __init__(self):
        self.performance_history = {}
        self.logger = self._setup_logger()

    def _setup_logger(self):
        import logging
        return logging.getLogger(__name__)

    def suggest_strategy(self, query: str, domain: str) -> str:
        """Navrhne nejlepší strategii pro dotaz"""
        # Jednoduchá heuristika - v budoucnu ML model
        query_length = len(query.split())

        if query_length < 3:
            return "fast"
        elif query_length < 8:
            return "balanced"
        else:
            return "thorough"

    def suggest_sources(self, domain: str) -> List[str]:
        """Navrhne nejlepší zdroje pro doménu"""
        domain_mapping = {
            'medical': ['pubmed', 'semantic_scholar', 'wikipedia'],
            'technology': ['arxiv', 'semantic_scholar', 'wikipedia'],
            'general': ['wikipedia', 'openalex', 'semantic_scholar']
        }
        return domain_mapping.get(domain, ['wikipedia'])

class UnifiedResearchEngine:
    """Unified research engine spojující všechny komponenty"""

    def __init__(self):
        self.cost_controller = CostController()
        self.cache_manager = UnifiedResultCache()
        self.strategy_optimizer = StrategyOptimizer()
        self.logger = self._setup_logger()

    def _setup_logger(self):
        import logging
        return logging.getLogger(__name__)

    async def research(self, request: ResearchRequest) -> ResearchResult:
        """Hlavní research metoda"""
        start_time = time.time()

        # 1. Optimalizace strategie pokud není zadána
        if not request.sources:
            request.sources = self.strategy_optimizer.suggest_sources(request.domain)

        # 2. Check cache
        cached_result = await self.cache_manager.get(
            request.query, request.domain, request.strategy
        )
        if cached_result:
            cached_result.cached = True
            self.logger.info(f"Cache hit for query: {request.query}")
            return cached_result

        # 3. Estimate cost
        estimated_cost = self._estimate_cost(request)
        if not await self.cost_controller.check_budget(estimated_cost):
            raise Exception("Budget limit exceeded")

        # 4. Execute research
        result = await self._execute_research(request, start_time)

        # 5. Record actual cost
        await self.cost_controller.record_cost(
            result.cost, result.total_tokens, "research"
        )

        # 6. Cache result
        await self.cache_manager.set(
            request.query, request.domain, request.strategy, result
        )

        # 7. Save to database (pokud dostupné)
        if OPTIMIZED_DB_AVAILABLE:
            await self._save_to_database(request, result)

        self.logger.info(f"Research completed in {result.execution_time:.2f}s")
        return result

    async def _execute_research(self, request: ResearchRequest, start_time: float) -> ResearchResult:
        """Spustí research proces"""
        query_id = hashlib.md5(f"{request.query}:{time.time()}".encode()).hexdigest()[:16]

        # Scraping - použij optimized pokud dostupné
        if OPTIMIZED_SCRAPER_AVAILABLE:
            async with scraping_session() as scraper:
                scraping_results = await scraper.scrape_all_sources(
                    request.query, request.sources
                )
        else:
            # Fallback na orchestrátor v academic_scraper
            from academic_scraper import create_scraping_orchestrator
            orchestrator = create_scraping_orchestrator()
            scraping_results = await orchestrator.scrape_all_sources(request.query, request.sources)

        # Process results
        successful_results = [r for r in scraping_results if r.success]

        # --- Vylepšení: deduplikace a extrakce klíčových informací ---
        def deduplicate_results(results):
            seen = set()
            deduped = []
            for r in results:
                # Zkusíme deduplikovat podle DOI, titulu nebo URL (co je dostupné)
                key = None
                data = getattr(r, 'data', {})
                if 'doi' in data:
                    key = data['doi']
                elif 'title' in data:
                    key = data['title'].lower()
                elif 'url' in data:
                    key = data['url']
                if key and key not in seen:
                    seen.add(key)
                    deduped.append(r)
                elif not key:
                    deduped.append(r)  # Pokud není klíč, přidáme vše
            return deduped

        def extract_key_findings(results):
            findings = []
            for r in results:
                data = getattr(r, 'data', {})
                if 'title' in data:
                    findings.append(data['title'])
                elif 'articles' in data and data['articles']:
                    findings.extend([a['title'] for a in data['articles'] if 'title' in a])
                elif 'works' in data and data['works']:
                    findings.extend([w['title'] for w in data['works'] if 'title' in w])
            return findings[:10]  # Omezíme na 10 nejdůležitějších

        deduped_results = deduplicate_results(successful_results)
        key_findings = extract_key_findings(deduped_results)
        summary = self._generate_summary(deduped_results)
        # Calculate metrics
        total_tokens = sum(len(str(r.data)) // 4 for r in deduped_results)  # Rough estimate
        cost = self._calculate_actual_cost(total_tokens)
        execution_time = time.time() - start_time
        return ResearchResult(
            query_id=query_id,
            success=len(deduped_results) > 0,
            sources_found=len(deduped_results),
            total_tokens=total_tokens,
            cost=cost,
            execution_time=execution_time,
            summary=summary,
            key_findings=key_findings,
            detailed_results=[r.__dict__ if hasattr(r, '__dict__') else vars(r) for r in deduped_results]
        )

    async def _save_to_database(self, request: ResearchRequest, result: ResearchResult):
        """Uloží do databáze pokud je dostupná optimized verze"""
        async with database_session() as db:
            query_record = QueryRecord(
                id=result.query_id,
                query=request.query,
                domain=request.domain,
                strategy=request.strategy,
                sources=request.sources or [],
                user_id=request.user_id,
                timestamp=datetime.now(timezone.utc),
                cost_estimate=result.cost
            )
            await db.save_research_query_optimized(query_record)

            result_record = ResultRecord(
                id=f"{result.query_id}_result",
                query_id=result.query_id,
                strategy=request.strategy,
                sources_found=result.sources_found,
                total_tokens=result.total_tokens,
                cost=result.cost,
                execution_time=result.execution_time,
                summary=result.summary,
                key_findings=result.key_findings,
                timestamp=result.timestamp
            )
            await db.save_research_result_optimized(result_record)

    def _estimate_cost(self, request: ResearchRequest) -> float:
        """Odhadne náklady na research"""
        base_cost = 0.001  # $0.001 per source
        strategy_multiplier = {
            "fast": 1.0,
            "balanced": 2.0,
            "thorough": 4.0
        }

        num_sources = len(request.sources) if request.sources else 3
        return base_cost * num_sources * strategy_multiplier.get(request.strategy, 2.0)

    def _calculate_actual_cost(self, tokens: int) -> float:
        """Vypočítá skutečné náklady"""
        # Využij konfiguraci pokud je dostupná
        if UNIFIED_CONFIG_AVAILABLE:
            cfg = get_config()
            price = getattr(cfg.cost, 'token_price_per_1k', 0.00025)
        else:
            price = 0.00025
        return (tokens / 1000) * price

    def _generate_summary(self, results) -> str:
        """Generuje souhrn výsledků"""
        if not results:
            return "No results found."

        source_names = [r.source for r in results]
        return f"Found information from {len(results)} sources: {', '.join(source_names)}"

    def _extract_key_findings(self, results) -> List[str]:
        """Extrahuje klíčová zjištění"""
        findings = []
        for result in results[:3]:  # Top 3 results
            if hasattr(result, 'data') and result.data.get('title'):
                findings.append(f"From {result.source}: {result.data['title']}")
        return findings

    async def get_statistics(self) -> Dict[str, Any]:
        """Vrátí statistiky systému"""
        cache_stats = self.cache_manager.get_stats()

        return {
            'cache_performance': cache_stats,
            'daily_costs': self.cost_controller.daily_costs,
            'system_health': 'healthy'
        }

# Factory functions
def create_unified_research_engine() -> UnifiedResearchEngine:
    """Factory pro unified research engine"""
    return UnifiedResearchEngine()

@asynccontextmanager
async def research_session():
    """Context manager pro research session"""
    engine = create_unified_research_engine()
    try:
        yield engine
    finally:
        # Cleanup if needed
        pass
