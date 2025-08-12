"""
Unified Research Engine - Kompletní řešení spojující všechny komponenty
Řeší všechny identifikované problémy v jedné konzistentní architektuře
"""

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
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
            self.daily_budget = self.config.cost.daily_limit
            self.monthly_budget = self.config.cost.monthly_target
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

class CacheManager:
    """Unified caching system"""
    
    def __init__(self):
        if UNIFIED_CONFIG_AVAILABLE:
            self.config = get_config()
            self.enabled = self.config.cache.enabled
            self.ttl_hours = self.config.cache.ttl_hours
        else:
            self.enabled = True
            self.ttl_hours = 24
            
        self.memory_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _generate_cache_key(self, query: str, domain: str, strategy: str) -> str:
        """Generuje cache klíč"""
        content = f"{query}:{domain}:{strategy}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get(self, query: str, domain: str, strategy: str) -> Optional[ResearchResult]:
        """Získá cached výsledek"""
        if not self.enabled:
            return None
        
        cache_key = self._generate_cache_key(query, domain, strategy)
        
        # Check memory cache first
        cached_item = self.memory_cache.get(cache_key)
        if cached_item:
            expiry_str = cached_item.get('expires_at')
            if expiry_str:
                expiry = datetime.fromisoformat(expiry_str)
                if expiry > datetime.now(timezone.utc):
                    self.cache_hits += 1
                    return cached_item['result']
        
        self.cache_misses += 1
        return None
    
    async def set(self, query: str, domain: str, strategy: str, result: ResearchResult):
        """Uloží výsledek do cache"""
        if not self.enabled:
            return
        
        cache_key = self._generate_cache_key(query, domain, strategy)
        from datetime import timedelta
        expires_at = datetime.now(timezone.utc) + timedelta(hours=self.ttl_hours)
        
        self.memory_cache[cache_key] = {
            'result': result,
            'expires_at': expires_at.isoformat(),
            'created_at': datetime.now(timezone.utc).isoformat()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Vrátí cache statistiky"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cached_items': len(self.memory_cache)
        }

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
        self.cache_manager = CacheManager()
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
            # Fallback na legacy scraper
            from academic_scraper import scrape_all_sources
            scraping_results = await scrape_all_sources(request.query, request.sources)
        
        # Process results
        successful_results = [r for r in scraping_results if r.success]
        
        # Generate summary a key findings
        summary = self._generate_summary(successful_results)
        key_findings = self._extract_key_findings(successful_results)
        
        # Calculate metrics
        total_tokens = sum(len(str(r.data)) // 4 for r in successful_results)  # Rough estimate
        cost = self._calculate_actual_cost(total_tokens)
        execution_time = time.time() - start_time
        
        return ResearchResult(
            query_id=query_id,
            success=len(successful_results) > 0,
            sources_found=len(successful_results),
            total_tokens=total_tokens,
            cost=cost,
            execution_time=execution_time,
            summary=summary,
            key_findings=key_findings,
            detailed_results=[r.__dict__ if hasattr(r, '__dict__') else vars(r) for r in successful_results]
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
        # Gemini pricing: $0.00025 per 1K tokens
        return (tokens / 1000) * 0.00025
    
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
