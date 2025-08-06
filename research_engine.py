"""
Hlavní třída pro research engine - koordinuje všechny komponenty
"""
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json

from config import *
from gemini_manager import GeminiAIManager, TokenOptimizer
from text_processing_utils import text_processor
from config_personal import *
from high_performance_cache import cache_result
from text_processing_utils import fast_text_processor

@dataclass
class ResearchQuery:
    """Struktura pro research dotaz s centralizovanou optimalizací"""
    query: str
    sources: List[str]
    depth: str = "medium"
    max_results: int = 50
    languages: List[str] = None
    analysis_type: str = "summary"
    token_budget: int = None
    domain: str = "general"  # Nové pole pro doménu

    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en", "cs"]

        # Automatické nastavení podle depth
        if self.token_budget is None:
            depth_config = RESEARCH_DEPTHS.get(self.depth, RESEARCH_DEPTHS['medium'])
            self.token_budget = depth_config['token_budget']
            self.analysis_type = depth_config['analysis_type']
            if self.max_results > depth_config['max_sources']:
                self.max_results = depth_config['max_sources']

        # Centralizovaná optimalizace dotazu
        self._optimize_query()

    def _optimize_query(self):
        """Centralizovaná optimalizace dotazu při vytvoření"""
        optimizer = QueryOptimizer()

        # Optimalizace podle domény
        self.query = optimizer.optimize_query(self.query, self.domain)

        # Rozšíření o synonyma pro lepší pokrytí
        self.query = optimizer.expand_query_with_synonyms(self.query)

        logging.info(f"Optimalizovaný dotaz ({self.domain}): {self.query}")

@dataclass
class ResearchResult:
    """Struktura pro výsledky researche"""
    query: str
    timestamp: datetime
    sources_found: int
    summary: str
    key_findings: List[str]
    sources: List[Dict]
    confidence_score: float
    full_report_path: str
    cost_info: Dict[str, Any] = None
    tokens_used: int = 0
    processing_time: float = 0

class QueryOptimizer:
    """Centrální třída pro optimalizaci dotazů"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def optimize_query(self, query: str, domain: str = "general", max_length: int = 200) -> str:
        """Centrální optimalizace dotazu podle domény"""
        if not query.strip():
            raise ValueError("Dotaz nemůže být prázdný")

        # Základní čištění
        cleaned_query = text_processor.clean_text(query)

        # Optimalizace podle domény
        if domain == "medical":
            return text_processor.distill_medical_text(cleaned_query, max_sentences=5)
        elif domain == "academic":
            return text_processor.prioritize_by_keywords(cleaned_query, text_processor.research_keywords, 5)
        else:
            # Obecná optimalizace
            return text_processor.optimize_for_tokens(cleaned_query, max_length * 5)  # Přibližný převod na tokeny

    def expand_query_with_synonyms(self, query: str) -> str:
        """Rozšíření dotazu o synonyma pro lepší pokrytí"""
        query = query.strip().lower()

        expansions = {
            'ai': 'artificial intelligence machine learning',
            'ml': 'machine learning artificial intelligence',
            'crypto': 'cryptocurrency blockchain bitcoin',
            'covid': 'coronavirus pandemic covid-19',
            'nootropika': 'nootropics cognitive enhancers smart drugs',
            'peptidy': 'peptides bioactive peptides therapeutic peptides'
        }

        for abbrev, expansion in expansions.items():
            if abbrev in query:
                query = query.replace(abbrev, expansion)

        return query

    def optimize_medical_query(self, query: str) -> str:
        """Specializovaná optimalizace pro lékařské dotazy"""
        return self.optimize_query(query, domain="medical")

class ResearchEngine:
    """Hlavní třída s centralizovanou optimalizací dotazů"""

    def __init__(self):
        self.setup_logging()
        self.query_optimizer = QueryOptimizer()
        self.scrapers = {}
        self.analyzers = {}
        self.database = None
        self.gemini_manager = None
        self.token_optimizer = TokenOptimizer()
        self.daily_cost_tracker = 0.0
        self.initialize_components()

    def setup_logging(self):
        """Nastavení logování"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('research.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_components(self):
        """Inicializace všech komponent"""
        try:
            self.logger.info("Inicializuji research engine s Gemini optimalizací...")

            # Inicializace Gemini AI managera
            gemini_key = GEMINI_API_KEY
            if gemini_key:
                self.gemini_manager = GeminiAIManager(gemini_key)
                self.logger.info("✅ Gemini AI manager připraven")
            else:
                self.logger.warning("⚠️ GEMINI_API_KEY není nastaven - použijí se fallback metody")

            # Inicializace textového analyzátoru
            self.text_analyzer = TextAnalyzer()

            # Další komponenty budou inicializovány postupně
            self.logger.info("Research engine úspěšně inicializován")
        except Exception as e:
            self.logger.error(f"Chyba při inicializaci: {e}")

    async def conduct_research(self, query: ResearchQuery) -> ResearchResult:
        """Hlavní metoda pro provedení researche s Gemini optimalizací"""
        start_time = datetime.now()
        self.logger.info(f"🔍 Zahajuji token-optimalizovaný research pro: {query.query}")

        # Kontrola cost limitu
        if ENABLE_COST_TRACKING and self.daily_cost_tracker >= DAILY_COST_LIMIT:
            raise Exception(f"Denní cost limit {DAILY_COST_LIMIT} USD byl překročen")

        try:
            # 1. Příprava a validace dotazu
            validated_query = self._validate_and_optimize_query(query)

            # 2. Shromáždění dat z různých zdrojů
            self.logger.info("📥 Shromažďuji data ze zdrojů...")
            raw_data = await self._gather_data_efficiently(validated_query)

            # 3. Token-optimalizovaná analýza dat s Gemini
            self.logger.info("🧠 Provádím AI analýzu s Gemini...")
            processed_data, analysis_cost = await self._analyze_data_with_gemini(raw_data, validated_query)

            # 4. Vytvoření master shrnutí
            self.logger.info("📊 Vytvářím finální report...")
            result = await self._generate_optimized_report(validated_query, processed_data)

            # 5. Cost tracking a uložení
            result.cost_info = analysis_cost
            result.processing_time = (datetime.now() - start_time).total_seconds()

            if ENABLE_COST_TRACKING:
                self.daily_cost_tracker += analysis_cost.get('estimated_cost_usd', 0)
                self.logger.info(f"💰 Denní náklady: ${self.daily_cost_tracker:.4f}")

            # 6. Uložení do databáze
            await self._save_results(result)

            self.logger.info(f"✅ Research dokončen za {result.processing_time:.1f}s")
            return result

        except Exception as e:
            self.logger.error(f"❌ Chyba během researche: {e}")
            raise

    async def conduct_health_research(self, query: ResearchQuery) -> ResearchResult:
        """
        Provádí výzkum zaměřený na zdravotní témata.
        """
        start_time = datetime.now()
        self.logger.info(f"🔍 Zahajuji zdravotní výzkum pro: {query.query}")

        try:
            # 1. Shromáždění dat z relevantních zdrojů
            self.logger.info("📥 Shromažďuji data ze zdravotních zdrojů...")
            health_scraper = HealthResearchScraper()
            pubmed_data = await health_scraper.scrape_pubmed(query.query, query.max_results)
            clinical_trials_data = await health_scraper.scrape_clinical_trials(query.query, query.max_results)

            combined_data = pubmed_data + clinical_trials_data

            # 2. Analýza textů pomocí Gemini
            self.logger.info("🧠 Provádím analýzu textů pomocí Gemini...")
            texts = [item.get('abstract', '') or item.get('summary', '') for item in combined_data]
            health_prompt = self.gemini_manager.create_health_prompt(texts, query.query)

            response = await self.gemini_manager._call_gemini_async(health_prompt)
            analysis_results = self.gemini_manager._parse_batch_response(response, len(texts))

            # 3. Extrakce zdravotních klíčových slov
            self.logger.info("🔑 Extrahuji zdravotní klíčová slova...")
            all_texts = ' '.join(texts)
            health_keywords = self.text_analyzer.extract_health_keywords(all_texts)

            # 4. Sumarizace výsledků
            self.logger.info("📊 Vytvářím shrnutí výsledků...")
            summary = self.text_analyzer.summarize_text(all_texts, max_sentences=5)

            # 5. Vytvoření výsledného objektu
            result = ResearchResult(
                query=query.query,
                timestamp=start_time,
                sources_found=len(combined_data),
                summary=summary,
                key_findings=health_keywords,
                sources=analysis_results,
                confidence_score=0.85,
                full_report_path=""
            )

            self.logger.info("✅ Zdravotní výzkum úspěšně dokončen.")
            return result

        except Exception as e:
            self.logger.error(f"❌ Chyba během zdravotního výzkumu: {e}")
            raise

    def _validate_and_optimize_query(self, query: ResearchQuery) -> ResearchQuery:
        """Validace a optimalizace dotazu pro token efficiency"""
        if not query.query.strip():
            raise ValueError("Dotaz nemůže být prázdný")

        # Optimalizace dotazu pro lepší výsledky
        optimized_query_text = self._optimize_search_query(query.query)

        # Kontrola token budgetu
        depth_config = RESEARCH_DEPTHS.get(query.depth, RESEARCH_DEPTHS['medium'])

        # Aktualizace query s optimalizovanými parametry
        query.query = optimized_query_text
        query.analysis_type = depth_config['analysis_type']

        if query.max_results > depth_config['max_sources']:
            query.max_results = depth_config['max_sources']
            self.logger.info(f"📊 Snížen počet zdrojů na {query.max_results} pro {query.depth} research")

        self.logger.info(f"🎯 Optimalizovaný dotaz: '{optimized_query_text}' | Budget: {query.token_budget} tokenů")

        return query

    def _optimize_search_query(self, query: str) -> str:
        """Optimalizace vyhledávacího dotazu"""
        # Základní čistění a rozšíření dotazu
        query = query.strip().lower()

        # Přidání synoným pro lepší pokrytí (můžeme použít i Gemini pro toto)
        query_expansion = {
            'ai': 'artificial intelligence machine learning',
            'ml': 'machine learning artificial intelligence',
            'crypto': 'cryptocurrency blockchain bitcoin',
            'covid': 'coronavirus pandemic covid-19'
        }

        for abbrev, expansion in query_expansion.items():
            if abbrev in query:
                query = query.replace(abbrev, expansion)

        return query

    async def _gather_data_efficiently(self, query: ResearchQuery) -> Dict[str, Any]:
        """Efektivní shromáždění dat s ohledem na token budget"""
        # Rozdělení token budgetu: 70% pro analýzu, 30% pro scraping overhead
        scraping_budget = int(query.token_budget * 0.3)
        analysis_budget = int(query.token_budget * 0.7)

        self.logger.info(f"📊 Scraping budget: {scraping_budget}, Analysis budget: {analysis_budget} tokenů")

        # Parallelní scraping s omezením
        tasks = []
        sources_per_type = max(1, query.max_results // len(query.sources))

        # Tady by byly skutečné scrapery, pro demo používám mock data
        gathered_data = await self._mock_efficient_scraping(query, sources_per_type)

        # Prefiltrování dat podle relevance před AI analýzou
        filtered_data = self._prefilter_sources_by_relevance(gathered_data, query.query)

        return filtered_data

    async def _mock_efficient_scraping(self, query: ResearchQuery, sources_per_type: int) -> Dict[str, Any]:
        """Mock efektivní scraping - v produkci by zde byly skutečné scrapery"""
        mock_data = {
            'web': [
                {
                    'title': f'Web Article about {query.query}',
                    'content': f'This is a comprehensive article discussing {query.query} and its implications...' * 10,
                    'source': 'TechCrunch',
                    'url': 'https://techcrunch.com/mock',
                    'type': 'web'
                }
            ] * min(sources_per_type, 10),

            'academic': [
                {
                    'title': f'Academic Research on {query.query}',
                    'abstract': f'This study examines {query.query} through comprehensive analysis...' * 5,
                    'authors': ['Dr. Smith', 'Dr. Johnson'],
                    'source': 'arXiv',
                    'citation_count': 25,
                    'type': 'academic_paper'
                }
            ] * min(sources_per_type, 8)
        }

        return mock_data

    def _prefilter_sources_by_relevance(self, data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Prefiltrování zdrojů podle relevance před AI analýzou"""
        query_words = set(query.lower().split())
        filtered_data = {}

        for source_type, sources in data.items():
            scored_sources = []

            for source in sources:
                # Rychlé skórování relevance
                title = source.get('title', '').lower()
                content = (source.get('content', '') + ' ' + source.get('abstract', '')).lower()

                title_matches = sum(1 for word in query_words if word in title)
                content_matches = sum(1 for word in query_words if word in content[:1000])  # Pouze začátek

                relevance_score = (title_matches * 2) + content_matches

                if relevance_score > 0:  # Pouze relevantní zdroje
                    scored_sources.append((source, relevance_score))

            # Seřazení podle relevance a výběr nejlepších
            scored_sources.sort(key=lambda x: x[1], reverse=True)
            filtered_data[source_type] = [source for source, score in scored_sources[:15]]  # Top 15 per type

        total_filtered = sum(len(sources) for sources in filtered_data.values())
        self.logger.info(f"🔍 Prefiltrováno {total_filtered} relevantních zdrojů")

        return filtered_data

    async def _analyze_data_with_gemini(self, raw_data: Dict[str, Any], query: ResearchQuery) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Token-optimalizovaná analýza dat pomocí Gemini"""
        if not self.gemini_manager:
            self.logger.warning("Gemini manager není dostupný, používám fallback analýzu")
            return await self._fallback_analysis(raw_data), {"estimated_cost_usd": 0}

        # Příprava textů pro analýzu
        all_texts = []
        source_metadata = []

        for source_type, sources in raw_data.items():
            for source in sources:
                text = f"{source.get('title', '')} {source.get('content', '')} {source.get('abstract', '')}"
                all_texts.append(text)
                source_metadata.append({
                    'type': source_type,
                    'source': source.get('source', ''),
                    'url': source.get('url', '')
                })

        if not all_texts:
            return {}, {"estimated_cost_usd": 0}

        # Odhad nákladů před zpracováním
        cost_estimate = self.gemini_manager.estimate_cost(all_texts, query.analysis_type)
        estimated_cost = cost_estimate['estimated_cost_usd']

        self.logger.info(f"💰 Odhadované náklady: ${estimated_cost:.4f} USD")

        # Kontrola cost limitu
        if ENABLE_COST_TRACKING:
            remaining_budget = DAILY_COST_LIMIT - self.daily_cost_tracker
            if estimated_cost > remaining_budget:
                self.logger.warning(f"⚠️ Estimated cost ${estimated_cost:.4f} exceeds remaining budget ${remaining_budget:.4f}")
                # Snížení počtu textů pro zůstání v budgetu
                max_texts = int(len(all_texts) * (remaining_budget / estimated_cost) * 0.8)  # 80% safety margin
                all_texts = all_texts[:max_texts]
                source_metadata = source_metadata[:max_texts]
                self.logger.info(f"📊 Snížen počet analyzovaných zdrojů na {len(all_texts)}")

        # Gemini batch analýza
        try:
            analysis_results = await self.gemini_manager.analyze_batch_efficiently(
                all_texts, query.analysis_type
            )

            # Spojení výsledků s metadata
            processed_data = {
                'analysis_results': analysis_results,
                'source_metadata': source_metadata,
                'analysis_type': query.analysis_type,
                'total_analyzed': len(all_texts)
            }

            # Master shrnutí celého researche
            if len(all_texts) > 5:  # Pouze pokud máme dostatek dat
                master_summary = await self.gemini_manager.smart_summarize_research(
                    [dict(zip(['title', 'content'], [text[:200], text[200:]])) for text in all_texts[:20]],
                    query.query
                )
                processed_data['master_summary'] = master_summary

            return processed_data, cost_estimate

        except Exception as e:
            self.logger.error(f"Chyba při Gemini analýze: {e}")
            return await self._fallback_analysis(raw_data), {"estimated_cost_usd": 0}

    async def _fallback_analysis(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analýza bez AI při nedostupnosti Gemini"""
        # Použití základního text analyzeru
        from text_analyzer import DataAnalyzer

        analyzer = DataAnalyzer()
        return await analyzer.analyze_research_data(raw_data)

    async def _generate_optimized_report(self, query: ResearchQuery, processed_data: Dict[str, Any]) -> ResearchResult:
        """Generování optimalizovaného reportu"""
        timestamp = datetime.now()

        # Extrakce informací z processed_data
        analysis_results = processed_data.get('analysis_results', [])
        master_summary = processed_data.get('master_summary', {})

        # Sestavení finálního shrnutí
        if master_summary:
            summary = master_summary.get('executive_summary', 'Shrnutí není k dispozici')
            key_findings = master_summary.get('key_findings', [])
            confidence_score = master_summary.get('confidence_score', 0.5)
        else:
            # Fallback shrnutí z individuálních analýz
            summaries = [result.get('summary', '') for result in analysis_results if result.get('summary')]
            summary = ' '.join(summaries[:3]) if summaries else 'Analýza nedostupná'
            key_findings = ['Fallback analýza použita']
            confidence_score = 0.3

        # Metadata
        total_sources = processed_data.get('total_analyzed', 0)
        tokens_used = processed_data.get('tokens_used_estimated', 0)

        # Uložení detailního reportu
        report_path = self._save_detailed_report(query, processed_data, timestamp)

        return ResearchResult(
            query=query.query,
            timestamp=timestamp,
            sources_found=total_sources,
            summary=summary,
            key_findings=key_findings,
            sources=analysis_results,
            confidence_score=confidence_score,
            full_report_path=report_path,
            tokens_used=tokens_used
        )

    def _save_detailed_report(self, query: ResearchQuery, data: Dict[str, Any], timestamp: datetime) -> str:
        """Uložení detailního reportu"""
        filename = f"report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        report_path = REPORTS_DIR / filename

        detailed_report = {
            "query": query.__dict__,
            "timestamp": timestamp.isoformat(),
            "data": data
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, ensure_ascii=False, indent=2, default=str)

        return str(report_path)

    async def _save_results(self, result: ResearchResult):
        """Skutečné uložení výsledků do databáze"""
        try:
            from database_manager import DatabaseManager

            db_manager = DatabaseManager()
            await db_manager.initialize()

            # Uložení hlavního dotazu
            query_id = await db_manager.save_research_query(
                result.query,
                {
                    "timestamp": result.timestamp.isoformat(),
                    "sources_found": result.sources_found,
                    "processing_time": result.processing_time,
                    "tokens_used": result.tokens_used
                }
            )

            # Uložení zdrojů
            if result.sources:
                await db_manager.save_sources(query_id, result.sources)

            # Uložení analýzy
            analysis_data = {
                "summary": result.summary,
                "key_findings": result.key_findings,
                "confidence_score": result.confidence_score,
                "cost_info": result.cost_info
            }
            await db_manager.save_analysis(query_id, "comprehensive", analysis_data)

            # Uložení reportu
            if result.full_report_path:
                await db_manager.save_report(query_id, "detailed", result.summary, result.full_report_path)

            # Aktualizace stavu dotazu
            await db_manager.update_query_status(
                query_id,
                "completed",
                result.sources_found,
                result.confidence_score
            )

            self.logger.info(f"✅ Výsledky úspěšně uloženy do databáze (ID: {query_id})")

        except Exception as e:
            self.logger.error(f"❌ Chyba při ukládání do databáze: {e}")
            # Pokračujeme v běhu i při chybě ukládání

class FastResearchEngine(ResearchEngine):
    """Ultrarychlý research engine pro soukromé použití"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gemini_manager = None
        self._fast_setup()

    def _fast_setup(self):
        """Rychlá inicializace bez zbytečností"""
        if GEMINI_API_KEY:
            from gemini_manager import FastGeminiManager
            self.gemini_manager = FastGeminiManager(GEMINI_API_KEY)

    @cache_result(ttl_hours=6)  # Cache celých research výsledků
    async def fast_research(self, query: ResearchQuery) -> ResearchResult:
        """Ultrarychlý research workflow"""
        start_time = datetime.now()

        try:
            # 1. Rychlá optimalizace dotazu
            optimized_query = fast_text_processor.optimize_for_tokens(query.query, MAX_TOKEN_LIMIT)

            # 2. Paralelní scraping bez čekání
            scraped_data = await self._parallel_fast_scraping(optimized_query, query.max_results)

            # 3. Rychlá AI analýza (pokud je Gemini dostupný)
            if self.gemini_manager and scraped_data:
                texts = [item.get('abstract', item.get('title', ''))[:1000] for item in scraped_data[:BATCH_SIZE]]
                analysis = await self.gemini_manager.analyze_batch_efficiently(texts, "quick")
            else:
                analysis = []

            # 4. Rychlé sestavení výsledku
            processing_time = (datetime.now() - start_time).total_seconds()

            # Extrakce klíčových informací
            all_text = ' '.join([item.get('abstract', item.get('title', '')) for item in scraped_data])
            keywords = fast_text_processor.extract_keywords(all_text, 10)
            summary = fast_text_processor.optimize_for_tokens(all_text, 500) if all_text else "Žádný obsah k analýze"

            result = ResearchResult(
                query=optimized_query,
                timestamp=start_time,
                sources_found=len(scraped_data),
                summary=summary,
                key_findings=[kw[0] for kw in keywords[:5]],
                sources=analysis or scraped_data,
                confidence_score=0.8 if analysis else 0.6,
                full_report_path="",
                cost_info={"estimated_cost_usd": len(texts) * 0.001 if analysis else 0},
                processing_time=processing_time
            )

            return result

        except Exception as e:
            self.logger.error(f"Fast research failed: {e}")
            raise

    async def _parallel_fast_scraping(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Paralelní rychlé scrapování"""
        from academic_scraper import GoogleScholarScraper, SemanticScholarScraper

        # Mock query object
        class MockQuery:
            def __init__(self, q, max_r):
                self.query = q
                self.max_results = min(max_r, 20)  # Limit pro rychlost

        mock_query = MockQuery(query, max_results)

        # Paralelní spuštění scraperů
        tasks = []

        scholar_scraper = GoogleScholarScraper()
        tasks.append(asyncio.create_task(scholar_scraper.scrape(mock_query)))

        semantic_scraper = SemanticScholarScraper()
        tasks.append(asyncio.create_task(semantic_scraper.scrape(mock_query)))

        # Čekej maximálně 10 sekund
        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10)

            # Slij výsledky
            all_results = []
            for result in results:
                if isinstance(result, list):
                    all_results.extend(result)

            return all_results[:max_results]

        except asyncio.TimeoutError:
            self.logger.warning("Fast scraping timeout - returning empty results")
            return []

class UltraCheapResearchEngine(ResearchEngine):
    """Maximálně úsporný research engine - cíl: být levnější než Perplexity"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gemini_manager = None
        self.daily_cost = 0.0
        self._cheap_setup()

    def _cheap_setup(self):
        """Úsporná inicializace"""
        if GEMINI_API_KEY:
            from gemini_manager import UltraCheapGeminiManager
            self.gemini_manager = UltraCheapGeminiManager(GEMINI_API_KEY)

    @cache_result(ttl_hours=72)  # 3denní cache pro maximální úspory
    async def ultra_cheap_research(self, query: ResearchQuery) -> ResearchResult:
        """Maximálně úsporný research workflow"""
        start_time = datetime.now()

        try:
            # 1. Optimalizace dotazu pro úspory
            optimized_query = cost_optimizer.optimize_query_for_cost(query.query)
            optimized_query = fast_text_processor.optimize_for_tokens(optimized_query, 1000)  # Menší limit

            # 2. Kontrola denního rozpočtu
            if self.daily_cost >= DAILY_COST_LIMIT:
                self.logger.warning("💰 Denní rozpočet vyčerpán - používám pouze cache a free zdroje")
                return await self._free_only_research(optimized_query, start_time)

            # 3. Cache-first scraping (preferuj cache)
            scraped_data = await self._cache_first_scraping(optimized_query, query.max_results)

            # 4. Minimální AI analýza pouze pokud je nutná
            if self.gemini_manager and scraped_data and len(scraped_data) > 5:
                # Pouze pro větší množství dat, jinak fallback
                texts = [item.get('abstract', item.get('title', ''))[:300] for item in scraped_data[:5]]  # Max 5 textů
                analysis = await self.gemini_manager.analyze_batch_cheaply(texts, "quick")
            else:
                analysis = []

            # 5. Sestavení výsledku s minimálními náklady
            result = await self._build_cheap_result(optimized_query, scraped_data, analysis, start_time)

            # 6. Tracking nákladů
            estimated_cost = 0.005 if analysis else 0.0  # Odhad nákladů
            self.daily_cost += estimated_cost
            result.cost_info = {"estimated_cost_usd": estimated_cost}

            return result

        except Exception as e:
            self.logger.error(f"Ultra cheap research failed: {e}")
            return await self._free_only_research(query.query, start_time)

    async def _cache_first_scraping(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Scraping s prioritou cache"""
        # Zkus nejdřív cache
        cache_key = f"scraping:{query}:{max_results}"
        cached_results = high_perf_cache.get(cache_key)
        if cached_results:
            self.logger.info("🎯 Scraping výsledky z cache - 0 nákladů!")
            return cached_results

        # Pokud není cache, použij free scraping
        from academic_scraper import SemanticScholarScraper

        class MockQuery:
            def __init__(self, q, max_r):
                self.query = q
                self.max_results = min(max_r, 15)  # Limit pro úsporu

        mock_query = MockQuery(query, max_results)

        try:
            # Pouze Semantic Scholar (free API)
            semantic_scraper = SemanticScholarScraper()
            results = await asyncio.wait_for(semantic_scraper.scrape(mock_query), timeout=8)

            # Ulož do cache
            high_perf_cache.set(cache_key, results, 72)  # 3denní cache

            return results[:15]  # Max 15 výsledků

        except Exception as e:
            self.logger.warning(f"Free scraping failed: {e}")
            return []

    async def _free_only_research(self, query: str, start_time: datetime) -> ResearchResult:
        """Research pouze s free zdroji"""
        # Použij pouze cache a lokální analýzu
        all_text = f"Research query: {query}. Using local analysis due to budget constraints."
        keywords = fast_text_processor.extract_keywords(all_text, 5)

        processing_time = (datetime.now() - start_time).total_seconds()

        return ResearchResult(
            query=query,
            timestamp=start_time,
            sources_found=0,
            summary=f"Lokální analýza dotazu: {query}. Pro detailní výsledky použijte zítra (nový denní rozpočet).",
            key_findings=[kw[0] for kw in keywords],
            sources=[{"source": "local", "type": "budget_limited"}],
            confidence_score=0.3,
            full_report_path="",
            cost_info={"estimated_cost_usd": 0.0},
            processing_time=processing_time
        )

    async def _build_cheap_result(self, query: str, scraped_data: List, analysis: List, start_time: datetime) -> ResearchResult:
        """Sestavení výsledku s minimálními náklady"""
        processing_time = (datetime.now() - start_time).total_seconds()

        # Lokální zpracování pro úsporu
        all_text = ' '.join([item.get('abstract', item.get('title', '')) for item in scraped_data])
        keywords = fast_text_processor.extract_keywords(all_text, 8)

        # Inteligentní shrnutí
        if analysis:
            summary = f"AI analýza {len(analysis)} zdrojů: " + '; '.join([a.get('summary', a.get('content', ''))[:100] for a in analysis[:3]])
        else:
            summary = fast_text_processor.optimize_for_tokens(all_text, 300) if all_text else "Lokální analýza dokončena."

        return ResearchResult(
            query=query,
            timestamp=start_time,
            sources_found=len(scraped_data),
            summary=summary,
            key_findings=[kw[0] for kw in keywords[:5]],
            sources=analysis or scraped_data[:10],
            confidence_score=0.8 if analysis else 0.6,
            full_report_path="",
            processing_time=processing_time
        )

    def get_savings_report(self) -> Dict[str, Any]:
        """Report úspor oproti komerčním službám"""
        return cost_optimizer.get_cost_savings_report()

if __name__ == "__main__":
    # Test základní funkcionality
    engine = ResearchEngine()

    query = ResearchQuery(
        query="artificial intelligence trends 2024",
        sources=["web", "academic"],
        depth="medium"
    )

    # result = asyncio.run(engine.conduct_research(query))
    print("Research Engine úspěšně vytvořen!")
