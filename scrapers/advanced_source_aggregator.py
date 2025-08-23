Advanced Source Aggregator - Rozšířený systém pro sběr dat z více zdrojů
Optimalizováno pro biohacking a peptidový výzkum s maximální soukromostí
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import re
from urllib.parse import urlencode, quote
import time
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SourceConfig:
    """Konfigurace zdroje dat"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    rate_limit: float = 1.0  # requests per second
    reliability_score: int = 5  # 1-10
    specialties: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, str] = field(default_factory=dict)

@dataclass
class SearchResult:
    """Výsledek vyhledávání"""
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float
    publication_date: Optional[datetime] = None
    authors: List[str] = field(default_factory=list)
    citations: int = 0
    study_type: Optional[str] = None
    full_text: Optional[str] = None

class AdvancedSourceAggregator:
    """Pokročilý agregátor zdrojů pro biohacking výzkum"""

    def __init__(self):
        self.sources = self._initialize_sources()
        self.session = None
        self.cache_dir = Path("data/source_cache")
        self.cache_dir.mkdir(exist_ok=True)

        # Rate limiting
        self.last_request_times = {}

        # Advanced search features
        self.search_operators = {
            "exact_phrase": '"{}"',
            "exclude": '-{}',
            "site_specific": 'site:{}',
            "filetype": 'filetype:{}',
            "intitle": 'intitle:{}',
            "inurl": 'inurl:{}',
            "date_range": 'after:{} before:{}'
        }

    def _initialize_sources(self) -> Dict[str, SourceConfig]:
        """Inicializace zdrojů dat"""
        return {
            # Academic sources
            "pubmed": SourceConfig(
                name="PubMed",
                base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
                rate_limit=3.0,
                reliability_score=10,
                specialties=["medical", "clinical_trials", "biochemistry"],
                params={"db": "pubmed", "retmode": "json"}
            ),

            "google_scholar": SourceConfig(
                name="Google Scholar",
                base_url="https://scholar.google.com/scholar",
                rate_limit=0.5,  # Conservative to avoid blocking
                reliability_score=8,
                specialties=["academic", "citations", "preprints"],
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
            ),

            "clinical_trials": SourceConfig(
                name="ClinicalTrials.gov",
                base_url="https://clinicaltrials.gov/api/query/",
                rate_limit=2.0,
                reliability_score=9,
                specialties=["clinical_trials", "safety", "dosing"],
                params={"fmt": "json"}
            ),

            # Specialized biohacking sources
            "examine": SourceConfig(
                name="Examine.com",
                base_url="https://examine.com/",
                rate_limit=1.0,
                reliability_score=8,
                specialties=["supplements", "evidence_summary", "dosing"],
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
            ),

            "selfhacked": SourceConfig(
                name="SelfHacked",
                base_url="https://selfhacked.com/",
                rate_limit=1.0,
                reliability_score=6,
                specialties=["biohacking", "personal_optimization", "mechanisms"],
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
            ),

            "longevity_research": SourceConfig(
                name="Fight Aging!",
                base_url="https://www.fightaging.org/",
                rate_limit=1.0,
                reliability_score=7,
                specialties=["longevity", "aging_research", "interventions"]
            ),

            # Community sources
            "reddit_peptides": SourceConfig(
                name="Reddit Peptides",
                base_url="https://www.reddit.com/r/Peptides/",
                rate_limit=0.5,
                reliability_score=4,
                specialties=["user_experiences", "practical_protocols", "side_effects"],
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
            ),

            "reddit_nootropics": SourceConfig(
                name="Reddit Nootropics",
                base_url="https://www.reddit.com/r/nootropics/",
                rate_limit=0.5,
                reliability_score=4,
                specialties=["cognitive_enhancement", "stacking", "experiences"]
            ),

            "longecity": SourceConfig(
                name="LongeCity",
                base_url="https://www.longecity.org/",
                rate_limit=1.0,
                reliability_score=5,
                specialties=["longevity", "anti_aging", "experimental_protocols"]
            ),

            # Regulatory and safety
            "fda_orange_book": SourceConfig(
                name="FDA Orange Book",
                base_url="https://www.accessdata.fda.gov/",
                rate_limit=2.0,
                reliability_score=9,
                specialties=["regulatory", "safety", "approved_drugs"]
            ),

            "ema_database": SourceConfig(
                name="EMA Database",
                base_url="https://www.ema.europa.eu/",
                rate_limit=1.0,
                reliability_score=9,
                specialties=["european_regulation", "safety", "clinical_data"]
            ),

            # Specialized databases
            "drugbank": SourceConfig(
                name="DrugBank",
                base_url="https://go.drugbank.com/",
                rate_limit=1.0,
                reliability_score=8,
                specialties=["drug_interactions", "pharmacology", "targets"]
            ),

            "bindingdb": SourceConfig(
                name="BindingDB",
                base_url="https://www.bindingdb.org/",
                rate_limit=1.0,
                reliability_score=7,
                specialties=["receptor_binding", "affinity", "selectivity"]
            )
        }

    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=10,  # M1 optimized
            limit_per_host=3,
            ttl_dns_cache=300,
            use_dns_cache=True
        )

        timeout = aiohttp.ClientTimeout(total=30, connect=10)

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Accept-Encoding": "gzip, deflate"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def multi_source_search(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_results_per_source: int = 5,
        include_community: bool = True,
        evidence_level: str = "all"  # high, medium, all
    ) -> Dict[str, List[SearchResult]]:
        """Multi-source vyhledávání s pokročilými filtry"""

        if sources is None:
            sources = self._select_optimal_sources(query, include_community, evidence_level)

        logger.info(f"🔍 Searching across {len(sources)} sources for: {query}")

        # Parallel search across sources
        tasks = []
        for source_name in sources:
            if source_name in self.sources:
                task = self._search_source(
                    source_name,
                    query,
                    max_results_per_source
                )
                tasks.append((source_name, task))

        # Execute searches with rate limiting
        results = {}
        for source_name, task in tasks:
            try:
                await self._rate_limit(source_name)
                source_results = await task
                results[source_name] = source_results
                logger.info(f"✅ {source_name}: {len(source_results)} results")
            except Exception as e:
                logger.warning(f"❌ {source_name} search failed: {e}")
                results[source_name] = []

        return results

    def _select_optimal_sources(
        self,
        query: str,
        include_community: bool,
        evidence_level: str
    ) -> List[str]:
        """Výběr optimálních zdrojů na základě dotazu"""

        # Keyword-based source selection
        peptide_keywords = ["peptide", "protein", "amino", "growth hormone", "igf"]
        nootropic_keywords = ["nootropic", "cognitive", "memory", "focus", "brain"]
        safety_keywords = ["safety", "side effect", "adverse", "interaction", "contraindication"]
        dosing_keywords = ["dose", "dosing", "protocol", "administration", "mg", "mcg"]

        selected_sources = []

        # Always include high-reliability academic sources
        if evidence_level in ["high", "all"]:
            selected_sources.extend(["pubmed", "clinical_trials"])

        # Peptide-specific sources
        if any(keyword in query.lower() for keyword in peptide_keywords):
            selected_sources.extend(["examine", "reddit_peptides"])

        # Nootropic-specific sources
        if any(keyword in query.lower() for keyword in nootropic_keywords):
            selected_sources.extend(["examine", "reddit_nootropics"])

        # Safety-focused sources
        if any(keyword in query.lower() for keyword in safety_keywords):
            selected_sources.extend(["fda_orange_book", "drugbank"])

        # Dosing protocol sources
        if any(keyword in query.lower() for keyword in dosing_keywords):
            selected_sources.extend(["examine", "clinical_trials"])

        # Longevity research
        if "longevity" in query.lower() or "aging" in query.lower():
            selected_sources.extend(["longevity_research", "longecity"])

        # Medium evidence sources
        if evidence_level in ["medium", "all"]:
            selected_sources.extend(["google_scholar", "selfhacked"])

        # Community sources
        if include_community and evidence_level == "all":
            selected_sources.extend(["reddit_peptides", "reddit_nootropics", "longecity"])

        # Remove duplicates and limit for M1 performance
        selected_sources = list(dict.fromkeys(selected_sources))[:8]

        return selected_sources

    async def _search_source(
        self,
        source_name: str,
        query: str,
        max_results: int
    ) -> List[SearchResult]:
        """Vyhledávání v konkrétním zdroji"""

        source = self.sources[source_name]

        # Check cache first
        cache_key = self._generate_cache_key(source_name, query, max_results)
        cached_result = await self._get_cached_results(cache_key)
        if cached_result:
            return cached_result

        try:
            if source_name == "pubmed":
                results = await self._search_pubmed(query, max_results)
            elif source_name == "google_scholar":
                results = await self._search_google_scholar(query, max_results)
            elif source_name == "clinical_trials":
                results = await self._search_clinical_trials(query, max_results)
            elif "reddit" in source_name:
                results = await self._search_reddit(source_name, query, max_results)
            else:
                results = await self._search_generic_web(source, query, max_results)

            # Cache results
            await self._cache_results(cache_key, results)
            return results

        except Exception as e:
            logger.error(f"Search failed for {source_name}: {e}")
            return []

    async def _search_pubmed(self, query: str, max_results: int) -> List[SearchResult]:
        """PubMed API vyhledávání"""
        # Enhanced query for better peptide/biohacking results
        enhanced_query = self._enhance_biomedical_query(query)

        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": enhanced_query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "field": "title/abstract"
        }

        async with self.session.get(search_url, params=search_params) as response:
            if response.status != 200:
                return []

            data = await response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return []

        # Fetch article details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json"
        }

        async with self.session.get(fetch_url, params=fetch_params) as response:
            if response.status != 200:
                return []

            data = await response.json()
            results = []

            for pmid, article_data in data.get("result", {}).items():
                if pmid == "uids":
                    continue

                try:
                    result = SearchResult(
                        title=article_data.get("title", ""),
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        snippet=article_data.get("summary", ""),
                        source="PubMed",
                        relevance_score=0.9,  # High for PubMed
                        publication_date=self._parse_pubmed_date(article_data.get("pubdate")),
                        authors=article_data.get("authors", []),
                        study_type=self._detect_study_type(article_data.get("title", ""))
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to parse PubMed article {pmid}: {e}")

            return results

    def _enhance_biomedical_query(self, query: str) -> str:
        """Vylepšení dotazu pro biomedicínské databáze"""
        # Add relevant MeSH terms and synonyms
        enhancements = {
            "peptide": "peptide OR peptides OR polypeptide",
            "growth hormone": "(growth hormone OR somatotropin OR GH OR hGH)",
            "nootropic": "(nootropic OR cognitive enhancer OR smart drug)",
            "biohacking": "(biohacking OR life extension OR longevity OR anti-aging)",
            "supplement": "(dietary supplement OR nutraceutical OR functional food)"
        }

        enhanced = query
        for term, replacement in enhancements.items():
            if term in query.lower():
                enhanced = enhanced.replace(term, replacement)

        return enhanced

    async def _rate_limit(self, source_name: str):
        """Rate limiting pro zdroje"""
        source = self.sources[source_name]
        now = time.time()

        if source_name in self.last_request_times:
            time_since_last = now - self.last_request_times[source_name]
            min_interval = 1.0 / source.rate_limit

            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                await asyncio.sleep(sleep_time)

        self.last_request_times[source_name] = time.time()

    def _generate_cache_key(self, source: str, query: str, max_results: int) -> str:
        """Generování cache klíče"""
        content = f"{source}:{query}:{max_results}"
        return hashlib.md5(content.encode()).hexdigest()

    async def _get_cached_results(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Načtení z cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                # Check if cache is fresh (24 hours)
                if (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days < 1:
                    with open(cache_file) as f:
                        data = json.load(f)
                        return [SearchResult(**item) for item in data]
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")

        return None

    async def _cache_results(self, cache_key: str, results: List[SearchResult]):
        """Uložení do cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            serializable_results = []
            for result in results:
                result_dict = result.__dict__.copy()
                if result_dict['publication_date']:
                    result_dict['publication_date'] = result_dict['publication_date'].isoformat()
                serializable_results.append(result_dict)

            with open(cache_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

# Export hlavních tříd
__all__ = ['AdvancedSourceAggregator', 'SourceConfig', 'SearchResult']
