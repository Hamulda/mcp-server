Academic Scraper - Optimized version with unified config integration
Implementuje requests.Session, retry logic, robust error handling
"""

import asyncio
import logging
import random
import time
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from urllib.parse import quote_plus

# Prioritizuj unified config
try:
    from unified_config import get_config
    UNIFIED_CONFIG_AVAILABLE = True
    print("✅ Using unified configuration")
except ImportError:
    UNIFIED_CONFIG_AVAILABLE = False
    get_config = None  # Define fallback
    print("⚠️  Unified config not available, using legacy")

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

@dataclass
class ScrapingResult:
    """Standardizovaný výsledek scrapingu s detailními informacemi"""
    source: str
    query: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    response_time: Optional[float] = None
    cached: bool = False
    status_code: Optional[int] = None
    rate_limited: bool = False

class CircuitBreaker:
    """Jednoduchý circuit breaker pro ochranu proti zahlcení zdroje"""
    def __init__(self, max_failures=3, reset_timeout=60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = {}
        self.open_until = {}

    def record_failure(self, source):
        now = time.time()
        self.failures[source] = self.failures.get(source, 0) + 1
        if self.failures[source] >= self.max_failures:
            self.open_until[source] = now + self.reset_timeout

    def is_open(self, source):
        now = time.time()
        if source in self.open_until and now < self.open_until[source]:
            return True
        if source in self.open_until and now >= self.open_until[source]:
            del self.open_until[source]
            self.failures[source] = 0
        return False

    def reset(self, source):
        self.failures[source] = 0
        if source in self.open_until:
            del self.open_until[source]

class EnhancedRateLimiter:
    """Pokročilý rate limiter s exponential backoff"""

    def __init__(self):
        self._last_request_times = {}
        self._consecutive_429s = {}
        self.logger = logging.getLogger(__name__)

        if UNIFIED_CONFIG_AVAILABLE:
            self.config = get_config()
        else:
            self.config = None
    
    async def wait_if_needed(self, source: str, base_delay: float = None):
        """Čeká s exponential backoff při 429 errors"""
        current_time = time.time()
        last_time = self._last_request_times.get(source, 0)
        
        # Použij konfigurovaný delay nebo default
        if base_delay is None and self.config:
            source_config = self.config.get_source_config(source)
            base_delay = source_config.rate_limit_delay if source_config else 1.0
        else:
            base_delay = base_delay or 1.0

        # Exponential backoff při opakovaných 429 chybách
        consecutive_429s = self._consecutive_429s.get(source, 0)
        if consecutive_429s > 0:
            backoff_multiplier = 2 ** min(consecutive_429s, 5)  # Max 32x delay
            # Přidej jitter pro rozptyl synchronizace
            jitter = random.uniform(0.5, 1.5)
            base_delay *= backoff_multiplier * jitter
            self.logger.warning(f"Exponential backoff for {source}: {base_delay:.2f}s (attempt {consecutive_429s + 1})")

        time_since_last = current_time - last_time
        if time_since_last < base_delay:
            wait_time = base_delay - time_since_last
            self.logger.debug(f"Rate limiting {source}: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        
        self._last_request_times[source] = time.time()

    def record_429_error(self, source: str):
        """Zaznamená 429 error pro exponential backoff"""
        self._consecutive_429s[source] = self._consecutive_429s.get(source, 0) + 1
        self.logger.warning(f"Rate limit hit for {source} ({self._consecutive_429s[source]} consecutive times)")

    def reset_429_count(self, source: str):
        """Resetuje 429 counter po úspěšném requestu"""
        if source in self._consecutive_429s:
            del self._consecutive_429s[source]

class EnhancedSessionManager:
    """Pokročilý session manager s retry strategií a connection pooling"""

    def __init__(self):
        self.session = None
        self.logger = logging.getLogger(__name__)
        self._setup_session()

    def _setup_session(self):
        """Nastaví session s pokročilou retry strategií a optimalizovaným poolingem"""
        self.session = requests.Session()

        # Získej konfiguraci
        if UNIFIED_CONFIG_AVAILABLE:
            config = get_config()
            max_retries = config.scraping.max_retries
            retry_delay = config.scraping.retry_delay
            user_agents = config.scraping.user_agents
            timeout = config.scraping.request_timeout
        else:
            # Fallback values
            max_retries = 3
            retry_delay = 1.0
            user_agents = ['Mozilla/5.0 (compatible; AcademicBot/1.0)']
            timeout = 30

        # Pokročilá retry strategie s exponential backoff
        retry_strategy = Retry(
            total=max_retries,
            read=max_retries,
            connect=max_retries,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            raise_on_status=False
        )

        # Optimalizované connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,  # Zvýšeno z 10 na 20
            pool_maxsize=50       # Zvýšeno z 20 na 50
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Optimalizované headers s kompresí
        self.session.headers.update({
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',  # Přidána brotli komprese
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'Pragma': 'no-cache'
        })

        self.default_timeout = timeout

    def get(self, url: str, **kwargs) -> requests.Response:
        """Enhanced GET s retry logic a error handling"""
        # Merguj s default timeout
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.default_timeout
        
        try:
            # Rotuj user agent pro každý request
            if UNIFIED_CONFIG_AVAILABLE:
                config = get_config()
                self.session.headers['User-Agent'] = random.choice(config.scraping.user_agents)
            
            response = self.session.get(url, **kwargs)
            # Oprava: ošetři případy kdy response.content může být Mock
            try:
                content_length = len(response.content) if hasattr(response.content, '__len__') else 0
                self.logger.debug(f"GET {url} -> {response.status_code} ({content_length} bytes)")
            except (TypeError, AttributeError):
                self.logger.debug(f"GET {url} -> {response.status_code}")
            return response
            
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout for {url}")
            raise
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Connection error for {url}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error for {url}: {e}")
            raise

    def close(self):
        """Cleanup session"""
        if self.session:
            self.session.close()

# Circuit breaker instance (globální pro všechny scrapery)
global circuit_breaker
circuit_breaker = CircuitBreaker()

# Global session manager instance
_session_manager = None

def get_session_manager() -> EnhancedSessionManager:
    """Singleton pattern pro session manager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = EnhancedSessionManager()
    return _session_manager

class BaseScraper(ABC):
    """Abstraktní base class pro všechny scrapery"""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.logger = logging.getLogger(f"{__name__}.{source_name}")
        self.session_manager = get_session_manager()
        self.rate_limiter = EnhancedRateLimiter()
        
        # Load source config
        if UNIFIED_CONFIG_AVAILABLE:
            self.config = get_config()
            self.source_config = self.config.get_source_config(source_name)
        else:
            self.config = None
            self.source_config = None

    @abstractmethod
    async def scrape(self, query: str) -> ScrapingResult:
        """Abstract method pro scraping"""
        pass

    def _create_result(self, query: str, success: bool, data: Dict[str, Any] = None, 
                      error: str = None, response_time: float = None, 
                      status_code: int = None, rate_limited: bool = False) -> ScrapingResult:
        """Helper pro vytvoření standardizovaného výsledku"""
        return ScrapingResult(
            source=self.source_name,
            query=query,
            success=success,
            data=data or {},
            error=error,
            response_time=response_time,
            status_code=status_code,
            rate_limited=rate_limited
        )

class WikipediaScraper(BaseScraper):
    """Optimalizovaný Wikipedia scraper s robustním error handling"""
    
    def __init__(self):
        super().__init__("wikipedia")
        
    async def scrape(self, query: str) -> ScrapingResult:
        """Scrape Wikipedia s kompletním error handling"""
        start_time = time.time()
        
        try:
            # Rate limiting
            await self.rate_limiter.wait_if_needed(self.source_name)
            
            # Circuit breaker check
            if circuit_breaker.is_open(self.source_name):
                return self._create_result(query, False, error="Circuit breaker open", response_time=0, status_code=503)

            # Construct URL
            search_url = f"https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': 5,
                'srprop': 'snippet|titlesnippet|size|timestamp'
            }
            
            # Make request - použije asyncio.to_thread pro neblokující volání
            response = await asyncio.to_thread(
                self.session_manager.get, search_url, params=params
            )
            response_time = time.time() - start_time
            
            # Handle rate limiting
            if response.status_code == 429:
                self.rate_limiter.record_429_error(self.source_name)
                return self._create_result(
                    query, False, error="Rate limited", 
                    response_time=response_time, status_code=429, rate_limited=True
                )
            
            # Handle other errors
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.reason}"
                self.logger.error(f"Wikipedia API error: {error_msg}")
                circuit_breaker.record_failure(self.source_name)
                return self._create_result(
                    query, False, error=error_msg,
                    response_time=response_time, status_code=response.status_code
                )
            
            # Parse JSON response
            try:
                data = response.json()
            except ValueError as e:
                self.logger.error(f"Invalid JSON response from Wikipedia: {e}")
                circuit_breaker.record_failure(self.source_name)
                return self._create_result(
                    query, False, error="Invalid JSON response",
                    response_time=response_time, status_code=response.status_code
                )
            
            # Extract results
            if 'query' not in data or 'search' not in data['query']:
                self.logger.warning(f"No search results in Wikipedia response for query: {query}")
                return self._create_result(
                    query, True, data={'articles': [], 'total_found': 0},
                    response_time=response_time, status_code=response.status_code
                )
            
            search_results = data['query']['search']
            articles = []
            
            for result in search_results:
                article = {
                    'title': result.get('title', ''),
                    'snippet': result.get('snippet', ''),
                    'url': f"https://en.wikipedia.org/wiki/{result.get('title', '').replace(' ', '_')}",
                    'size': result.get('size', 0),
                    'timestamp': result.get('timestamp', '')
                }
                articles.append(article)
            
            # Reset rate limit counter on success
            self.rate_limiter.reset_429_count(self.source_name)
            circuit_breaker.reset(self.source_name)

            return self._create_result(
                query, True,
                data={'articles': articles, 'total_found': len(articles)},
                response_time=response_time, status_code=response.status_code
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Wikipedia scraping failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            circuit_breaker.record_failure(self.source_name)
            return self._create_result(
                query, False, error=error_msg, response_time=response_time
            )

class PubMedScraper(BaseScraper):
    """Optimalizovaný PubMed scraper"""

    def __init__(self):
        super().__init__("pubmed")

    async def scrape(self, query: str) -> ScrapingResult:
        """Scrape PubMed s robustním error handling"""
        start_time = time.time()
        
        try:
            # Rate limiting
            await self.rate_limiter.wait_if_needed(self.source_name)
            
            # Circuit breaker check
            if circuit_breaker.is_open(self.source_name):
                return self._create_result(query, False, error="Circuit breaker open", response_time=0, status_code=503)

            # Construct URL
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': 10,
                'retmode': 'json',
                'sort': 'relevance'
            }
            
            # Make request - použije asyncio.to_thread pro neblokující volání
            response = await asyncio.to_thread(
                self.session_manager.get, search_url, params=params
            )
            response_time = time.time() - start_time
            
            if response.status_code == 429:
                self.rate_limiter.record_429_error(self.source_name)
                return self._create_result(
                    query, False, error="Rate limited",
                    response_time=response_time, status_code=429, rate_limited=True
                )
            
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.reason}"
                circuit_breaker.record_failure(self.source_name)
                return self._create_result(
                    query, False, error=error_msg,
                    response_time=response_time, status_code=response.status_code
                )
            
            # Parse response (support JSON and XML for robustness in tests)
            try:
                data = response.json()
                if isinstance(data, dict):
                    id_list = data.get('esearchresult', {}).get('idlist', [])
                    if not isinstance(id_list, list):
                        id_list = []
                else:
                    raise ValueError("Non-dict JSON")
            except Exception:
                # Fallback: attempt to parse XML structure
                id_list = []
                try:
                    import xml.etree.ElementTree as ET
                    xml_root = ET.fromstring(getattr(response, 'text', '') or '')
                    # Extract all <Id> under <IdList>
                    for id_node in xml_root.findall('.//IdList/Id'):
                        if id_node.text:
                            id_list.append(id_node.text.strip())
                except Exception:
                    id_list = []

            papers = [{'pmid': pmid, 'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"} for pmid in id_list]

            self.rate_limiter.reset_429_count(self.source_name)
            circuit_breaker.reset(self.source_name)
            return self._create_result(
                query, True,
                data={'papers': papers, 'total_found': len(papers)},
                response_time=response_time, status_code=response.status_code
            )

        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"PubMed scraping failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            circuit_breaker.record_failure(self.source_name)
            return self._create_result(
                query, False, error=error_msg, response_time=response_time
            )

class OpenAlexScraper(BaseScraper):
    """Scraper pro OpenAlex API"""
    def __init__(self):
        super().__init__("openalex")

    async def scrape(self, query: str) -> ScrapingResult:
        start_time = time.time()
        try:
            await self.rate_limiter.wait_if_needed(self.source_name)
            if circuit_breaker.is_open(self.source_name):
                return self._create_result(query, False, error="Circuit breaker open", response_time=0, status_code=503)
            # OpenAlex API endpoint
            search_url = f"https://api.openalex.org/works"
            params = {
                'search': query,
                'per_page': 5
            }
            response = await asyncio.to_thread(
                self.session_manager.get, search_url, params=params
            )
            response_time = time.time() - start_time
            if response.status_code == 429:
                self.rate_limiter.record_429_error(self.source_name)
                return self._create_result(query, False, error="Rate limited", response_time=response_time, status_code=429, rate_limited=True)
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.reason}"
                circuit_breaker.record_failure(self.source_name)
                return self._create_result(query, False, error=error_msg, response_time=response_time, status_code=response.status_code)
            try:
                data = response.json()
                results = data.get('results', [])
                works = []
                for item in results:
                    # Bezpečné získání URL s kontrolou None hodnot
                    primary_location = item.get('primary_location') or {}
                    source = primary_location.get('source') or {}
                    url = source.get('url', '') if isinstance(source, dict) else ''

                    work = {
                        'id': item.get('id', ''),
                        'title': item.get('title', ''),
                        'doi': item.get('doi', ''),
                        'url': url,
                        'publication_year': item.get('publication_year', ''),
                        'type': item.get('type', '')
                    }
                    works.append(work)
                self.rate_limiter.reset_429_count(self.source_name)
                circuit_breaker.reset(self.source_name)
                return self._create_result(query, True, data={'works': works, 'total_found': len(works)}, response_time=response_time, status_code=response.status_code)
            except ValueError as e:
                return self._create_result(query, False, error="Invalid JSON response", response_time=response_time, status_code=response.status_code)
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"OpenAlex scraping failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            circuit_breaker.record_failure(self.source_name)
            return self._create_result(query, False, error=error_msg, response_time=response_time)

# Rozšířené zdroje pro zdraví, peptidy, doplňky stravy a AI
class ClinicalTrialsScraper(BaseScraper):
    """Scraper pro ClinicalTrials.gov - oficiální databáze klinických studií"""

    def __init__(self):
        super().__init__("clinicaltrials")

    async def scrape(self, query: str) -> ScrapingResult:
        """Scrape ClinicalTrials.gov pro zdravotní research"""
        start_time = time.time()

        try:
            await self.rate_limiter.wait_if_needed(self.source_name, 2.0)  # Pomalejší rate limit

            if circuit_breaker.is_open(self.source_name):
                return self._create_result(query, False, error="Circuit breaker open")

            # ClinicalTrials.gov API
            search_url = "https://clinicaltrials.gov/api/v2/studies"
            params = {
                'query.term': query,
                'format': 'json',
                'pageSize': 10,
                'fields': 'NCTId,BriefTitle,Condition,InterventionName,Phase,OverallStatus,StudyType'
            }

            response = await asyncio.to_thread(
                self.session_manager.get, search_url, params=params, timeout=15
            )

            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                studies = data.get('studies', [])

                results = []
                for study in studies:
                    protocol = study.get('protocolSection', {})
                    identification = protocol.get('identificationModule', {})
                    conditions = protocol.get('conditionsModule', {})
                    interventions = protocol.get('armsInterventionsModule', {})

                    results.append({
                        'title': identification.get('briefTitle', 'No title'),
                        'nct_id': identification.get('nctId', ''),
                        'conditions': conditions.get('conditions', []),
                        'interventions': [i.get('name', '') for i in interventions.get('interventions', [])],
                        'phase': protocol.get('designModule', {}).get('phases', []),
                        'status': protocol.get('statusModule', {}).get('overallStatus', ''),
                        'url': f"https://clinicaltrials.gov/study/{identification.get('nctId', '')}",
                        'source': 'ClinicalTrials.gov',
                        'summary': identification.get('briefSummary', {}).get('textBlock', '')[:300]
                    })

                circuit_breaker.reset(self.source_name)
                return self._create_result(query, True, {'results': results}, response_time=response_time)
            else:
                return self._create_result(query, False, error=f"HTTP {response.status_code}", response_time=response_time, status_code=response.status_code)

        except Exception as e:
            circuit_breaker.record_failure(self.source_name)
            return self._create_result(query, False, error=str(e), response_time=time.time() - start_time)

class NIHReporterScraper(BaseScraper):
    """Scraper pro NIH RePORTER - databáze NIH grantů a výzkumů"""

    def __init__(self):
        super().__init__("nih_reporter")

    async def scrape(self, query: str) -> ScrapingResult:
        """Scrape NIH RePORTER pro biomedicínský research"""
        start_time = time.time()

        try:
            await self.rate_limiter.wait_if_needed(self.source_name, 1.5)

            if circuit_breaker.is_open(self.source_name):
                return self._create_result(query, False, error="Circuit breaker open")

            # NIH RePORTER API
            search_url = "https://api.reporter.nih.gov/v2/projects/search"
            payload = {
                "criteria": {
                    "advanced_text_search": {
                        "operator": "advanced",
                        "search_field": "projecttitle,abstract,terms",
                        "search_text": query
                    }
                },
                "limit": 10,
                "offset": 0
            }

            response = await asyncio.to_thread(
                self.session_manager.session.post, search_url, json=payload, timeout=15
            )

            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                projects = data.get('results', [])

                results = []
                for project in projects:
                    results.append({
                        'title': project.get('project_title', 'No title'),
                        'abstract': project.get('abstract_text', '')[:400],
                        'pi_name': project.get('contact_pi_name', ''),
                        'organization': project.get('organization', {}).get('org_name', ''),
                        'fiscal_year': project.get('fiscal_year', ''),
                        'award_amount': project.get('award_amount', 0),
                        'agency': project.get('agency_ic_admin', {}).get('name', ''),
                        'url': f"https://reporter.nih.gov/project-details/{project.get('appl_id', '')}",
                        'source': 'NIH RePORTER',
                        'summary': project.get('abstract_text', '')[:300]
                    })

                circuit_breaker.reset(self.source_name)
                return self._create_result(query, True, {'results': results}, response_time=response_time)
            else:
                return self._create_result(query, False, error=f"HTTP {response.status_code}", response_time=response_time, status_code=response.status_code)

        except Exception as e:
            circuit_breaker.record_failure(self.source_name)
            return self._create_result(query, False, error=str(e), response_time=time.time() - start_time)

class ArxivScraper(BaseScraper):
    """Scraper pro arXiv - preprint server pro AI a tech research"""

    def __init__(self):
        super().__init__("arxiv")

    async def scrape(self, query: str) -> ScrapingResult:
        """Scrape arXiv pro AI a tech research"""
        start_time = time.time()

        try:
            await self.rate_limiter.wait_if_needed(self.source_name, 3.0)  # arXiv má přísné limity

            if circuit_breaker.is_open(self.source_name):
                return self._create_result(query, False, error="Circuit breaker open")

            # arXiv API
            search_url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': 10,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }

            response = await asyncio.to_thread(
                self.session_manager.get, search_url, params=params, timeout=15
            )

            response_time = time.time() - start_time

            if response.status_code == 200:
                # Parse XML response
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)

                results = []
                entries = root.findall('{http://www.w3.org/2005/Atom}entry')

                for entry in entries:
                    title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
                    summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()[:400]
                    arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
                    published = entry.find('{http://www.w3.org/2005/Atom}published').text

                    # Get authors
                    authors = []
                    for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                        name = author.find('{http://www.w3.org/2005/Atom}name').text
                        authors.append(name)

                    # Get categories
                    categories = []
                    for category in entry.findall('{http://www.w3.org/2005/Atom}category'):
                        categories.append(category.get('term'))

                    results.append({
                        'title': title,
                        'summary': summary,
                        'arxiv_id': arxiv_id,
                        'authors': authors,
                        'categories': categories,
                        'published': published,
                        'url': f"https://arxiv.org/abs/{arxiv_id}",
                        'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                        'source': 'arXiv',
                        'abstract': summary
                    })

                circuit_breaker.reset(self.source_name)
                return self._create_result(query, True, {'results': results}, response_time=response_time)
            else:
                return self._create_result(query, False, error=f"HTTP {response.status_code}", response_time=response_time, status_code=response.status_code)

        except Exception as e:
            circuit_breaker.record_failure(self.source_name)
            return self._create_result(query, False, error=str(e), response_time=time.time() - start_time)

class FoodDataCentralScraper(BaseScraper):
    """Scraper pro USDA FoodData Central - databáze výživových hodnot a doplňků"""

    def __init__(self):
        super().__init__("fooddata_central")

    async def scrape(self, query: str) -> ScrapingResult:
        """Scrape FoodData Central pro výživu a doplňky"""
        start_time = time.time()

        try:
            await self.rate_limiter.wait_if_needed(self.source_name, 1.0)

            if circuit_breaker.is_open(self.source_name):
                return self._create_result(query, False, error="Circuit breaker open")

            # USDA FoodData Central API (vyžaduje API key - použijeme demo klíč)
            api_key = os.getenv('USDA_API_KEY', 'DEMO_KEY')
            search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"

            params = {
                'query': query,
                'pageSize': 10,
                'api_key': api_key,
                'dataType': ['Foundation', 'SR Legacy', 'Branded']
            }

            response = await asyncio.to_thread(
                self.session_manager.get, search_url, params=params, timeout=15
            )

            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                foods = data.get('foods', [])

                results = []
                for food in foods:
                    nutrients = []
                    for nutrient in food.get('foodNutrients', [])[:10]:  # Top 10 nutrients
                        nutrients.append({
                            'name': nutrient.get('nutrientName', ''),
                            'amount': nutrient.get('value', 0),
                            'unit': nutrient.get('unitName', '')
                        })

                    results.append({
                        'title': food.get('description', 'No description'),
                        'fdc_id': food.get('fdcId', ''),
                        'data_type': food.get('dataType', ''),
                        'brand_owner': food.get('brandOwner', ''),
                        'nutrients': nutrients,
                        'ingredients': food.get('ingredients', ''),
                        'url': f"https://fdc.nal.usda.gov/fdc-app.html#/food-details/{food.get('fdcId', '')}",
                        'source': 'USDA FoodData Central',
                        'summary': f"{food.get('description', '')} - {len(nutrients)} nutritional values available"[:300]
                    })

                circuit_breaker.reset(self.source_name)
                return self._create_result(query, True, {'results': results}, response_time=response_time)
            else:
                return self._create_result(query, False, error=f"HTTP {response.status_code}", response_time=response_time, status_code=response.status_code)

        except Exception as e:
            circuit_breaker.record_failure(self.source_name)
            return self._create_result(query, False, error=str(e), response_time=time.time() - start_time)

class DrugBankScraper(BaseScraper):
    """Scraper pro DrugBank Open Data - databáze léků a molekul"""

    def __init__(self):
        super().__init__("drugbank")

    async def scrape(self, query: str) -> ScrapingResult:
        """Scrape DrugBank pro farmakologické informace"""
        start_time = time.time()

        try:
            await self.rate_limiter.wait_if_needed(self.source_name, 2.0)

            if circuit_breaker.is_open(self.source_name):
                return self._create_result(query, False, error="Circuit breaker open")

            # DrugBank search (veřejné API je omezené, použijeme web scraping s opatrností)
            search_url = "https://go.drugbank.com/releases/latest/downloads/all-drugbank-vocabulary"

            # Alternativně použijeme ChEMBL API, které je více dostupné
            chembl_url = "https://www.ebi.ac.uk/chembl/api/data/molecule/search"
            params = {
                'q': query,
                'format': 'json',
                'limit': 10
            }

            response = await asyncio.to_thread(
                self.session_manager.get, chembl_url, params=params, timeout=15
            )

            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                molecules = data.get('molecules', [])

                results = []
                for molecule in molecules:
                    results.append({
                        'title': molecule.get('pref_name', 'Unknown compound'),
                        'chembl_id': molecule.get('molecule_chembl_id', ''),
                        'molecular_formula': molecule.get('molecule_properties', {}).get('molecular_formula', ''),
                        'molecular_weight': molecule.get('molecule_properties', {}).get('molecular_weight', 0),
                        'max_phase': molecule.get('max_phase', 0),
                        'molecule_type': molecule.get('molecule_type', ''),
                        'url': f"https://www.ebi.ac.uk/chembl/compound_report_card/{molecule.get('molecule_chembl_id', '')}",
                        'source': 'ChEMBL',
                        'summary': f"Compound: {molecule.get('pref_name', 'Unknown')} (Phase {molecule.get('max_phase', 0)})"[:300]
                    })

                circuit_breaker.reset(self.source_name)
                return self._create_result(query, True, {'results': results}, response_time=response_time)
            else:
                return self._create_result(query, False, error=f"HTTP {response.status_code}", response_time=response_time, status_code=response.status_code)

        except Exception as e:
            circuit_breaker.record_failure(self.source_name)
            return self._create_result(query, False, error=str(e), response_time=time.time() - start_time)

class PapersWithCodeScraper(BaseScraper):
    """Scraper pro Papers With Code - AI research s implementacemi"""

    def __init__(self):
        super().__init__("papers_with_code")

    async def scrape(self, query: str) -> ScrapingResult:
        """Scrape Papers With Code pro AI research"""
        start_time = time.time()

        try:
            await self.rate_limiter.wait_if_needed(self.source_name, 1.0)

            if circuit_breaker.is_open(self.source_name):
                return self._create_result(query, False, error="Circuit breaker open")

            # Papers With Code API
            search_url = "https://paperswithcode.com/api/v1/papers/"
            params = {
                'q': query,
                'items_per_page': 10
            }

            response = await asyncio.to_thread(
                self.session_manager.get, search_url, params=params, timeout=15
            )

            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                papers = data.get('results', [])

                results = []
                for paper in papers:
                    results.append({
                        'title': paper.get('title', 'No title'),
                        'abstract': paper.get('abstract', '')[:400],
                        'authors': paper.get('authors', []),
                        'published': paper.get('published', ''),
                        'arxiv_id': paper.get('arxiv_id', ''),
                        'github_url': paper.get('repository_url', ''),
                        'tasks': [task.get('name', '') for task in paper.get('tasks', [])],
                        'url_abs': paper.get('url_abs', ''),
                        'url_pdf': paper.get('url_pdf', ''),
                        'source': 'Papers With Code',
                        'summary': paper.get('abstract', '')[:300]
                    })

                circuit_breaker.reset(self.source_name)
                return self._create_result(query, True, {'results': results}, response_time=response_time)
            else:
                return self._create_result(query, False, error=f"HTTP {response.status_code}", response_time=response_time, status_code=response.status_code)

        except Exception as e:
            circuit_breaker.record_failure(self.source_name)
            return self._create_result(query, False, error=str(e), response_time=time.time() - start_time)

class ScrapingOrchestrator:
    """Hlavní orchestrátor pro koordinaci všech scraperů s pokročilou optimalizací"""

    def __init__(self):
        self.scrapers = {
            'wikipedia': WikipediaScraper(),
            'pubmed': PubMedScraper(),
            'openalex': OpenAlexScraper(),
            'clinicaltrials': ClinicalTrialsScraper(),
            'nih_reporter': NIHReporterScraper(),
            'arxiv': ArxivScraper(),
            'fooddata_central': FoodDataCentralScraper(),
            'drugbank': DrugBankScraper(),
            'papers_with_code': PapersWithCodeScraper()
        }
        self.logger = logging.getLogger(__name__)
        self._semaphore = asyncio.Semaphore(10)  # Limit concurrent requests

    async def scrape_all_sources(self, query: str, sources: Optional[List[str]] = None) -> List[ScrapingResult]:
        """Scrape všechny zdroje asynchronně s optimalizovaným batchingem"""
        if sources is None:
            sources = list(self.scrapers.keys())

        # Filter pouze dostupné zdroje
        available_sources = [s for s in sources if s in self.scrapers]
        if not available_sources:
            self.logger.warning(f"No available sources from requested: {sources}")
            return []

        # Batch scraping s rate limiting semaphorem
        async def scrape_with_semaphore(source: str):
            async with self._semaphore:
                scraper = self.scrapers[source]
                return await scraper.scrape(query)

        # Create tasks for parallel execution
        tasks = [scrape_with_semaphore(source) for source in available_sources]

        # Execute with timeout and error handling
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=60  # 60 second timeout for all scraping
            )
        except asyncio.TimeoutError:
            self.logger.error(f"Scraping timeout for query: {query}")
            # Return partial results with timeout errors
            results = [Exception("Timeout") for _ in available_sources]

        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                source = available_sources[i]
                error_result = ScrapingResult(
                    source=source,
                    query=query,
                    success=False,
                    data={},
                    error=f"Scraping exception: {str(result)}"
                )
                processed_results.append(error_result)
                self.logger.error(f"Scraping failed for {source}: {result}")
            else:
                processed_results.append(result)

        return processed_results

    async def batch_scrape_queries(self, queries: List[str], sources: Optional[List[str]] = None) -> Dict[str, List[ScrapingResult]]:
        """Batch scraping pro více queries najednou"""
        self.logger.info(f"Batch scraping {len(queries)} queries")

        # Create tasks for all query-source combinations
        tasks = []
        query_map = {}

        for query in queries:
            task = self.scrape_all_sources(query, sources)
            tasks.append(task)
            query_map[len(tasks) - 1] = query

        # Execute batch with controlled concurrency
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        batch_results = {}
        for i, result in enumerate(results):
            query = query_map[i]
            if isinstance(result, Exception):
                batch_results[query] = [ScrapingResult(
                    source="batch_error",
                    query=query,
                    success=False,
                    data={},
                    error=str(result)
                )]
            else:
                batch_results[query] = result

        return batch_results

    async def scrape_multiple_sources(
        self,
        query: str,
        sources: List[str],
        max_results_per_source: int = 5,
        timeout: int = 30
    ) -> List[Dict]:
        """Enhanced scraping s lepším error handling a results processing"""

        # Filter pouze dostupné zdroje
        available_sources = [s for s in sources if s in self.scrapers]
        if not available_sources:
            self.logger.warning(f"No available sources from: {sources}")
            return []

        # Spusť scraping všech zdrojů
        scraping_results = await self.scrape_all_sources(query, available_sources)

        # Process a normalize výsledky
        processed_results = []

        for result in scraping_results:
            if result.success and result.data:
                # Extract a normalize data podle typu zdroje
                normalized_data = self._normalize_source_data(result)

                # Omeз počet výsledků per source
                if isinstance(normalized_data, list):
                    normalized_data = normalized_data[:max_results_per_source]

                processed_results.extend(normalized_data if isinstance(normalized_data, list) else [normalized_data])
            else:
                # Add error placeholder
                error_result = {
                    'title': f'Error from {result.source}',
                    'summary': result.error or 'Unknown error',
                    'source': result.source,
                    'url': '',
                    'error': True
                }
                processed_results.append(error_result)

        return processed_results

    def _normalize_source_data(self, scraping_result: ScrapingResult) -> List[Dict]:
        """Normalizuje data z různých zdrojů do jednotného formátu"""
        source = scraping_result.source
        data = scraping_result.data

        normalized = []

        if source == 'wikipedia':
            articles = data.get('articles', [])
            for article in articles:
                normalized.append({
                    'title': article.get('title', ''),
                    'summary': article.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', ''),
                    'url': article.get('url', ''),
                    'source': 'Wikipedia',
                    'type': 'encyclopedia'
                })

        elif source == 'pubmed':
            papers = data.get('papers', [])
            for paper in papers:
                normalized.append({
                    'title': f"PubMed Article {paper.get('pmid', 'Unknown')}",
                    'summary': 'Medical research article from PubMed database',
                    'url': paper.get('url', ''),
                    'source': 'PubMed',
                    'type': 'research_paper',
                    'pmid': paper.get('pmid', '')
                })

        elif source == 'openalex':
            works = data.get('works', [])
            for work in works:
                normalized.append({
                    'title': work.get('title', ''),
                    'summary': f"Academic work published in {work.get('publication_year', 'Unknown year')}",
                    'url': work.get('url', ''),
                    'source': 'OpenAlex',
                    'type': 'academic_paper',
                    'doi': work.get('doi', ''),
                    'year': work.get('publication_year', '')
                })

        elif source == 'clinicaltrials':
            studies = data.get('results', [])
            for study in studies:
                normalized.append({
                    'title': study.get('title', ''),
                    'summary': study.get('summary', ''),
                    'url': study.get('url', ''),
                    'source': 'ClinicalTrials.gov',
                    'type': 'clinical_trial',
                    'nct_id': study.get('nct_id', ''),
                    'status': study.get('status', ''),
                    'conditions': study.get('conditions', [])
                })

        elif source == 'nih_reporter':
            projects = data.get('results', [])
            for project in projects:
                normalized.append({
                    'title': project.get('title', ''),
                    'summary': project.get('abstract', '')[:400] + '...' if project.get('abstract') else '',
                    'url': project.get('url', ''),
                    'source': 'NIH RePORTER',
                    'type': 'research_grant',
                    'pi_name': project.get('pi_name', ''),
                    'organization': project.get('organization', ''),
                    'award_amount': project.get('award_amount', 0)
                })

        elif source == 'arxiv':
            papers = data.get('results', [])
            for paper in papers:
                normalized.append({
                    'title': paper.get('title', ''),
                    'summary': paper.get('summary', ''),
                    'url': paper.get('url', ''),
                    'source': 'arXiv',
                    'type': 'preprint',
                    'arxiv_id': paper.get('arxiv_id', ''),
                    'authors': paper.get('authors', []),
                    'categories': paper.get('categories', []),
                    'pdf_url': paper.get('pdf_url', '')
                })

        elif source == 'papers_with_code':
            papers = data.get('results', [])
            for paper in papers:
                normalized.append({
                    'title': paper.get('title', ''),
                    'summary': paper.get('abstract', '')[:400] + '...' if paper.get('abstract') else '',
                    'url': paper.get('url_abs', ''),
                    'source': 'Papers With Code',
                    'type': 'ai_paper_with_code',
                    'github_url': paper.get('github_url', ''),
                    'tasks': paper.get('tasks', []),
                    'pdf_url': paper.get('url_pdf', '')
                })

        elif source == 'fooddata_central':
            foods = data.get('results', [])
            for food in foods:
                normalized.append({
                    'title': food.get('title', ''),
                    'summary': food.get('summary', ''),
                    'url': food.get('url', ''),
                    'source': 'USDA FoodData Central',
                    'type': 'nutrition_data',
                    'fdc_id': food.get('fdc_id', ''),
                    'nutrients': food.get('nutrients', []),
                    'brand_owner': food.get('brand_owner', '')
                })

        elif source == 'drugbank':
            compounds = data.get('results', [])
            for compound in compounds:
                normalized.append({
                    'title': compound.get('title', ''),
                    'summary': compound.get('summary', ''),
                    'url': compound.get('url', ''),
                    'source': 'ChEMBL/DrugBank',
                    'type': 'compound_data',
                    'chembl_id': compound.get('chembl_id', ''),
                    'molecular_formula': compound.get('molecular_formula', ''),
                    'max_phase': compound.get('max_phase', 0)
                })

        return normalized
```
