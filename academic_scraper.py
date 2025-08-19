"""
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
            pool_connections=20,
            pool_maxsize=50
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Optimalizované headers s kompresí
        self.session.headers.update({
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
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
            
            # Parse response (support JSON and XML for robustness)
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
                try:
                    import xml.etree.ElementTree as ET
                    xml_root = ET.fromstring(getattr(response, 'text', '') or '')
                    id_list = []
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

# Simplified scrapers without timeout issues
class ScrapingOrchestrator:
    """Hlavní orchestrátor pro koordinaci všech scraperů s pokročilou optimalizací"""

    def __init__(self):
        self.scrapers = {
            'wikipedia': WikipediaScraper(),
            'pubmed': PubMedScraper(),
            'openalex': OpenAlexScraper()
        }
        self.logger = logging.getLogger(__name__)
        self._semaphore = asyncio.Semaphore(10)

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

# Factory function for creating orchestrator
def create_scraping_orchestrator() -> ScrapingOrchestrator:
    """Factory function pro vytvoření scraping orchestratoru"""
    return ScrapingOrchestrator()
