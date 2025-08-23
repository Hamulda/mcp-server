#!/usr/bin/env python3
"""
Academic Scraper - Optimalizovaný scraper pro akademické zdroje
"""

import asyncio
import aiohttp
import time
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote
import logging

logger = logging.getLogger(__name__)

class AcademicScraper:
    """Optimalizovaný scraper pro akademické databáze"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = {}  # Simple rate limiting

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _rate_limit(self, domain: str, delay: float = 1.0):
        """Simple rate limiting per domain"""
        now = time.time()
        if domain in self.rate_limiter:
            elapsed = now - self.rate_limiter[domain]
            if elapsed < delay:
                await asyncio.sleep(delay - elapsed)
        self.rate_limiter[domain] = time.time()

    async def search_wikipedia(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Vyhledá články na Wikipedii"""
        try:
            await self._rate_limit('wikipedia.org')

            # Wikipedia API search
            search_url = f"https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': limit
            }

            async with self.session.get(search_url, params=params) as response:
                data = await response.json()

                results = []
                for item in data.get('query', {}).get('search', []):
                    article_url = f"https://en.wikipedia.org/wiki/{quote(item['title'].replace(' ', '_'))}"
                    results.append({
                        'title': item['title'],
                        'snippet': item['snippet'],
                        'url': article_url,
                        'source': 'wikipedia',
                        'confidence': 0.8
                    })

                return results

        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
            return []

    async def search_pubmed(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Vyhledá články v PubMed"""
        try:
            await self._rate_limit('pubmed.ncbi.nlm.nih.gov')

            # PubMed search via E-utilities
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': limit,
                'retmode': 'json'
            }

            async with self.session.get(search_url, params=params) as response:
                data = await response.json()

                pmids = data.get('esearchresult', {}).get('idlist', [])

                if not pmids:
                    return []

                # Získej detaily článků
                details_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                details_params = {
                    'db': 'pubmed',
                    'id': ','.join(pmids),
                    'retmode': 'json'
                }

                async with self.session.get(details_url, params=details_params) as details_response:
                    details_data = await details_response.json()

                    results = []
                    for pmid in pmids:
                        article = details_data.get('result', {}).get(pmid, {})
                        if article:
                            results.append({
                                'title': article.get('title', ''),
                                'authors': ', '.join([author.get('name', '') for author in article.get('authors', [])]),
                                'journal': article.get('source', ''),
                                'pubdate': article.get('pubdate', ''),
                                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                'pmid': pmid,
                                'source': 'pubmed',
                                'confidence': 0.9
                            })

                    return results

        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []

    async def search_comprehensive(self, query: str) -> Dict[str, Any]:
        """Komplexní vyhledávání napříč všemi zdroji"""
        tasks = [
            self.search_wikipedia(query),
            self.search_pubmed(query)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        wikipedia_results = results[0] if not isinstance(results[0], Exception) else []
        pubmed_results = results[1] if not isinstance(results[1], Exception) else []

        return {
            'query': query,
            'wikipedia': wikipedia_results,
            'pubmed': pubmed_results,
            'total_results': len(wikipedia_results) + len(pubmed_results),
            'timestamp': time.time()
        }

def create_scraping_orchestrator():
    """Factory function pro vytvoření scraper instance"""
    return AcademicScraper()
