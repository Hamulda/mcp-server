"""
Web scraper pro obecné webové stránky, zprávy a blogy
"""
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import logging
from urllib.parse import urljoin, urlparse
import re
from datetime import datetime
import time

from config import REQUEST_HEADERS, REQUEST_DELAY

class WebScraper:
    """Scraper pro obecné webové stránky"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=REQUEST_HEADERS,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def scrape(self, query) -> List[Dict[str, Any]]:
        """Hlavní metoda pro scraping"""
        try:
            # Získání URL z různých zdrojů
            search_urls = await self._get_search_urls(query.query)
            
            # Scraping jednotlivých stránek
            results = []
            for url in search_urls[:query.max_results]:
                try:
                    content = await self._scrape_page(url)
                    if content:
                        results.append(content)
                    await asyncio.sleep(REQUEST_DELAY)
                except Exception as e:
                    self.logger.warning(f"Chyba při scraping {url}: {e}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Chyba při web scraping: {e}")
            return []
    
    async def _get_search_urls(self, query: str) -> List[str]:
        """Získání URL pro vyhledávání"""
        # Pro začátek použijeme předem definované zdroje
        # V produkční verzi by se použily vyhledávací API
        
        base_urls = [
            "https://techcrunch.com",
            "https://www.wired.com",
            "https://arstechnica.com",
            "https://www.theverge.com",
            "https://news.ycombinator.com"
        ]
        
        # Simulace vyhledávání - v produkci by se použilo skutečné API
        search_urls = []
        for base_url in base_urls:
            search_urls.append(f"{base_url}/search?q={query}")
        
        return search_urls
    
    async def _scrape_page(self, url: str) -> Dict[str, Any]:
        """Scraping jednotlivé stránky"""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extrakce základních informací
                title = self._extract_title(soup)
                content = self._extract_content(soup)
                meta_info = self._extract_meta_info(soup)
                
                return {
                    'url': url,
                    'title': title,
                    'content': content,
                    'meta': meta_info,
                    'scraped_at': datetime.now().isoformat(),
                    'word_count': len(content.split()) if content else 0
                }
                
        except Exception as e:
            self.logger.error(f"Chyba při scraping {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extrakce titulku stránky"""
        title = soup.find('title')
        if title:
            return title.get_text().strip()
        
        # Alternativní způsoby
        h1 = soup.find('h1')
        if h1:
            return h1.get_text().strip()
        
        return "Bez titulku"
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extrakce hlavního obsahu stránky"""
        # Odebrání skriptů a stylů
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Hledání hlavního obsahu
        content_selectors = [
            'article',
            '[role="main"]',
            '.content',
            '.post-content',
            '.entry-content',
            'main',
            '.article-body'
        ]
        
        for selector in content_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                text = content_div.get_text()
                return self._clean_text(text)
        
        # Fallback - použití celého body
        body = soup.find('body')
        if body:
            text = body.get_text()
            return self._clean_text(text)
        
        return ""
    
    def _extract_meta_info(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extrakce meta informací"""
        meta_info = {}
        
        # Autor
        author_selectors = [
            'meta[name="author"]',
            '.author',
            '.byline',
            '[rel="author"]'
        ]
        
        for selector in author_selectors:
            author_elem = soup.select_one(selector)
            if author_elem:
                if author_elem.name == 'meta':
                    meta_info['author'] = author_elem.get('content', '')
                else:
                    meta_info['author'] = author_elem.get_text().strip()
                break
        
        # Datum
        date_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="date"]',
            '.date',
            '.published',
            'time[datetime]'
        ]
        
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                if date_elem.name == 'meta':
                    meta_info['date'] = date_elem.get('content', '')
                elif date_elem.name == 'time':
                    meta_info['date'] = date_elem.get('datetime', '')
                else:
                    meta_info['date'] = date_elem.get_text().strip()
                break
        
        # Popis
        description = soup.find('meta', attrs={'name': 'description'})
        if description:
            meta_info['description'] = description.get('content', '')
        
        return meta_info
    
    def _clean_text(self, text: str) -> str:
        """Vyčištění textu"""
        # Odebrání přebytečných bílých znaků
        text = re.sub(r'\s+', ' ', text)
        
        # Odebrání speciálních znaků
        text = re.sub(r'[^\w\s\.,!?;:()\[\]{}"\'-]', '', text)
        
        return text.strip()


class NewsAPIScraper:
    """Scraper pro News API a podobné služby"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
    
    async def scrape(self, query) -> List[Dict[str, Any]]:
        """Scraping zpráv přes API"""
        if not self.api_key:
            self.logger.warning("News API klíč není nastaven")
            return []
        
        try:
            # Implementace News API
            # Pro demonstraci vracím mock data
            return [
                {
                    'title': f'News article about {query.query}',
                    'content': f'Mock news content for {query.query}',
                    'source': 'NewsAPI',
                    'url': 'https://example.com/news',
                    'published_at': datetime.now().isoformat()
                }
            ]
        except Exception as e:
            self.logger.error(f"Chyba při News API scraping: {e}")
            return []


class RSSFeedScraper:
    """Scraper pro RSS feedy"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def scrape(self, query) -> List[Dict[str, Any]]:
        """Scraping RSS feedů"""
        try:
            import feedparser
            
            results = []
            rss_feeds = [
                'https://feeds.feedburner.com/oreilly/radar',
                'https://techcrunch.com/feed/',
                'https://www.wired.com/feed/rss'
            ]
            
            for feed_url in rss_feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:10]:  # Top 10 z každého feedu
                        if self._is_relevant(entry, query.query):
                            results.append({
                                'title': entry.get('title', ''),
                                'content': entry.get('summary', ''),
                                'url': entry.get('link', ''),
                                'published_at': entry.get('published', ''),
                                'source': feed.feed.get('title', 'RSS Feed')
                            })
                
                except Exception as e:
                    self.logger.warning(f"Chyba při zpracování RSS {feed_url}: {e}")
            
            return results
            
        except ImportError:
            self.logger.error("feedparser není nainstalován")
            return []
        except Exception as e:
            self.logger.error(f"Chyba při RSS scraping: {e}")
            return []
    
    def _is_relevant(self, entry, query: str) -> bool:
        """Kontrola relevance článku k dotazu"""
        query_words = query.lower().split()
        
        title = entry.get('title', '').lower()
        summary = entry.get('summary', '').lower()
        
        content = f"{title} {summary}"
        
        # Počítání shod
        matches = sum(1 for word in query_words if word in content)
        
        # Artikel je relevantní pokud obsahuje alespoň 30% slov z dotazu
        return matches >= len(query_words) * 0.3
