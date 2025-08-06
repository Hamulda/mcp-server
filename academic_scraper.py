"""
Scraper pro akademické zdroje - arXiv, PubMed, Google Scholar, atd.
"""
import asyncio
import aiohttp
from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import re
import json

class ArxivScraper:
    """Scraper pro arXiv.org - matematické a fyzikální výzkumy"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "http://export.arxiv.org/api/query"
    
    async def scrape(self, query) -> List[Dict[str, Any]]:
        """Scraping arXiv databáze"""
        try:
            search_query = self._build_arxiv_query(query.query)
            url = f"{self.base_url}?search_query={search_query}&start=0&max_results={query.max_results}&sortBy=submittedDate&sortOrder=descending"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        xml_data = await response.text()
                        return self._parse_arxiv_xml(xml_data)
                    
        except Exception as e:
            self.logger.error(f"Chyba při arXiv scraping: {e}")
        
        return []
    
    def _build_arxiv_query(self, query: str) -> str:
        """Sestavení dotazu pro arXiv API"""
        # Základní vyhledávání v title, abstract a všech polích
        terms = query.replace(' ', '+AND+')
        return f"all:{terms}"
    
    def _parse_arxiv_xml(self, xml_data: str) -> List[Dict[str, Any]]:
        """Parsování XML odpovědi z arXiv"""
        results = []
        
        try:
            root = ET.fromstring(xml_data)
            
            # Namespace pro arXiv
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            entries = root.findall('atom:entry', ns)
            
            for entry in entries:
                title = entry.find('atom:title', ns)
                summary = entry.find('atom:summary', ns)
                published = entry.find('atom:published', ns)
                updated = entry.find('atom:updated', ns)
                
                # Autoři
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns)
                    if name is not None:
                        authors.append(name.text)
                
                # Kategorie
                categories = []
                for category in entry.findall('atom:category', ns):
                    term = category.get('term')
                    if term:
                        categories.append(term)
                
                # URL
                pdf_url = None
                for link in entry.findall('atom:link', ns):
                    if link.get('type') == 'application/pdf':
                        pdf_url = link.get('href')
                        break
                
                results.append({
                    'title': title.text.strip() if title is not None else '',
                    'abstract': summary.text.strip() if summary is not None else '',
                    'authors': authors,
                    'categories': categories,
                    'published_date': published.text if published is not None else '',
                    'updated_date': updated.text if updated is not None else '',
                    'pdf_url': pdf_url,
                    'source': 'arXiv',
                    'type': 'academic_paper'
                })
                
        except ET.ParseError as e:
            self.logger.error(f"Chyba při parsování arXiv XML: {e}")
        
        return results


class BasePubMedScraper:
    """Základní třída pro PubMed scraping"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    async def scrape_pubmed(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Univerzální PubMed scraping"""
        try:
            # Krok 1: Vyhledání ID článků
            search_url = f"{self.base_url}/esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'pub_date'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=search_params) as response:
                    if response.status != 200:
                        return []
                    
                    search_data = await response.json()
                    id_list = search_data.get('esearchresult', {}).get('idlist', [])
                    
                    if not id_list:
                        return []
                
                # Krok 2: Získání detailů článků
                fetch_url = f"{self.base_url}/efetch.fcgi"
                fetch_params = {
                    'db': 'pubmed',
                    'id': ','.join(id_list),
                    'retmode': 'xml'
                }
                
                async with session.get(fetch_url, params=fetch_params) as response:
                    if response.status == 200:
                        xml_data = await response.text()
                        return self._parse_pubmed_xml(xml_data)
                        
        except Exception as e:
            self.logger.error(f"Chyba při PubMed scraping: {e}")
        
        return []
    
    def _parse_pubmed_xml(self, xml_data: str) -> List[Dict[str, Any]]:
        """Parsování XML odpovědi z PubMed"""
        results = []
        
        try:
            root = ET.fromstring(xml_data)
            articles = root.findall('.//PubmedArticle')
            
            for article in articles:
                title_elem = article.find('.//ArticleTitle')
                abstract_elem = article.find('.//AbstractText')
                
                # Autoři
                authors = []
                author_list = article.find('.//AuthorList')
                if author_list is not None:
                    for author in author_list.findall('.//Author'):
                        last_name = author.find('.//LastName')
                        first_name = author.find('.//ForeName')
                        if last_name is not None:
                            name = last_name.text
                            if first_name is not None:
                                name = f"{first_name.text} {name}"
                            authors.append(name)
                
                # Datum publikace
                pub_date = article.find('.//PubDate')
                pub_date_str = ""
                if pub_date is not None:
                    year = pub_date.find('.//Year')
                    month = pub_date.find('.//Month')
                    day = pub_date.find('.//Day')
                    
                    if year is not None:
                        pub_date_str = year.text
                        if month is not None:
                            pub_date_str += f"-{month.text}"
                            if day is not None:
                                pub_date_str += f"-{day.text}"
                
                # PMID
                pmid_elem = article.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else ""
                
                results.append({
                    'title': title_elem.text if title_elem is not None else '',
                    'abstract': abstract_elem.text if abstract_elem is not None else '',
                    'authors': authors,
                    'publication_date': pub_date_str,
                    'pmid': pmid,
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else '',
                    'source': 'PubMed',
                    'type': 'academic_paper'
                })
                
        except ET.ParseError as e:
            self.logger.error(f"Chyba při parsování PubMed XML: {e}")
        
        return results


class PubMedScraper(BasePubMedScraper):
    """Standardní PubMed scraper"""

    async def scrape(self, query) -> List[Dict[str, Any]]:
        """Scraping PubMed databáze"""
        return await self.scrape_pubmed(query.query, query.max_results)


class GoogleScholarScraper:
    """Skutečný scraper pro Google Scholar pomocí scholarly knihovny"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://scholar.google.com/scholar"
        self._setup_scholarly()
    
    def _setup_scholarly(self):
        """Nastavění scholarly knihovny"""
        try:
            from scholarly import scholarly
            self.scholarly = scholarly
            # Nastavení pro opatrné scrapování
            self.scholarly.set_timeout(10)
            self.logger.info("Scholarly knihovna úspěšně načtena")
        except ImportError:
            self.logger.error("Scholarly knihovna není nainstalována. Použijte: pip install scholarly")
            self.scholarly = None
        except Exception as e:
            self.logger.error(f"Chyba při nastavování scholarly: {e}")
            self.scholarly = None
    
    async def scrape(self, query, max_retries: int = 3) -> List[Dict[str, Any]]:
        """Skutečné scraping Google Scholar s retry mechanikou"""
        if not self.scholarly:
            self.logger.warning("Scholarly není dostupná, vracím prázdný seznam")
            return []
        
        results = []
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Spouštím Google Scholar scraping (pokus {attempt + 1})")
                
                # Vyhledání publikací
                search_query = self.scholarly.search_pubs(query.query)
                
                for i, publication in enumerate(search_query):
                    if i >= query.max_results:
                        break
                    
                    try:
                        # Získání detailních informací
                        pub_filled = self.scholarly.fill(publication)
                        
                        # Parsování autorů
                        authors = []
                        if 'author' in pub_filled:
                            authors = [author.get('name', '') for author in pub_filled['author']]
                        
                        # Parsování roku
                        year = None
                        if 'pub_year' in pub_filled:
                            year = pub_filled['pub_year']
                        elif 'year' in pub_filled:
                            year = pub_filled['year']
                        
                        result = {
                            'title': pub_filled.get('title', ''),
                            'abstract': pub_filled.get('abstract', ''),
                            'authors': authors,
                            'year': year,
                            'citation_count': pub_filled.get('num_citations', 0),
                            'venue': pub_filled.get('venue', ''),
                            'url': pub_filled.get('pub_url', ''),
                            'pdf_url': pub_filled.get('eprint_url', ''),
                            'source': 'Google Scholar',
                            'type': 'academic_paper',
                            'scholar_id': pub_filled.get('scholar_id', ''),
                            'citedby_url': pub_filled.get('citedby_url', '')
                        }
                        
                        results.append(result)
                        
                        # Pauza mezi požadavky pro předcházení blokaci
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        self.logger.warning(f"Chyba při zpracování publikace {i}: {e}")
                        continue
                
                self.logger.info(f"Google Scholar scraping úspěšný: {len(results)} výsledků")
                return results
                
            except Exception as e:
                self.logger.error(f"Chyba při Google Scholar scraping (pokus {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * 5  # 5, 10, 20 sekund
                    self.logger.info(f"Čekám {wait_time} sekund před dalším pokusem...")
                    await asyncio.sleep(wait_time)
                else:
                    # Poslední pokus - zaloguj failed URL
                    self._log_failed_query(query.query)
        
        return results
    
    def _log_failed_query(self, query: str):
        """Zalogování neúspěšného dotazu"""
        try:
            with open('failed_scholar_queries.log', 'a') as f:
                f.write(f"{datetime.now().isoformat()}: {query}\n")
        except Exception as e:
            self.logger.error(f"Nelze uložit failed query: {e}")
    
    async def get_author_info(self, author_name: str) -> Dict[str, Any]:
        """Získání informací o autorovi"""
        if not self.scholarly:
            return {}
        
        try:
            search_query = self.scholarly.search_author(author_name)
            author = next(search_query)
            author_filled = self.scholarly.fill(author)
            
            return {
                'name': author_filled.get('name', ''),
                'affiliation': author_filled.get('affiliation', ''),
                'interests': author_filled.get('interests', []),
                'citedby': author_filled.get('citedby', 0),
                'h_index': author_filled.get('hindex', 0),
                'scholar_id': author_filled.get('scholar_id', ''),
                'homepage': author_filled.get('homepage', '')
            }
        except Exception as e:
            self.logger.error(f"Chyba při získávání info o autorovi {author_name}: {e}")
            return {}


class SemanticScholarScraper:
    """Scraper pro Semantic Scholar API"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://api.semanticscholar.org/graph/v1"
    
    async def scrape(self, query) -> List[Dict[str, Any]]:
        """Scraping Semantic Scholar pomocí jejich API"""
        try:
            search_url = f"{self.base_url}/paper/search"
            params = {
                'query': query.query,
                'limit': min(query.max_results, 100),  # API limit
                'fields': 'paperId,title,abstract,authors,year,citationCount,url,venue'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_semantic_scholar_response(data)
                    else:
                        self.logger.warning(f"Semantic Scholar API vrátilo status {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Chyba při Semantic Scholar scraping: {e}")
        
        return []
    
    def _parse_semantic_scholar_response(self, data: Dict) -> List[Dict[str, Any]]:
        """Parsování odpovědi z Semantic Scholar API"""
        results = []
        
        papers = data.get('data', [])
        
        for paper in papers:
            authors = []
            if paper.get('authors'):
                authors = [author.get('name', '') for author in paper['authors']]
            
            results.append({
                'title': paper.get('title', ''),
                'abstract': paper.get('abstract', ''),
                'authors': authors,
                'year': paper.get('year'),
                'citation_count': paper.get('citationCount', 0),
                'venue': paper.get('venue', ''),
                'url': paper.get('url', ''),
                'paper_id': paper.get('paperId', ''),
                'source': 'Semantic Scholar',
                'type': 'academic_paper'
            })
        
        return results


class AcademicScraperManager:
    """Manager pro koordinaci všech akademických scraperů"""
    
    def __init__(self):
        self.scrapers = {
            'arxiv': ArxivScraper(),
            'pubmed': PubMedScraper(),
            'google_scholar': GoogleScholarScraper(),
            'semantic_scholar': SemanticScholarScraper()
        }
        self.logger = logging.getLogger(__name__)
    
    async def scrape_all(self, query) -> Dict[str, List[Dict[str, Any]]]:
        """Spuštění všech akademických scraperů paralelně"""
        tasks = []
        
        for source_name, scraper in self.scrapers.items():
            task = asyncio.create_task(
                self._scrape_with_timeout(scraper, query, source_name)
            )
            tasks.append((source_name, task))
        
        results = {}
        
        for source_name, task in tasks:
            try:
                result = await task
                results[source_name] = result
            except Exception as e:
                self.logger.error(f"Chyba při scraping {source_name}: {e}")
                results[source_name] = []
        
        return results
    
    async def _scrape_with_timeout(self, scraper, query, source_name: str, timeout: int = 30):
        """Spuštění scraperu s timeoutem"""
        try:
            return await asyncio.wait_for(scraper.scrape(query), timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout při scraping {source_name}")
            return []
        except Exception as e:
            self.logger.error(f"Neočekávaná chyba při scraping {source_name}: {e}")
            return []


class HealthResearchScraper(BasePubMedScraper):
    """Scraper pro zdravotní výzkum - rozšiřuje BasePubMedScraper"""

    def __init__(self):
        super().__init__()
        self.clinical_trials_base_url = "https://clinicaltrials.gov/api/query/full_studies"

    async def scrape_clinical_trials(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Scraping ClinicalTrials.gov databáze s retry mechanikou"""
        for attempt in range(3):  # Retry mechanism
            try:
                search_url = f"{self.clinical_trials_base_url}"
                search_params = {
                    'expr': query,
                    'min_rnk': 1,
                    'max_rnk': max_results,
                    'fmt': 'json'
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(search_url, params=search_params) as response:
                        if response.status == 503:  # Service temporarily unavailable
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        elif response.status != 200:
                            return []

                        data = await response.json()
                        studies = data.get('FullStudiesResponse', {}).get('FullStudies', [])

                        results = []
                        for study in studies:
                            protocol_section = study.get('Study', {}).get('ProtocolSection', {})
                            results.append({
                                'title': protocol_section.get('IdentificationModule', {}).get('OfficialTitle', ''),
                                'summary': protocol_section.get('DescriptionModule', {}).get('BriefSummary', ''),
                                'status': protocol_section.get('StatusModule', {}).get('OverallStatus', ''),
                                'source': 'ClinicalTrials.gov',
                                'type': 'clinical_trial'
                            })

                        return results

            except Exception as e:
                self.logger.error(f"Chyba při ClinicalTrials.gov scraping (pokus {attempt + 1}): {e}")
                if attempt == 2:  # Last attempt
                    # Log failed URL for later review
                    self._log_failed_url(f"{search_url}?{search_params}")
                await asyncio.sleep(1)

        return []

    def _log_failed_url(self, url: str):
        """Zalogování neúspěšného URL pro pozdější kontrolu"""
        try:
            with open('failed_urls.log', 'a') as f:
                f.write(f"{datetime.now().isoformat()}: {url}\n")
        except Exception as e:
            self.logger.error(f"Nelze uložit failed URL: {e}")

    def scrape_medical_sources(self, query: str, medical_keywords: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Scraping lékařských zdrojů s optimalizací pro klíčová slova"""
        from text_processing_utils import text_processor

        if medical_keywords is None:
            medical_keywords = text_processor.medical_keywords

        # Optimalizace dotazu pomocí centrálního procesoru
        optimized_query = text_processor.distill_medical_text(query, max_sentences=5)

        return {
            "PubMed": asyncio.run(self.scrape_pubmed(optimized_query)),
            "ClinicalTrials": asyncio.run(self.scrape_clinical_trials(optimized_query))
        }
