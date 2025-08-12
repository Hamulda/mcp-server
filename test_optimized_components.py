"""
Comprehensive Test Suite pro optimalizované komponenty
Pokrývá academic_scraper, unified_config s mock testy a validací
"""

import asyncio
import pytest
import json
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any

# Import komponenty k testování - opravené importy
from academic_scraper import (
    WikipediaScraper, PubMedScraper, ScrapingResult, EnhancedRateLimiter,
    EnhancedSessionManager, ScrapingOrchestrator, create_scraping_orchestrator
)
from unified_config import (
    get_config, validate_config_on_startup, Environment, SourceConfig, 
    ScrapingConfig, UnifiedConfig
)

# FastAPI app testování
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from unified_server import create_app

class TestEnhancedRateLimiter:
    """Test rate limiteru s exponential backoff"""

    def setup_method(self):
        self.rate_limiter = EnhancedRateLimiter()

    @pytest.mark.asyncio
    async def test_basic_rate_limiting(self):
        """Test základního rate limitingu"""
        source = "test_source"
        start_time = time.time()

        # První request - žádné čekání
        await self.rate_limiter.wait_if_needed(source, 0.1)
        first_time = time.time()

        # Druhý request - měl by čekat
        await self.rate_limiter.wait_if_needed(source, 0.1)
        second_time = time.time()

        # Ověř, že druhý request čekal
        assert second_time - first_time >= 0.05  # Trochu tolerance

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test exponential backoff při 429 errors"""
        source = "test_source"

        # Simuluj několik 429 errors
        self.rate_limiter.record_429_error(source)
        self.rate_limiter.record_429_error(source)

        assert self.rate_limiter._consecutive_429s[source] == 2

        # Reset by měl vyčistit counter
        self.rate_limiter.reset_429_count(source)
        assert source not in self.rate_limiter._consecutive_429s

class TestEnhancedSessionManager:
    """Test session manageru"""

    def setup_method(self):
        self.session_manager = EnhancedSessionManager()

    def test_session_initialization(self):
        """Test inicializace session"""
        assert self.session_manager.session is not None
        assert self.session_manager.default_timeout > 0
        assert 'User-Agent' in self.session_manager.session.headers

    @patch('requests.Session.get')
    def test_get_with_retry(self, mock_get):
        """Test GET s retry logic"""
        # Mock úspěšnou odpověď
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"test content"
        mock_get.return_value = mock_response

        response = self.session_manager.get("http://example.com")

        assert response.status_code == 200
        mock_get.assert_called_once()

    def teardown_method(self):
        self.session_manager.close()

class TestScrapingResult:
    """Test ScrapingResult dataclass"""

    def test_scraping_result_creation(self):
        """Test vytvoření ScrapingResult"""
        result = ScrapingResult(
            source="test",
            query="test query",
            success=True,
            data={"papers": []},
            response_time=1.5
        )

        assert result.source == "test"
        assert result.success is True
        assert result.response_time == 1.5
        assert result.error is None

class TestWikipediaScraper:
    """Test Wikipedia scraperu s mock API"""

    def setup_method(self):
        self.scraper = WikipediaScraper()

    @pytest.mark.asyncio
    @patch('academic_scraper.get_session_manager')
    async def test_successful_wikipedia_scrape(self, mock_session_manager):
        """Test úspěšného scrapingu Wikipedia"""
        # Mock session manager
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"test content"  # Přidám content pro logging
        mock_response.json.return_value = {
            'query': {
                'search': [
                    {
                        'title': 'Test Article',
                        'snippet': 'Test snippet',
                        'size': 1000,
                        'timestamp': '2024-01-01T00:00:00Z'
                    }
                ]
            }
        }
        mock_session.get.return_value = mock_response
        mock_session_manager.return_value = mock_session
        
        # Patch také global session manager
        with patch('academic_scraper._session_manager', mock_session):
            result = await self.scraper.scrape("test query")
        
        assert result.success is True
        # Oprava: Wikipedia vrací 1 článek, ne víc
        assert len(result.data['articles']) >= 1
        assert result.data['articles'][0]['title'] == 'Test Article'

    @pytest.mark.asyncio
    @patch('academic_scraper.get_session_manager')
    async def test_wikipedia_rate_limit(self, mock_session_manager):
        """Test rate limit handling"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.reason = "Too Many Requests"
        mock_response.content = b""
        mock_session.get.return_value = mock_response
        mock_session_manager.return_value = mock_session

        result = await self.scraper.scrape("test query")

        assert result.success is False
        assert result.rate_limited is True
        assert result.status_code == 429

class TestPubMedScraper:
    """Test PubMed scraperu"""

    def setup_method(self):
        self.scraper = PubMedScraper()

    @pytest.mark.asyncio
    @patch('academic_scraper.get_session_manager')
    async def test_successful_pubmed_scrape(self, mock_session_manager):
        """Test úspěšného PubMed scrapingu"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [
                {
                    'title': 'Test Paper',
                    'abstract': 'Test abstract',
                    'publication_year': 2024,
                    'doi': '10.1000/test',
                    'cited_by_count': 10,
                    'open_access': {'is_oa': True},
                    'authorships': [{'author': {'display_name': 'Test Author'}}],
                    'id': 'https://pubmed.ncbi.nlm.nih.gov/W123'
                }
            ],
            'meta': {'count': 1}
        }
        mock_session.get.return_value = mock_response
        mock_session_manager.return_value = mock_session

        result = await self.scraper.scrape("test query")

        assert result.success is True
        assert len(result.data['papers']) == 1
        assert result.data['papers'][0]['title'] == 'Test Paper'
        assert result.data['papers'][0]['citations'] == 10

class TestEnhancedAcademicScraper:
    """Test hlavního scraper orchestrátoru"""

    def setup_method(self):
        self.scraper = EnhancedAcademicScraper()

    @pytest.mark.asyncio
    async def test_scrape_unknown_source(self):
        """Test scrapingu neznámého zdroje"""
        result = await self.scraper.scrape_single_source("unknown_source", "test")

        assert result.success is False
        assert "Unknown source" in result.error

    @pytest.mark.asyncio
    @patch('academic_scraper.WikipediaScraper.scrape')
    @patch('academic_scraper.PubMedScraper.scrape')
    async def test_scrape_multiple_sources(self, mock_pubmed, mock_wikipedia):
        """Test scrapingu více zdrojů současně"""
        # Mock výsledky
        mock_wikipedia.return_value = ScrapingResult(
            source="wikipedia", query="test", success=True,
            data={"articles": [{"title": "Wiki"}]}
        )
        mock_pubmed.return_value = ScrapingResult(
            source="pubmed", query="test", success=True,
            data={"papers": [{"title": "Paper"}]}
        )

        results = await self.scraper.scrape_multiple_sources(
            "test query", ["wikipedia", "pubmed"]
        )

        assert len(results) == 2
        assert "wikipedia" in results
        assert "pubmed" in results
        assert results["wikipedia"]["success"] is True
        assert results["pubmed"]["success"] is True

    @pytest.mark.asyncio
    async def test_scrape_timeout(self):
        """Test timeout při scrapingu"""
        # Mock pomalý scraper - správné použití AsyncMock
        async def slow_scrape(*args, **kwargs):
            await asyncio.sleep(2)
            return ScrapingResult(
                source="wikipedia", query="test", success=True,
                data={"articles": []}, response_time=2.0
            )

        with patch('academic_scraper.WikipediaScraper.scrape', new=slow_scrape):
            with pytest.raises(asyncio.TimeoutError):
                await self.scraper.scrape_multiple_sources(
                    "test", ["wikipedia"], timeout=1
                )

class TestConfigSystem:
    """Test konfiguračního systému"""

    def test_get_config_development(self):
        """Test získání development konfigurace"""
        with patch.dict(os.environ, {'FLASK_ENV': 'development'}):
            config = get_config()
            assert config.ENVIRONMENT == Environment.DEVELOPMENT
            assert isinstance(config, DevelopmentConfig)

    def test_get_config_testing(self):
        """Test testing konfigurace"""
        config = get_config(Environment.TESTING)
        assert config.ENVIRONMENT == Environment.TESTING
        assert isinstance(config, TestingConfig)

    def test_get_config_production(self):
        """Test production konfigurace"""
        config = get_config(Environment.PRODUCTION)
        assert config.ENVIRONMENT == Environment.PRODUCTION
        assert isinstance(config, ProductionConfig)

    def test_source_config_validation(self):
        """Test validace source konfigurace"""
        config = DevelopmentConfig()

        # Test valid config - dočasně nastavíme API klíč pro test
        with patch.dict(os.environ, {'SEMANTIC_SCHOLAR_API_KEY': 'test_key'}):
            errors = validate_config_on_startup(config)
            assert len(errors) == 0

        # Test invalid config
        config.SOURCES['test'] = SourceConfig(name='Test', base_url='')
        errors = validate_config_on_startup(config)
        assert any('missing base_url' in error for error in errors)

    def test_get_enabled_sources(self):
        """Test získání povolených zdrojů"""
        config = DevelopmentConfig()
        enabled = config.get_enabled_sources()

        # Google Scholar je disabled by default
        assert 'google_scholar' not in enabled
        assert 'wikipedia' in enabled
        assert 'openalex' in enabled

    def test_sources_by_priority(self):
        """Test řazení zdrojů podle priority"""
        config = DevelopmentConfig()
        sources_by_priority = config.get_sources_by_priority()

        # OpenAlex a Semantic Scholar mají prioritu 1, Wikipedia 2
        assert sources_by_priority[0][1].priority <= sources_by_priority[1][1].priority

class TestFlaskApp:
    """Test Flask aplikace"""

    def setup_method(self):
        self.app = create_app('testing')
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_health_endpoint(self):
        """Test health check endpointu"""
        response = self.client.get('/api/health')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data

    def test_sources_endpoint(self):
        """Test sources endpointu"""
        response = self.client.get('/api/sources')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'sources' in data
        assert isinstance(data['sources'], list)
        assert len(data['sources']) > 0

    def test_scrape_endpoint_validation(self):
        """Test validace scrape endpointu"""
        # Test chybějící query
        response = self.client.post('/api/scrape',
                                  json={},
                                  content_type='application/json')
        assert response.status_code == 400

        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'Validation failed'

    def test_scrape_endpoint_empty_query(self):
        """Test prázdného query"""
        response = self.client.post('/api/scrape',
                                  json={'query': '   '},
                                  content_type='application/json')
        assert response.status_code == 400

    @patch('app.EnhancedAcademicScraper.scrape_multiple_sources')
    def test_scrape_endpoint_success(self, mock_scrape):
        """Test úspěšného scrape endpointu"""
        # Mock výsledek
        mock_scrape.return_value = {
            'wikipedia': {
                'success': True,
                'data': {'articles': [{'title': 'Test'}]},
                'response_time': 1.0
            }
        }

        response = self.client.post('/api/scrape',
                                  json={'query': 'test query'},
                                  content_type='application/json')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True
        assert 'results' in data
        assert 'summary' in data

    def test_404_error_handler(self):
        """Test 404 error handler"""
        response = self.client.get('/api/nonexistent')
        assert response.status_code == 404

        data = json.loads(response.data)
        assert data['error'] == 'Endpoint not found'

    def test_method_not_allowed(self):
        """Test 405 error handler"""
        response = self.client.put('/api/health')
        assert response.status_code == 405

        data = json.loads(response.data)
        assert data['error'] == 'Method not allowed'

class TestIntegration:
    """Integration testy celého systému"""

    @pytest.mark.asyncio
    @patch('academic_scraper.requests.Session.get')
    async def test_end_to_end_scraping(self, mock_get):
        """Test kompletního end-to-end scrapingu"""
        # Mock Wikipedia API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'query': {
                'search': [
                    {
                        'title': 'Machine Learning',
                        'snippet': 'Machine learning is...',
                        'size': 5000,
                        'timestamp': '2024-01-01T00:00:00Z'
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        # Test complete scraping flow
        scraper = EnhancedAcademicScraper()
        results = await scraper.scrape_multiple_sources(
            "machine learning", ["wikipedia"]
        )

        assert len(results) == 1
        assert "wikipedia" in results
        assert results["wikipedia"]["success"] is True

# Performance testy
class TestPerformance:
    """Performance testy"""

    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self):
        """Test výkonu při concurrent requests"""
        scraper = EnhancedAcademicScraper()

        # Mock rychlé odpovědi
        with patch('academic_scraper.WikipediaScraper.scrape') as mock_scrape:
            mock_scrape.return_value = ScrapingResult(
                source="wikipedia", query="test", success=True,
                data={"articles": []}, response_time=0.1
            )

            start_time = time.time()
            results = await scraper.scrape_multiple_sources(
                "test", ["wikipedia"] * 3
            )
            end_time = time.time()

            # Concurrent requests by měly být rychlejší než sekvenční
            total_time = end_time - start_time
            assert total_time < 1.0  # Mělo by být rychlé kvůli mock

# Test fixtures
@pytest.fixture
def sample_config():
    """Fixture pro testovací konfiguraci"""
    return TestingConfig()

@pytest.fixture
def mock_wikipedia_response():
    """Fixture pro mock Wikipedia odpověď"""
    return {
        'query': {
            'search': [
                {
                    'title': 'Test Article',
                    'snippet': 'Test snippet',
                    'size': 1000,
                    'timestamp': '2024-01-01T00:00:00Z'
                }
            ]
        }
    }

if __name__ == "__main__":
    # Spusti testy
    pytest.main([__file__, "-v", "--tb=short"])
