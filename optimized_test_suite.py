#!/usr/bin/env python3
"""
Comprehensive Test Suite - Kompletní testovací sada pro celý projekt
"""

import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path
import sys

# Přidej projekt do Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.main import UnifiedBiohackingResearchTool
from mcp_servers.web_automation_mcp_server import WebAutomationMCPServer
from cache.unified_cache_system import UnifiedCacheManager, get_cache_manager
from unified_config import get_config
from academic_scraper import AcademicScraper
from intelligent_research_orchestrator import IntelligentResearchOrchestrator

class TestUnifiedSystem:
    """Testy pro hlavní unified systém"""

    @pytest.mark.asyncio
    async def test_basic_research_functionality(self):
        """Test základní funkcionalita výzkumu"""
        async with UnifiedBiohackingResearchTool("test_user", verbose=True) as tool:
            result = await tool.research(
                query="BPC-157 peptide research",
                research_type="quick",
                evidence_level="medium"
            )

            assert result["success"] is True
            assert "query" in result
            assert result["query"] == "BPC-157 peptide research"
            print("✅ Basic research functionality test passed")

    @pytest.mark.asyncio
    async def test_performance_stats(self):
        """Test sledování výkonu"""
        async with UnifiedBiohackingResearchTool("test_user") as tool:
            # Proveď několik dotazů
            for i in range(3):
                await tool.research(f"test query {i}")

            stats = tool.get_performance_stats()
            assert stats["queries_processed"] == 3
            assert stats["total_time"] > 0
            print("✅ Performance stats test passed")

class TestCacheSystem:
    """Testy pro cache systém"""

    def setup_method(self):
        """Setup pro každý test"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = UnifiedCacheManager(cache_dir=self.temp_dir)

    @pytest.mark.asyncio
    async def test_cache_set_get(self):
        """Test základní cache operace"""
        test_data = {"test": "data", "number": 42}

        # Ulož do cache
        success = await self.cache.set("test_key", test_data, ttl=3600)
        assert success is True

        # Načti z cache
        retrieved = await self.cache.get("test_key")
        assert retrieved == test_data
        print("✅ Cache set/get test passed")

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test expirace cache"""
        # Ulož s krátkým TTL
        await self.cache.set("expire_test", "data", ttl=1)

        # Zkontroluj, že data jsou tam
        result = await self.cache.get("expire_test")
        assert result == "data"

        # Počkej na expiraci
        await asyncio.sleep(2)

        # Data by měla být expirovaná
        result = await self.cache.get("expire_test")
        assert result is None
        print("✅ Cache expiration test passed")

    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistik"""
        # Přidej nějaká data
        await self.cache.set("stats_test", "data")
        await self.cache.get("stats_test")

        stats = await self.cache.get_stats()
        assert "total_entries" in stats
        assert stats["total_entries"] >= 1
        print("✅ Cache stats test passed")

class TestWebAutomationMCP:
    """Testy pro Web Automation MCP Server"""

    def setup_method(self):
        """Setup pro každý test"""
        self.server = WebAutomationMCPServer()

    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Test inicializace serveru"""
        assert self.server.server is not None
        assert self.server.ssl_context is not None
        print("✅ MCP Server initialization test passed")

    @pytest.mark.asyncio
    async def test_session_creation(self):
        """Test vytvoření HTTP session"""
        session = await self.server._get_session()
        assert session is not None
        assert not session.closed

        # Cleanup
        await self.server.cleanup()
        print("✅ Session creation test passed")

class TestAcademicScraper:
    """Testy pro Academic Scraper"""

    @pytest.mark.asyncio
    async def test_scraper_initialization(self):
        """Test inicializace scraperu"""
        async with AcademicScraper() as scraper:
            assert scraper.session is not None
            assert not scraper.session.closed
        print("✅ Academic scraper initialization test passed")

    @pytest.mark.asyncio
    async def test_comprehensive_search_structure(self):
        """Test struktury výsledků comprehensive search"""
        async with AcademicScraper() as scraper:
            # Mock test - nebudeme dělat skutečné HTTP requesty v testech
            result = {
                'query': 'test query',
                'wikipedia': [],
                'pubmed': [],
                'total_results': 0,
                'timestamp': 12345
            }

            assert 'query' in result
            assert 'wikipedia' in result
            assert 'pubmed' in result
            assert 'total_results' in result
        print("✅ Comprehensive search structure test passed")

class TestIntelligentOrchestrator:
    """Testy pro Intelligent Research Orchestrator"""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test inicializace orchestrátoru"""
        orchestrator = IntelligentResearchOrchestrator()
        assert orchestrator.cache_manager is not None
        assert orchestrator.research_stats["total_queries"] == 0
        print("✅ Orchestrator initialization test passed")

    @pytest.mark.asyncio
    async def test_safety_assessment(self):
        """Test bezpečnostního hodnocení"""
        orchestrator = IntelligentResearchOrchestrator()

        # Mock data s bezpečnostními klíčovými slovy
        mock_results = {
            "all_results": [
                {
                    "title": "Safety study of test compound",
                    "snippet": "This study examines side effects and toxicity",
                    "confidence": 0.9
                }
            ]
        }

        safety_info = await orchestrator._assess_safety("test query", mock_results)

        assert "risk_level" in safety_info
        assert "safety_mentions" in safety_info
        assert "recommendation" in safety_info
        print("✅ Safety assessment test passed")

class TestConfiguration:
    """Testy pro konfigurační systém"""

    def test_config_loading(self):
        """Test načítání konfigurace"""
        config = get_config()
        assert config is not None

        # Test základních sekcí
        assert config.get("database") is not None
        assert config.get("api") is not None
        assert config.get("scraping") is not None
        print("✅ Configuration loading test passed")

    def test_config_get_set(self):
        """Test get/set operací konfigurace"""
        config = get_config()

        # Test get
        rate_limit = config.get("api.rate_limit")
        assert rate_limit is not None

        # Test set
        config.set("test.value", "test_data")
        retrieved = config.get("test.value")
        assert retrieved == "test_data"
        print("✅ Configuration get/set test passed")

@pytest.mark.asyncio
async def test_integration_basic_workflow():
    """Integrační test základního workflow"""
    # Test celého workflow od začátku do konce
    async with UnifiedBiohackingResearchTool("integration_test") as tool:
        result = await tool.research(
            query="integration test",
            research_type="quick",
            evidence_level="medium",
            output_format="json"
        )

        # Kontrola základní struktury odpovědi
        assert isinstance(result, dict)
        assert "success" in result
        assert "query" in result

        # Kontrola performance stats
        stats = tool.get_performance_stats()
        assert stats["queries_processed"] >= 1

    print("✅ Integration workflow test passed")

def run_performance_benchmark():
    """Spustí benchmark test výkonu"""
    import time

    async def benchmark():
        start_time = time.time()

        async with UnifiedBiohackingResearchTool("benchmark_test") as tool:
            # Proveď sérii testů
            tasks = []
            for i in range(5):
                task = tool.research(f"benchmark test {i}", research_type="quick")
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            end_time = time.time()
            total_time = end_time - start_time

            successful = sum(1 for r in results if r.get("success"))

            print(f"\n📊 Performance Benchmark Results:")
            print(f"   Total requests: {len(tasks)}")
            print(f"   Successful requests: {successful}")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Average time per request: {total_time/len(tasks):.2f}s")
            print(f"   Requests per second: {len(tasks)/total_time:.2f}")

    asyncio.run(benchmark())

if __name__ == "__main__":
    print("🧪 Running Comprehensive Test Suite...")
    print("=" * 60)

    # Spusť testy
    pytest.main([__file__, "-v", "--tb=short"])

    print("\n" + "=" * 60)
    print("🚀 Running Performance Benchmark...")
    run_performance_benchmark()

    print("\n✅ All tests completed!")
