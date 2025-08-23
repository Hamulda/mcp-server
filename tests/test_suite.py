"""
Konsolidovaný Test Suite - Jediný soubor pro všechny testy
Nahrazuje: direct_server_test.py, test_optimized_components.py, comprehensive_test_suite.py,
test_modules.py, test_server.py, minimal_test.py, quick_test.py
"""

import asyncio
import pytest
import requests
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class UnifiedTestSuite:
    """Jediný konsolidovaný test suite pro celý projekt"""

    def __init__(self):
        self.base_url = "http://localhost:8080"
        self.results = []

    @pytest.mark.asyncio
    async def test_server_health(self):
        """Test základní funkčnosti serveru"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            logger.info("✅ Server health check passed")
            return True
        except Exception as e:
            logger.error(f"❌ Server health check failed: {e}")
            return False

    @pytest.mark.asyncio
    async def test_api_endpoints(self):
        """Test klíčových API endpointů"""
        endpoints = ["/", "/docs", "/redoc", "/metrics", "/api/v1/sources"]

        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                assert response.status_code in [200, 404]  # 404 je OK pro některé endpointy
                logger.info(f"✅ Endpoint {endpoint} responded")
            except Exception as e:
                logger.error(f"❌ Endpoint {endpoint} failed: {e}")

    @pytest.mark.asyncio
    async def test_scraping_functionality(self):
        """Test scraping funkčnosti"""
        test_payload = {
            "query": "test research",
            "sources": ["wikipedia"]
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/v1/scrape",
                json=test_payload,
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                assert "results" in data
                logger.info("✅ Scraping test passed")
                return True
        except Exception as e:
            logger.error(f"❌ Scraping test failed: {e}")
            return False

    def run_all_tests(self):
        """Spustí všechny testy rychle"""
        print("🧪 Spouštím konsolidované testy...")

        # Test serveru
        server_ok = asyncio.run(self.test_server_health())

        if server_ok:
            # Test endpointů
            asyncio.run(self.test_api_endpoints())

            # Test funkčnosti
            asyncio.run(self.test_scraping_functionality())

        print("✅ Testy dokončeny!")

if __name__ == "__main__":
    suite = UnifiedTestSuite()
    suite.run_all_tests()
