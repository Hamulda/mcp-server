"""
Komprehenzivní testovací sada pro optimalizovaný projekt
"""

import asyncio
import pytest
import time
import json
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock

# Test základních funkcí
class TestSecurityManager:
    """Testy pro SecurityManager"""

    @pytest.fixture
    def security_manager(self):
        from security.security_manager import SecurityManager
        return SecurityManager()

    def test_password_hashing(self, security_manager):
        """Test hashování hesel"""
        password = "test_password_123"
        hashed = security_manager.hash_password(password)

        assert hashed != password
        assert security_manager.verify_password(password, hashed)
        assert not security_manager.verify_password("wrong_password", hashed)

    def test_jwt_token_creation(self, security_manager):
        """Test vytváření JWT tokenů"""
        data = {"sub": "test_user", "permissions": ["read"]}
        token = security_manager.create_access_token(data)

        assert isinstance(token, str)
        assert len(token) > 50  # JWT by měl být dlouhý

        # Verify token
        payload = security_manager.verify_token(token)
        assert payload["sub"] == "test_user"
        assert payload["permissions"] == ["read"]

    def test_input_sanitization(self, security_manager):
        """Test sanitizace vstupů"""
        malicious_input = "<script>alert('xss')</script>SELECT * FROM users--"
        sanitized = security_manager.sanitize_input(malicious_input)

        assert "<script>" not in sanitized
        assert "SELECT" not in sanitized
        assert "--" not in sanitized

    def test_rate_limiting(self, security_manager):
        """Test rate limitingu"""
        client_ip = "127.0.0.1"

        # První requesty by měly projít
        for i in range(50):
            assert security_manager.check_rate_limit(client_ip)

        # 61. request by měl být zamítnut
        for i in range(15):
            security_manager.check_rate_limit(client_ip)

        assert not security_manager.check_rate_limit(client_ip)

class TestAdvancedRateLimiter:
    """Testy pro pokročilý rate limiter"""

    @pytest.fixture
    def rate_limiter(self):
        from optimization.advanced_rate_limiter import AdaptiveRateLimiter
        return AdaptiveRateLimiter()

    @pytest.mark.asyncio
    async def test_source_rate_limiting(self, rate_limiter):
        """Test rate limitingu per source"""
        source = "pubmed"

        # Test normálního použití
        for i in range(5):
            result = await rate_limiter.check_rate_limit(source)
            assert result["allowed"]
            await rate_limiter.record_request(source)

        # Test překročení limitu
        for i in range(10):
            await rate_limiter.record_request(source)

        result = await rate_limiter.check_rate_limit(source)
        assert not result["allowed"]
        assert "rate_limit_exceeded" in result["reason"]

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, rate_limiter):
        """Test circuit breaker pattern"""
        source = "test_api"

        # Simuluj selhání
        for i in range(6):  # Překročení failure_threshold
            await rate_limiter.record_response(source, 1.0, False)

        result = await rate_limiter.check_rate_limit(source)
        assert not result["allowed"]
        assert result["circuit_state"] == "open"

    @pytest.mark.asyncio
    async def test_adaptive_delay(self, rate_limiter):
        """Test adaptivního delay"""
        source = "slow_api"

        # Simuluj pomalé response times
        for i in range(10):
            await rate_limiter.record_response(source, 3.0, True)

        result = await rate_limiter.check_rate_limit(source)
        assert result["adaptive_delay"] > 0

class TestIntelligentCacheSystem:
    """Testy pro intelligent cache systém"""

    @pytest.fixture
    def cache_manager(self):
        from cache.intelligent_cache_system import IntelligentCacheManager
        return IntelligentCacheManager()

    @pytest.mark.asyncio
    async def test_basic_cache_operations(self, cache_manager):
        """Test základních cache operací"""
        await cache_manager.initialize()

        # Test set/get
        test_data = {"key": "value", "number": 42}
        await cache_manager.set("test_key", test_data, ttl=60)

        retrieved = await cache_manager.get("test_key")
        assert retrieved == test_data

        # Test neexistujícího klíče
        missing = await cache_manager.get("nonexistent")
        assert missing is None

    @pytest.mark.asyncio
    async def test_cache_eviction(self, cache_manager):
        """Test eviction policy"""
        await cache_manager.initialize()
        cache_manager.max_memory_items = 5  # Malý limit pro test

        # Naplň cache
        for i in range(10):
            await cache_manager.set(f"key_{i}", f"value_{i}", ttl=3600)

        # Zkontroluj, že se cache nevyčerpala
        assert len(cache_manager.memory_cache) <= cache_manager.max_memory_items

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, cache_manager):
        """Test invalidation patterns"""
        await cache_manager.initialize()

        # Přidej několik klíčů
        await cache_manager.set("user_1_data", "data1", ttl=3600)
        await cache_manager.set("user_1_profile", "profile1", ttl=3600)
        await cache_manager.set("user_2_data", "data2", ttl=3600)

        # Invaliduj všechny user_1 klíče
        await cache_manager.invalidate("user_1")

        assert await cache_manager.get("user_1_data") is None
        assert await cache_manager.get("user_1_profile") is None
        assert await cache_manager.get("user_2_data") == "data2"

class TestRemoteMCPHandler:
    """Testy pro remote MCP handler"""

    @pytest.fixture
    def mcp_handler(self):
        from core.remote_mcp_handler import RemoteMCPHandler
        return RemoteMCPHandler()

    @pytest.mark.asyncio
    async def test_tool_registration(self, mcp_handler):
        """Test registrace nástrojů"""

        async def test_tool(params):
            return {"result": "success", "params": params}

        mcp_handler.register_tool("test_tool", test_tool)
        assert "test_tool" in mcp_handler.tool_registry

    @pytest.mark.asyncio
    async def test_tool_execution(self, mcp_handler):
        """Test spuštění nástroje"""

        async def mock_tool(params):
            return {"processed": params["input"]}

        mcp_handler.register_tool("mock_tool", mock_tool)

        # Simuluj připojeného klienta
        client_id = "test_client"
        mcp_handler.connected_clients[client_id] = {
            "user_id": "test_user",
            "permissions": ["use_mock_tool"],
            "connected_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }

        # Mock token manager
        mcp_handler.auth_manager.validate_token_permission = Mock(return_value=True)

        result = await mcp_handler.execute_tool(
            client_id,
            "mock_tool",
            {"input": "test_data"}
        )

        assert result["status"] == "success"
        assert result["result"]["processed"] == "test_data"

class TestUnifiedServer:
    """Integrační testy pro unified server"""

    @pytest.fixture
    def test_client(self):
        from fastapi.testclient import TestClient
        from core.unified_server import app
        return TestClient(app)

    def test_health_endpoint(self, test_client):
        """Test health check endpointu"""
        response = test_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "version" in data
        assert "components" in data

    def test_metrics_endpoint(self, test_client):
        """Test Prometheus metrics endpointu"""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_auth_endpoints(self, test_client):
        """Test autentifikačních endpointů"""
        # Test login
        login_data = {"username": "admin", "password": "admin"}
        response = test_client.post("/auth/login", json=login_data)
        assert response.status_code == 200

        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

        # Test protected endpoint s tokenem
        headers = {"Authorization": f"Bearer {data['access_token']}"}
        response = test_client.get("/api/stats", headers=headers)
        assert response.status_code == 200

class TestAdvancedMonitoring:
    """Testy pro monitoring systém"""

    @pytest.fixture
    def monitoring_system(self):
        from monitoring.advanced_monitoring_system import AdvancedMonitoringSystem
        return AdvancedMonitoringSystem()

    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, monitoring_system):
        """Test sběru systémových metrik"""
        metrics = await monitoring_system.collect_system_metrics()

        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert "disk_usage" in metrics
        assert 0 <= metrics["cpu_usage"] <= 100
        assert 0 <= metrics["memory_usage"] <= 100

    @pytest.mark.asyncio
    async def test_alert_evaluation(self, monitoring_system):
        """Test vyhodnocování alertů"""
        # Simuluj vysoké CPU usage
        test_metrics = {"cpu_usage": 95, "memory_usage": 50}

        alerts = await monitoring_system.evaluate_alerts(test_metrics)

        # Měl by být vytvořen critical CPU alert
        critical_alerts = [a for a in alerts if a.severity.value == "critical"]
        assert len(critical_alerts) > 0
        assert any("cpu" in alert.name.lower() for alert in critical_alerts)

    @pytest.mark.asyncio
    async def test_external_api_checks(self, monitoring_system):
        """Test kontrol externích API"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response

            results = await monitoring_system.check_external_apis()

            assert len(results) > 0
            for api_name, status in results.items():
                assert "available" in status
                assert "response_time" in status

class TestPerformanceBenchmarks:
    """Performance testy"""

    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test výkonu cache systému"""
        from cache.intelligent_cache_system import IntelligentCacheManager

        cache_manager = IntelligentCacheManager()
        await cache_manager.initialize()

        # Test rychlosti set operací
        start_time = time.time()
        for i in range(1000):
            await cache_manager.set(f"perf_key_{i}", f"value_{i}")
        set_time = time.time() - start_time

        # Test rychlosti get operací
        start_time = time.time()
        for i in range(1000):
            await cache_manager.get(f"perf_key_{i}")
        get_time = time.time() - start_time

        print(f"Cache SET performance: {1000/set_time:.2f} ops/sec")
        print(f"Cache GET performance: {1000/get_time:.2f} ops/sec")

        # Asserts pro minimální výkon
        assert set_time < 5.0  # 1000 set operací za méně než 5 sekund
        assert get_time < 2.0  # 1000 get operací za méně než 2 sekundy

    @pytest.mark.asyncio
    async def test_rate_limiter_performance(self):
        """Test výkonu rate limiteru"""
        from optimization.advanced_rate_limiter import AdaptiveRateLimiter

        rate_limiter = AdaptiveRateLimiter()

        start_time = time.time()
        for i in range(1000):
            await rate_limiter.check_rate_limit("test_source", f"user_{i%10}")
        check_time = time.time() - start_time

        print(f"Rate limiter performance: {1000/check_time:.2f} checks/sec")
        assert check_time < 3.0  # 1000 checks za méně než 3 sekundy

@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """End-to-end test celého workflow"""
    from fastapi.testclient import TestClient
    from core.unified_server import app

    client = TestClient(app)

    # 1. Health check
    response = client.get("/health")
    assert response.status_code == 200

    # 2. Login
    login_data = {"username": "admin", "password": "admin"}
    response = client.post("/auth/login", json=login_data)
    assert response.status_code == 200
    token = response.json()["access_token"]

    # 3. Authorized request
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/api/stats", headers=headers)
    assert response.status_code == 200

    # 4. Check metrics
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "requests_total" in response.text

if __name__ == "__main__":
    # Spuštění testů s detailním výstupem
    pytest.main([__file__, "-v", "--tb=short", "--show-capture=no"])
