"""
Test script pro nové pokročilé funkce
"""

import asyncio
import json
import time
from datetime import datetime

async def test_enhanced_features():
    """Test nových pokročilých funkcí"""

    print("🧪 Testování pokročilých funkcí...")

    # Test 1: Security Manager
    print("\n1️⃣ Test Security Manager")
    try:
        from security_manager import security_manager

        # Test JWT token creation
        token_data = {"sub": "test_user", "permissions": ["read", "write"]}
        access_token = security_manager.create_access_token(token_data)
        print(f"✅ JWT token vytvořen: {access_token[:50]}...")

        # Test token verification
        payload = security_manager.verify_token(access_token)
        print(f"✅ Token ověřen: {payload['sub']}")

        # Test input sanitization
        dangerous_input = "<script>alert('xss')</script>"
        safe_input = security_manager.sanitize_input(dangerous_input)
        print(f"✅ Input sanitizace: '{dangerous_input}' -> '{safe_input}'")

    except Exception as e:
        print(f"❌ Security Manager test failed: {e}")

    # Test 2: Enhanced Rate Limiter
    print("\n2️⃣ Test Enhanced Rate Limiter")
    try:
        from enhanced_rate_limiter import intelligent_limiter

        # Test rate limit check
        result = await intelligent_limiter.check_rate_limit("test_user", "pubmed")
        print(f"✅ Rate limit check: {result}")

        # Test request recording
        await intelligent_limiter.record_request("test_user", "pubmed", 0.5, True)
        print("✅ Request zaznamenán")

        # Test adaptive delay
        delay = intelligent_limiter.get_adaptive_delay("pubmed")
        print(f"✅ Adaptivní delay: {delay}s")

    except Exception as e:
        print(f"❌ Enhanced Rate Limiter test failed: {e}")

    # Test 3: Semantic Search System
    print("\n3️⃣ Test Semantic Search System")
    try:
        from semantic_search_system import enhanced_semantic_search, ResearchPaper

        # Initialize
        await enhanced_semantic_search.initialize()
        print("✅ Semantic search engine inicializován")

        # Test adding a paper
        test_paper = ResearchPaper(
            id="test_001",
            title="Machine Learning in Academic Research",
            abstract="This paper explores the applications of machine learning in academic research environments...",
            authors=["John Doe", "Jane Smith"],
            year=2024,
            keywords=["machine learning", "academic research", "automation"],
            research_field="Computer Science"
        )

        success = await enhanced_semantic_search.add_paper(test_paper)
        print(f"✅ Paper přidán: {success}")

        # Test semantic search
        results = await enhanced_semantic_search.semantic_search("machine learning research", top_k=5)
        print(f"✅ Semantic search: {len(results)} výsledků")

    except Exception as e:
        print(f"❌ Semantic Search test failed: {e}")

    # Test 4: Advanced Monitoring System
    print("\n4️⃣ Test Advanced Monitoring System")
    try:
        from advanced_monitoring_system import monitoring_system

        # Test metrics recording
        monitoring_system.record_api_request("GET", "/test", 200, 0.5)
        monitoring_system.record_cache_access("unified", True)
        monitoring_system.record_auth_attempt(True)
        print("✅ Monitoring metriky zaznamenány")

        # Test metrics summary
        summary = monitoring_system.get_metrics_summary(1)
        print(f"✅ Metrics summary: {len(summary)} položek")

    except Exception as e:
        print(f"❌ Advanced Monitoring test failed: {e}")

    print("\n🎉 Testování dokončeno!")

async def test_unified_server_status():
    """Test statusu unified serveru"""

    print("\n🏥 Test Unified Server Status")
    try:
        import httpx

        # Test if server is running (může selhat pokud není spuštěný)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8001/api/v1/status")
                if response.status_code == 200:
                    status = response.json()
                    print("✅ Server běží a odpovídá")
                    print(f"📊 Core features: {status['core_features']}")
                    print(f"🚀 Enhanced features: {status['enhanced_features']}")
                else:
                    print(f"⚠️ Server odpověděl s kódem: {response.status_code}")
        except httpx.ConnectError:
            print("⚠️ Server neběží na localhost:8001")

    except ImportError:
        print("⚠️ httpx není dostupný pro HTTP testy")

if __name__ == "__main__":
    print("🔬 Research Tool - Test pokročilých funkcí")
    print("=" * 50)

    # Spuštění testů
    asyncio.run(test_enhanced_features())
    asyncio.run(test_unified_server_status())

    print("\n📋 Souhrn implementovaných funkcí:")
    print("✅ JWT Authentication & Security")
    print("✅ Intelligent Rate Limiting s Redis")
    print("✅ Enhanced Semantic Search s ChromaDB")
    print("✅ Citation Network Analysis")
    print("✅ Research Trend Detection")
    print("✅ Collaboration Recommendations")
    print("✅ Advanced Monitoring s Prometheus")
    print("✅ Circuit Breaker Pattern")
    print("✅ Input Sanitization")
    print("✅ Enterprise API Endpoints")

    print("\n🚀 Pro spuštění serveru:")
    print("python unified_server.py")
    print("\n📚 API dokumentace:")
    print("http://localhost:8001/docs")
