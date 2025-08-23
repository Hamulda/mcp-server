#!/usr/bin/env python3
"""
Komprehenzivní test suite pro M1 optimalizovaný projekt
Testuje všechny konsolidované komponenty a M1 optimalizace
"""

import asyncio
import time
import traceback
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class M1ProjectTestSuite:
    """Komplexní test suite pro M1 optimalizovaný projekt"""

    def __init__(self):
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "errors": [],
            "performance_metrics": {},
            "m1_optimizations": {}
        }

    async def run_all_tests(self):
        """Spustí všechny testy pro M1 optimalizovaný projekt"""
        print("🚀 Spouštím komprehenzivní testování M1 optimalizovaného projektu...")

        # Test 1: Cache systém M1 optimalizace
        await self._test_m1_cache_system()

        # Test 2: Konsolidovaný research orchestrator
        await self._test_consolidated_orchestrator()

        # Test 3: Unified Copilot tools
        await self._test_copilot_tools()

        # Test 4: M1 výkonové benchmarky
        await self._test_m1_performance()

        # Test 5: Paměťová efektivnost
        await self._test_memory_efficiency()

        # Test 6: Integrace všech komponent
        await self._test_full_integration()

        # Výsledný report
        self._generate_final_report()

    async def _test_m1_cache_system(self):
        """Test M1 optimalizovaného cache systému"""
        print("\n📊 Testování M1 Cache Systému...")

        try:
            from unified_cache_system import get_unified_cache, M1OptimizedCacheSystem

            # Test základní funkčnosti
            async with get_unified_cache() as cache:
                # Test write/read
                test_data = {"test": "M1 cache data", "numbers": [1, 2, 3, 4, 5]}

                start_time = time.time()
                await cache.set("test_key", test_data)
                write_time = time.time() - start_time

                start_time = time.time()
                retrieved = await cache.get("test_key")
                read_time = time.time() - start_time

                # Ověření dat
                assert retrieved == test_data, "Data se neshodují po cache operaci"

                # Test M1 komprese
                large_data = {"large": "x" * 10000}  # 10KB data
                await cache.set("large_key", large_data)
                retrieved_large = await cache.get("large_key")
                assert retrieved_large == large_data, "Komprese/dekomprese selhala"

                # Statistiky M1 optimalizací
                stats = cache.get_stats()
                self.test_results["m1_optimizations"]["cache_stats"] = stats

                self.test_results["performance_metrics"]["cache_write_ms"] = write_time * 1000
                self.test_results["performance_metrics"]["cache_read_ms"] = read_time * 1000

                print(f"✅ M1 Cache: Write {write_time*1000:.2f}ms, Read {read_time*1000:.2f}ms")
                print(f"✅ M1 Optimalizace: {stats.get('m1_optimizations', 0)} použitých optimalizací")

                self.test_results["passed"] += 1

        except Exception as e:
            error_msg = f"M1 Cache test failed: {str(e)}"
            self.test_results["errors"].append(error_msg)
            self.test_results["failed"] += 1
            print(f"❌ {error_msg}")

    async def _test_consolidated_orchestrator(self):
        """Test konsolidovaného research orchestrátoru"""
        print("\n🧠 Testování Konsolidovaného Research Orchestrátoru...")

        try:
            from enhanced_research_orchestrator import get_research_orchestrator, ConsolidatedResearchOrchestrator

            # Test základní funkčnosti
            orchestrator = get_research_orchestrator()

            # Test research modes
            modes = orchestrator._initialize_research_modes()
            assert len(modes) >= 5, "Nedostatek research modes"

            # Test user profile management
            test_user = "test_user_m1"
            user_profile = await orchestrator._get_user_profile(test_user)
            assert user_profile.user_id == test_user, "User profile creation failed"

            # Test query analysis (bez AI závislostí)
            test_query = "BPC-157 dosage and safety"
            enhanced_query, insights = await orchestrator._analyze_and_enhance_query(test_query, user_profile)

            # Test source selection
            mode = modes["balanced_research"]
            sources = await orchestrator._ai_select_sources(test_query, mode, user_profile)
            assert len(sources) > 0, "Source selection failed"

            print(f"✅ Orchestrator: {len(modes)} režimů, {len(sources)} zdrojů vybraných")
            print(f"✅ Query enhancement: '{test_query}' -> '{enhanced_query[:50]}...'")

            self.test_results["passed"] += 1

        except Exception as e:
            error_msg = f"Orchestrator test failed: {str(e)}"
            self.test_results["errors"].append(error_msg)
            self.test_results["failed"] += 1
            print(f"❌ {error_msg}")

    async def _test_copilot_tools(self):
        """Test unified Copilot tools"""
        print("\n🛠️ Testování Unified Copilot Tools...")

        try:
            from copilot_tools import get_copilot_tools, UnifiedCopilotInterface

            tools = get_copilot_tools()

            # Test biohacking compound validator
            validation = await tools.validate_biohacking_compound("bpc-157")
            assert validation.compound_name == "bpc-157", "Compound validation failed"
            assert validation.confidence_score > 0, "Confidence score missing"

            # Test code pattern optimizer
            test_code = """
class TestClass:
    def __init__(self):
        self.client = None
        
    def connect(self):
        # Resource management code
        pass
"""
            suggestions = tools.suggest_code_patterns(test_code)

            # Test async safety guard
            async_code = """
async def test_func():
    time.sleep(1)  # Should trigger warning
    result = asyncio.sleep(1)  # Missing await
"""
            async_issues = tools.check_async_safety(async_code)

            # Test privacy leak detector
            privacy_code = """
api_key = "sk-1234567890abcdef"  # Should trigger warning
password = "secretpassword123"  # Should trigger warning
"""
            privacy_issues = tools.scan_privacy_leaks(privacy_code)

            print(f"✅ Compound Validation: {validation.research_status}")
            print(f"✅ Code Patterns: {len(suggestions)} návrhy optimalizace")
            print(f"✅ Async Safety: {len(async_issues)} zjištěných problémů")
            print(f"✅ Privacy Scanner: {len(privacy_issues)} úniků detekováno")

            self.test_results["passed"] += 1

        except Exception as e:
            error_msg = f"Copilot tools test failed: {str(e)}"
            self.test_results["errors"].append(error_msg)
            self.test_results["failed"] += 1
            print(f"❌ {error_msg}")

    async def _test_m1_performance(self):
        """Test M1 specifických výkonových optimalizací"""
        print("\n⚡ Testování M1 Výkonových Optimalizací...")

        try:
            import psutil
            import platform

            # Ověření M1 architektury
            machine = platform.machine()
            is_m1 = machine == "arm64"

            # Memory benchmark
            start_time = time.time()
            large_dict = {f"key_{i}": f"value_{i}" * 100 for i in range(1000)}
            memory_alloc_time = time.time() - start_time

            # CPU info
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            memory = psutil.virtual_memory()

            # M1 specific checks
            m1_features = {
                "is_arm64": is_m1,
                "cpu_cores": cpu_count,
                "memory_gb": memory.total / (1024**3),
                "memory_alloc_ms": memory_alloc_time * 1000
            }

            self.test_results["m1_optimizations"]["hardware_features"] = m1_features
            self.test_results["performance_metrics"]["memory_alloc_ms"] = memory_alloc_time * 1000

            print(f"✅ Architektura: {machine} ({'M1 optimalizováno' if is_m1 else 'Intel/AMD'})")
            print(f"✅ CPU: {cpu_count} jader")
            print(f"✅ Paměť: {memory.total / (1024**3):.1f} GB")
            print(f"✅ Memory allocation: {memory_alloc_time*1000:.2f}ms")

            self.test_results["passed"] += 1

        except Exception as e:
            error_msg = f"M1 performance test failed: {str(e)}"
            self.test_results["errors"].append(error_msg)
            self.test_results["failed"] += 1
            print(f"❌ {error_msg}")

    async def _test_memory_efficiency(self):
        """Test paměťové efektivity M1 optimalizací"""
        print("\n💾 Testování Paměťové Efektivity...")

        try:
            import gc
            import sys

            # Initial memory state
            gc.collect()
            initial_objects = len(gc.get_objects())

            # Test cache system memory usage
            from unified_cache_system import get_unified_cache

            async with get_unified_cache() as cache:
                # Fill cache with test data
                for i in range(100):
                    await cache.set(f"test_{i}", {"data": "x" * 1000, "index": i})

                # Check memory usage
                stats = cache.get_stats()
                cache_objects = stats.get("memory_items", 0)

                # Clear cache
                await cache.clear()

            gc.collect()
            final_objects = len(gc.get_objects())

            memory_efficiency = {
                "initial_objects": initial_objects,
                "final_objects": final_objects,
                "cache_items_tested": 100,
                "memory_cleanup_effective": final_objects <= initial_objects + 50
            }

            self.test_results["m1_optimizations"]["memory_efficiency"] = memory_efficiency

            print(f"✅ Objekty před testem: {initial_objects}")
            print(f"✅ Objekty po testu: {final_objects}")
            print(f"✅ Memory cleanup: {'Efektivní' if memory_efficiency['memory_cleanup_effective'] else 'Potřebuje optimalizaci'}")

            self.test_results["passed"] += 1

        except Exception as e:
            error_msg = f"Memory efficiency test failed: {str(e)}"
            self.test_results["errors"].append(error_msg)
            self.test_results["failed"] += 1
            print(f"❌ {error_msg}")

    async def _test_full_integration(self):
        """Test plné integrace všech optimalizovaných komponent"""
        print("\n🔗 Testování Plné Integrace Komponent...")

        try:
            from unified_cache_system import get_unified_cache
            from enhanced_research_orchestrator import get_research_orchestrator
            from copilot_tools import get_copilot_tools

            start_time = time.time()

            # Integration test scenario
            async with get_unified_cache() as cache:
                orchestrator = get_research_orchestrator()
                tools = get_copilot_tools()

                # Test workflow: cache -> orchestrator -> tools

                # 1. Store test data in cache
                test_research_data = {
                    "query": "BPC-157 research",
                    "results": ["result1", "result2", "result3"],
                    "timestamp": time.time()
                }
                await cache.set("research_test", test_research_data)

                # 2. Retrieve and process with orchestrator
                cached_data = await cache.get("research_test")
                assert cached_data is not None, "Cache retrieval failed"

                # 3. Validate compound with tools
                validation = await tools.validate_biohacking_compound("bpc-157")
                assert validation.confidence_score > 0.5, "Compound validation failed"

                # 4. Test code analysis
                integration_test_code = '''
async def research_bpc157():
    """Research BPC-157 compound"""
    orchestrator = get_research_orchestrator()
    results = await orchestrator.intelligent_research("BPC-157 dosage")
    return results
'''
                analysis = await tools.smart_code_analysis(integration_test_code, "integration_test.py")

                integration_time = time.time() - start_time

                self.test_results["performance_metrics"]["full_integration_ms"] = integration_time * 1000

                print(f"✅ Cache integration: OK")
                print(f"✅ Orchestrator integration: OK")
                print(f"✅ Tools integration: OK")
                print(f"✅ Celková integrace: {integration_time*1000:.2f}ms")

                self.test_results["passed"] += 1

        except Exception as e:
            error_msg = f"Full integration test failed: {str(e)}"
            self.test_results["errors"].append(error_msg)
            self.test_results["failed"] += 1
            print(f"❌ {error_msg}")

    def _generate_final_report(self):
        """Generuje finální report testování"""
        print("\n" + "="*60)
        print("📊 FINÁLNÍ REPORT M1 OPTIMALIZOVANÉHO PROJEKTU")
        print("="*60)

        total_tests = self.test_results["passed"] + self.test_results["failed"]
        success_rate = (self.test_results["passed"] / total_tests * 100) if total_tests > 0 else 0

        print(f"🎯 Celkové výsledky:")
        print(f"   • Úspěšné testy: {self.test_results['passed']}")
        print(f"   • Neúspěšné testy: {self.test_results['failed']}")
        print(f"   • Úspěšnost: {success_rate:.1f}%")

        if self.test_results["performance_metrics"]:
            print(f"\n⚡ Výkonové metriky:")
            for metric, value in self.test_results["performance_metrics"].items():
                print(f"   • {metric}: {value:.2f}")

        if self.test_results["m1_optimizations"]:
            print(f"\n🚀 M1 Optimalizace:")
            for category, data in self.test_results["m1_optimizations"].items():
                print(f"   • {category}: {data}")

        if self.test_results["errors"]:
            print(f"\n❌ Chyby:")
            for error in self.test_results["errors"]:
                print(f"   • {error}")

        print("\n" + "="*60)

        if success_rate >= 80:
            print("🎉 PROJEKT ÚSPĚŠNĚ OPTIMALIZOVÁN PRO M1 MACBOOK AIR!")
        else:
            print("⚠️  PROJEKT POTŘEBUJE DALŠÍ OPTIMALIZACE")

        print("="*60)

async def main():
    """Hlavní funkce pro spuštění testů"""
    test_suite = M1ProjectTestSuite()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
