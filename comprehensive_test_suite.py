"""
Updated Comprehensive Test Suite - Aktualizované testování pro unified systém
Testuje nové komponenty, integraci a výkon po optimalizaci
"""

import asyncio
import pytest
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Import updated components
try:
    from main import UnifiedBiohackingResearchTool
    from enhanced_research_orchestrator import EnhancedIntelligentOrchestrator
    from unified_cache_system import get_unified_cache
    from adaptive_learning_system import EnhancedAdaptiveLearningSystem
    from quality_assessment_system import QualityAssessmentSystem
    from local_ai_adapter import M1OptimizedOllamaClient
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    print(f"⚠️ Some components not available for testing: {e}")

logger = logging.getLogger(__name__)

class UpdatedComprehensiveTestSuite:
    """Aktualizovaná testovací suite pro unified systém"""

    def __init__(self):
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "performance_metrics": {},
            "component_tests": {},
            "integration_tests": {},
            "errors": []
        }

        # Updated test data for biohacking focus
        self.test_compounds = [
            "BPC-157",
            "TB-500",
            "GHRP-6",
            "Modafinil",
            "NAD+",
            "Metformin",
            "Rapamycin",
            "NMN"
        ]

        self.test_queries = [
            "BPC-157 dosing protocol safety",
            "TB-500 healing benefits research",
            "GHRP-6 growth hormone release",
            "Modafinil cognitive enhancement mechanisms",
            "NAD+ longevity anti-aging protocols",
            "Metformin life extension dosing",
            "Rapamycin autophagy activation",
            "NMN bioavailability administration"
        ]

    async def run_updated_test_suite(self) -> Dict[str, Any]:
        """Spuštění aktualizované testovací suite"""

        logger.info("🧪 Starting updated comprehensive test suite...")
        start_time = time.time()

        try:
            # 1. Test unified main interface
            await self._test_unified_main_interface()

            # 2. Test enhanced orchestrator
            await self._test_enhanced_orchestrator()

            # 3. Test adaptive learning system
            await self._test_adaptive_learning()

            # 4. Test unified cache system
            await self._test_unified_cache()

            # 5. Test quality assessment
            await self._test_quality_assessment()

            # 6. Test performance optimizations
            await self._test_performance_optimizations()

            # 7. Test error handling and edge cases
            await self._test_error_handling()

            # 8. Test personalization features
            await self._test_personalization()

            execution_time = time.time() - start_time
            self.test_results["total_execution_time"] = execution_time

            # Calculate success rate
            total_tests = self.test_results["tests_run"]
            if total_tests > 0:
                success_rate = self.test_results["tests_passed"] / total_tests
                self.test_results["success_rate"] = success_rate

            logger.info(f"✅ Updated test suite completed in {execution_time:.2f}s")
            logger.info(f"📊 Results: {self.test_results['tests_passed']}/{total_tests} tests passed")

            return self.test_results

        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            self.test_results["errors"].append(f"Test suite failure: {str(e)}")
            return self.test_results

    async def _test_unified_main_interface(self):
        """Test hlavního unified rozhraní"""

        logger.info("🔧 Testing unified main interface...")

        test_results = {}

        if COMPONENTS_AVAILABLE:
            try:
                # Test basic research functionality
                async with UnifiedBiohackingResearchTool("test_user") as tool:
                    start_time = time.time()

                    result = await tool.research(
                        "BPC-157 basic information",
                        research_type="quick",
                        output_format="brief"
                    )

                    execution_time = time.time() - start_time

                    test_results["basic_research"] = {
                        "status": "passed" if not result.get("error") else "failed",
                        "execution_time": execution_time,
                        "has_results": "research_results" in result,
                        "personalized": result.get("personalization_applied", False)
                    }

                    self._record_test_result(
                        not result.get("error"),
                        "Unified main interface basic research"
                    )

                # Test peptide-specific research
                async with UnifiedBiohackingResearchTool("test_user") as tool:
                    start_time = time.time()

                    peptide_result = await tool.peptide_research(
                        "TB-500",
                        research_focus="dosage"
                    )

                    execution_time = time.time() - start_time

                    test_results["peptide_research"] = {
                        "status": "passed" if not peptide_result.get("error") else "failed",
                        "execution_time": execution_time,
                        "specialized": peptide_result.get("peptide_research", {}).get("specialized_analysis", False)
                    }

                    self._record_test_result(
                        not peptide_result.get("error"),
                        "Unified main interface peptide research"
                    )

            except Exception as e:
                test_results["unified_interface"] = {"status": "failed", "error": str(e)}
                self._record_test_result(False, f"Unified main interface: {e}")
        else:
            test_results["unified_interface"] = {"status": "skipped", "reason": "Components not available"}

        self.test_results["component_tests"]["unified_main_interface"] = test_results

    async def _test_enhanced_orchestrator(self):
        """Test enhanced research orchestrátoru"""

        logger.info("🧠 Testing enhanced research orchestrator...")

        test_results = {}

        try:
            async with EnhancedIntelligentOrchestrator() as orchestrator:
                start_time = time.time()

                # Test AI-powered research
                result = await orchestrator.intelligent_research(
                    "GHRP-6 growth hormone effects",
                    user_id="test_user",
                    mode="balanced_research",
                    generate_insights=True
                )

                execution_time = time.time() - start_time

                test_results["ai_research"] = {
                    "status": "passed" if result.get("success") else "failed",
                    "execution_time": execution_time,
                    "ai_enhanced": result.get("ai_enhanced", False),
                    "has_insights": bool(result.get("predictive_insights")),
                    "quality_assessed": "quality_assessment" in result
                }

                self._record_test_result(
                    result.get("success", False),
                    "Enhanced orchestrator AI research"
                )

                # Test different research modes
                for mode in ["quick_overview", "deep_analysis", "safety_focused"]:
                    try:
                        mode_result = await orchestrator.intelligent_research(
                            f"Modafinil {mode} test",
                            user_id="test_user",
                            mode=mode,
                            generate_insights=False
                        )

                        test_results[f"mode_{mode}"] = {
                            "status": "passed" if mode_result.get("success") else "failed",
                            "mode": mode_result.get("research_mode")
                        }

                        self._record_test_result(
                            mode_result.get("success", False),
                            f"Enhanced orchestrator {mode} mode"
                        )

                    except Exception as e:
                        test_results[f"mode_{mode}"] = {"status": "failed", "error": str(e)}
                        self._record_test_result(False, f"Enhanced orchestrator {mode}: {e}")

        except Exception as e:
            test_results["enhanced_orchestrator"] = {"status": "failed", "error": str(e)}
            self._record_test_result(False, f"Enhanced orchestrator: {e}")

        self.test_results["component_tests"]["enhanced_orchestrator"] = test_results

    async def _test_adaptive_learning(self):
        """Test adaptivního learning systému"""

        logger.info("📚 Testing adaptive learning system...")

        test_results = {}

        try:
            async with EnhancedAdaptiveLearningSystem() as learning_system:
                # Test learning from interactions
                user_id = "test_learner"

                test_interaction = {
                    "query": "BPC-157 safety profile",
                    "response": {"quality_score": 8.5, "success": True},
                    "user_feedback": {"helpful": True, "detail_level": "appropriate"},
                    "success_metrics": {"response_time": 25.0, "completeness": 0.9}
                }

                start_time = time.time()

                insights = await learning_system.learn_from_interaction(
                    user_id=user_id,
                    query=test_interaction["query"],
                    response=test_interaction["response"],
                    user_feedback=test_interaction["user_feedback"],
                    success_metrics=test_interaction["success_metrics"]
                )

                execution_time = time.time() - start_time

                test_results["learning_interaction"] = {
                    "status": "passed" if insights is not None else "failed",
                    "execution_time": execution_time,
                    "has_insights": bool(insights),
                    "insights_count": len(insights) if insights else 0
                }

                self._record_test_result(
                    insights is not None,
                    "Adaptive learning from interaction"
                )

                # Test personalized prompt generation
                personalized_prompt = await learning_system.generate_personalized_prompt(
                    user_id=user_id,
                    prompt_type="research",
                    topic="TB-500"
                )

                test_results["personalized_prompts"] = {
                    "status": "passed" if personalized_prompt else "failed",
                    "has_adaptations": bool(personalized_prompt.user_adaptations) if personalized_prompt else False,
                    "complexity_level": personalized_prompt.complexity_level if personalized_prompt else None
                }

                self._record_test_result(
                    personalized_prompt is not None,
                    "Adaptive learning personalized prompts"
                )

        except Exception as e:
            test_results["adaptive_learning"] = {"status": "failed", "error": str(e)}
            self._record_test_result(False, f"Adaptive learning: {e}")

        self.test_results["component_tests"]["adaptive_learning"] = test_results

    async def _test_unified_cache(self):
        """Test unified cache systému"""

        logger.info("💾 Testing unified cache system...")

        test_results = {}

        try:
            cache = get_unified_cache()

            # Test basic cache operations
            test_key = "test_cache_key"
            test_data = {"test": "data", "timestamp": time.time()}

            # Test set operation
            set_success = await cache.set(test_key, test_data, ttl=60)

            # Test get operation
            retrieved_data = await cache.get(test_key)

            # Test cache hit
            cache_hit = retrieved_data is not None and retrieved_data.get("test") == "data"

            test_results["basic_operations"] = {
                "set_success": set_success,
                "get_success": retrieved_data is not None,
                "cache_hit": cache_hit,
                "data_integrity": retrieved_data == test_data if retrieved_data else False
            }

            self._record_test_result(
                set_success and cache_hit,
                "Unified cache basic operations"
            )

            # Test cache statistics
            stats = cache.get_stats()

            test_results["cache_stats"] = {
                "has_stats": bool(stats),
                "has_hit_rate": "hit_rate_percent" in stats,
                "has_memory_info": "items_in_memory" in stats
            }

            self._record_test_result(
                bool(stats),
                "Unified cache statistics"
            )

            # Test cache cleanup
            deleted = await cache.delete(test_key)

            test_results["cleanup"] = {
                "delete_success": deleted,
                "cleanup_verified": await cache.get(test_key) is None
            }

            self._record_test_result(
                deleted,
                "Unified cache cleanup"
            )

        except Exception as e:
            test_results["unified_cache"] = {"status": "failed", "error": str(e)}
            self._record_test_result(False, f"Unified cache: {e}")

        self.test_results["component_tests"]["unified_cache"] = test_results

    async def _test_performance_optimizations(self):
        """Test výkonnostních optimalizací"""

        logger.info("⚡ Testing performance optimizations...")

        performance_results = {}

        # Test concurrent research capabilities
        try:
            if COMPONENTS_AVAILABLE:
                async with UnifiedBiohackingResearchTool("perf_user") as tool:
                    start_time = time.time()

                    # Test 2 concurrent quick researches (M1 optimized)
                    tasks = [
                        tool.research(query, research_type="quick", output_format="brief")
                        for query in self.test_queries[:2]
                    ]

                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    execution_time = time.time() - start_time

                    successful_results = [r for r in results if not isinstance(r, Exception) and not r.get("error")]

                    performance_results["concurrent_research"] = {
                        "total_queries": len(tasks),
                        "successful_queries": len(successful_results),
                        "total_time": execution_time,
                        "avg_time_per_query": execution_time / len(tasks),
                        "throughput_qps": len(tasks) / execution_time,
                        "success_rate": len(successful_results) / len(tasks)
                    }

                    self._record_test_result(
                        len(successful_results) >= len(tasks) // 2,
                        f"Performance concurrent research: {len(successful_results)}/{len(tasks)} succeeded"
                    )
            else:
                performance_results["concurrent_research"] = {"status": "skipped"}

        except Exception as e:
            performance_results["concurrent_research"] = {"status": "failed", "error": str(e)}
            self._record_test_result(False, f"Performance concurrent research: {e}")

        # Test memory efficiency
        try:
            import psutil
            process = psutil.Process()

            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Perform memory-intensive operations
            if COMPONENTS_AVAILABLE:
                async with UnifiedBiohackingResearchTool("memory_user") as tool:
                    for compound in self.test_compounds[:3]:
                        await tool.peptide_research(compound, "safety")

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            performance_results["memory_efficiency"] = {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": memory_increase,
                "efficiency_score": max(0, 10 - memory_increase / 50),  # Penalty for high memory use
                "within_target": memory_increase < 200  # Target: less than 200MB increase
            }

            self._record_test_result(
                memory_increase < 200,
                f"Memory efficiency: {memory_increase:.1f}MB increase"
            )

        except Exception as e:
            performance_results["memory_efficiency"] = {"status": "failed", "error": str(e)}
            self._record_test_result(False, f"Memory efficiency test: {e}")

        self.test_results["performance_metrics"] = performance_results

    async def _test_personalization(self):
        """Test personalizačních funkcí"""

        logger.info("👤 Testing personalization features...")

        personalization_results = {}

        try:
            if COMPONENTS_AVAILABLE:
                # Test with different user profiles
                user_profiles = [
                    {"user_id": "beginner_user", "complexity": "beginner"},
                    {"user_id": "expert_user", "complexity": "expert"}
                ]

                for profile in user_profiles:
                    try:
                        async with UnifiedBiohackingResearchTool(profile["user_id"]) as tool:
                            result = await tool.research(
                                "BPC-157 mechanisms",
                                research_type="balanced",
                                output_format="detailed"
                            )

                            personalization_results[profile["user_id"]] = {
                                "status": "passed" if not result.get("error") else "failed",
                                "personalized": result.get("personalization_applied", False),
                                "has_user_context": profile["user_id"] in str(result)
                            }

                            self._record_test_result(
                                not result.get("error"),
                                f"Personalization for {profile['user_id']}"
                            )

                    except Exception as e:
                        personalization_results[profile["user_id"]] = {"status": "failed", "error": str(e)}
                        self._record_test_result(False, f"Personalization {profile["user_id"]}: {e}")

        except Exception as e:
            personalization_results["personalization"] = {"status": "failed", "error": str(e)}
            self._record_test_result(False, f"Personalization test: {e}")

        self.test_results["integration_tests"]["personalization"] = personalization_results

    def _record_test_result(self, passed: bool, test_name: str):
        """Zaznamenání výsledku testu"""
        self.test_results["tests_run"] += 1

        if passed:
            self.test_results["tests_passed"] += 1
            logger.info(f"✅ {test_name}")
        else:
            self.test_results["tests_failed"] += 1
            logger.warning(f"❌ {test_name}")

# Pytest fixtures and tests
@pytest.fixture
async def test_suite():
    """Pytest fixture pro test suite"""
    return UpdatedComprehensiveTestSuite()

@pytest.mark.asyncio
async def test_component_functionality(test_suite):
    """Test základní funkcionality komponent"""
    await test_suite._test_unified_main_interface()
    await test_suite._test_enhanced_orchestrator()
    await test_suite._test_adaptive_learning()
    await test_suite._test_unified_cache()
    assert test_suite.test_results["tests_passed"] > 0

@pytest.mark.asyncio
async def test_system_integration(test_suite):
    """Test integrace systému"""
    await test_suite._test_personalization()
    assert test_suite.test_results["tests_passed"] > 0

@pytest.mark.asyncio
async def test_performance_benchmarks(test_suite):
    """Test výkonnostních benchmarků"""
    await test_suite._test_performance_optimizations()
    assert test_suite.test_results["tests_passed"] > 0

# Main execution function
async def run_updated_comprehensive_tests():
    """Hlavní funkce pro spuštění aktualizovaných testů"""
    test_suite = UpdatedComprehensiveTestSuite()
    results = await test_suite.run_updated_test_suite()
    await test_suite.save_test_report()
    return results

if __name__ == "__main__":
    asyncio.run(run_updated_comprehensive_tests())

# Export
__all__ = ['UpdatedComprehensiveTestSuite', 'run_updated_comprehensive_tests']
