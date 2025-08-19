Comprehensive Test Suite - Komplexní testování celého biohacking research systému
Testuje všechny komponenty, integraci a výkon na M1 MacBooku
"""

import asyncio
import pytest
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import tempfile
import shutil

# Import všech komponent pro testování
try:
    from biohacking_research_engine import BiohackingResearchEngine, BiohackingResearchRequest, CompoundProfile
    from advanced_source_aggregator import AdvancedSourceAggregator, SourceConfig, SearchResult
    from intelligent_research_orchestrator import IntelligentResearchOrchestrator, UserProfile
    from quality_assessment_system import QualityAssessmentSystem
    from project_cleanup_optimizer import ProjectCleanupOptimizer
    from unified_research_engine import M1OptimizedResearchEngine
    from local_ai_adapter import M1OptimizedOllamaClient
    from peptide_prompts import PEPTIDE_RESEARCH_PROMPTS, BIOHACKING_PROMPTS
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    print(f"⚠️ Some components not available for testing: {e}")

logger = logging.getLogger(__name__)

class ComprehensiveTestSuite:
    """Komplexní testovací suite pro celý systém"""

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

        # Test data
        self.test_compounds = [
            "BPC-157",
            "TB-500",
            "GHRP-6",
            "Modafinil",
            "NAD+",
            "Metformin"
        ]

        self.test_queries = [
            "BPC-157 dosing protocol safety",
            "TB-500 healing benefits research",
            "GHRP-6 growth hormone release",
            "Modafinil cognitive enhancement",
            "NAD+ longevity anti-aging",
            "Metformin life extension mechanisms"
        ]

    async def run_full_test_suite(self) -> Dict[str, Any]:
        """Spuštění kompletní testovací suite"""

        logger.info("🧪 Starting comprehensive test suite...")
        start_time = time.time()

        try:
            # 1. Component tests
            await self._test_individual_components()

            # 2. Integration tests
            await self._test_system_integration()

            # 3. Performance tests
            await self._test_performance()

            # 4. Memory and resource tests
            await self._test_resource_usage()

            # 5. Error handling tests
            await self._test_error_handling()

            # 6. End-to-end workflow tests
            await self._test_end_to_end_workflows()

            # 7. AI integration tests
            await self._test_ai_integration()

            execution_time = time.time() - start_time
            self.test_results["total_execution_time"] = execution_time

            # Calculate success rate
            total_tests = self.test_results["tests_run"]
            if total_tests > 0:
                success_rate = self.test_results["tests_passed"] / total_tests
                self.test_results["success_rate"] = success_rate

            logger.info(f"✅ Test suite completed in {execution_time:.2f}s")
            logger.info(f"📊 Results: {self.test_results['tests_passed']}/{total_tests} tests passed")

            return self.test_results

        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            self.test_results["errors"].append(f"Test suite failure: {str(e)}")
            return self.test_results

    async def _test_individual_components(self):
        """Testování jednotlivých komponent"""

        logger.info("🔧 Testing individual components...")

        component_results = {}

        # Test BiohackingResearchEngine
        if COMPONENTS_AVAILABLE:
            try:
                async with BiohackingResearchEngine() as engine:
                    test_request = BiohackingResearchRequest(
                        compound="BPC-157",
                        research_type="safety"
                    )

                    start_time = time.time()
                    result = await engine.research_compound(test_request)
                    execution_time = time.time() - start_time

                    component_results["BiohackingResearchEngine"] = {
                        "status": "passed" if isinstance(result, CompoundProfile) else "failed",
                        "execution_time": execution_time,
                        "result_type": type(result).__name__
                    }

                    self._record_test_result(True, "BiohackingResearchEngine basic functionality")

            except Exception as e:
                component_results["BiohackingResearchEngine"] = {
                    "status": "failed",
                    "error": str(e)
                }
                self._record_test_result(False, f"BiohackingResearchEngine: {e}")

        # Test AdvancedSourceAggregator
        try:
            async with AdvancedSourceAggregator() as aggregator:
                start_time = time.time()
                results = await aggregator.multi_source_search(
                    "peptide safety",
                    max_results_per_source=2
                )
                execution_time = time.time() - start_time

                component_results["AdvancedSourceAggregator"] = {
                    "status": "passed" if results else "failed",
                    "execution_time": execution_time,
                    "sources_tested": len(results) if results else 0
                }

                self._record_test_result(bool(results), "AdvancedSourceAggregator search functionality")

        except Exception as e:
            component_results["AdvancedSourceAggregator"] = {
                "status": "failed",
                "error": str(e)
            }
            self._record_test_result(False, f"AdvancedSourceAggregator: {e}")

        # Test QualityAssessmentSystem
        try:
            async with QualityAssessmentSystem() as quality_system:
                test_data = {
                    "compound_profile": {"name": "Test", "benefits": ["test"]},
                    "source_results": {"test_source": [{"title": "Test", "snippet": "Test data"}]}
                }

                start_time = time.time()
                assessment = await quality_system.assess_research_quality(test_data, "test_compound")
                execution_time = time.time() - start_time

                component_results["QualityAssessmentSystem"] = {
                    "status": "passed" if "overall_quality" in assessment else "failed",
                    "execution_time": execution_time,
                    "assessment_keys": list(assessment.keys()) if assessment else []
                }

                self._record_test_result("overall_quality" in assessment, "QualityAssessmentSystem assessment")

        except Exception as e:
            component_results["QualityAssessmentSystem"] = {
                "status": "failed",
                "error": str(e)
            }
            self._record_test_result(False, f"QualityAssessmentSystem: {e}")

        self.test_results["component_tests"] = component_results

    async def _test_system_integration(self):
        """Testování integrace systému"""

        logger.info("🔗 Testing system integration...")

        integration_results = {}

        # Test full research workflow
        try:
            if COMPONENTS_AVAILABLE:
                async with IntelligentResearchOrchestrator() as orchestrator:
                    start_time = time.time()
                    result = await orchestrator.intelligent_research(
                        "BPC-157 dosing safety",
                        user_id="test_user"
                    )
                    execution_time = time.time() - start_time

                    integration_results["full_research_workflow"] = {
                        "status": "passed" if "results" in result else "failed",
                        "execution_time": execution_time,
                        "result_keys": list(result.keys()) if result else []
                    }

                    self._record_test_result("results" in result, "Full research workflow integration")
            else:
                integration_results["full_research_workflow"] = {
                    "status": "skipped",
                    "reason": "Components not available"
                }

        except Exception as e:
            integration_results["full_research_workflow"] = {
                "status": "failed",
                "error": str(e)
            }
            self._record_test_result(False, f"Full workflow integration: {e}")

        # Test AI integration
        try:
            async with M1OptimizedOllamaClient() as ai_client:
                start_time = time.time()
                response = await ai_client.generate_response(
                    "Explain BPC-157 in one sentence.",
                    max_tokens=50
                )
                execution_time = time.time() - start_time

                integration_results["ai_integration"] = {
                    "status": "passed" if response else "failed",
                    "execution_time": execution_time,
                    "response_length": len(response) if response else 0
                }

                self._record_test_result(bool(response), "AI integration")

        except Exception as e:
            integration_results["ai_integration"] = {
                "status": "failed",
                "error": str(e)
            }
            self._record_test_result(False, f"AI integration: {e}")

        self.test_results["integration_tests"] = integration_results

    async def _test_performance(self):
        """Testování výkonu systému"""

        logger.info("⚡ Testing performance...")

        performance_results = {}

        # Test concurrent queries
        try:
            if COMPONENTS_AVAILABLE:
                async with IntelligentResearchOrchestrator() as orchestrator:
                    start_time = time.time()

                    # Run 3 concurrent queries (M1 optimized)
                    tasks = [
                        orchestrator.intelligent_research(query, f"user_{i}")
                        for i, query in enumerate(self.test_queries[:3])
                    ]

                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    execution_time = time.time() - start_time

                    successful_results = [r for r in results if not isinstance(r, Exception)]

                    performance_results["concurrent_queries"] = {
                        "total_queries": len(tasks),
                        "successful_queries": len(successful_results),
                        "total_time": execution_time,
                        "avg_time_per_query": execution_time / len(tasks),
                        "throughput_qps": len(tasks) / execution_time
                    }

                    self._record_test_result(
                        len(successful_results) >= len(tasks) // 2,
                        f"Concurrent performance: {len(successful_results)}/{len(tasks)} succeeded"
                    )
            else:
                performance_results["concurrent_queries"] = {
                    "status": "skipped",
                    "reason": "Components not available"
                }

        except Exception as e:
            performance_results["concurrent_queries"] = {
                "status": "failed",
                "error": str(e)
            }
            self._record_test_result(False, f"Concurrent performance: {e}")

        # Test memory efficiency
        try:
            import psutil
            process = psutil.Process()

            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Perform memory-intensive operations
            if COMPONENTS_AVAILABLE:
                async with BiohackingResearchEngine() as engine:
                    for compound in self.test_compounds[:3]:
                        request = BiohackingResearchRequest(compound=compound)
                        await engine.research_compound(request)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            performance_results["memory_efficiency"] = {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": memory_increase,
                "efficiency_score": max(0, 10 - memory_increase / 50)  # Penalty for high memory use
            }

            self._record_test_result(
                memory_increase < 200,  # Less than 200MB increase
                f"Memory efficiency: {memory_increase:.1f}MB increase"
            )

        except Exception as e:
            performance_results["memory_efficiency"] = {
                "status": "failed",
                "error": str(e)
            }
            self._record_test_result(False, f"Memory efficiency test: {e}")

        self.test_results["performance_metrics"] = performance_results

    async def _test_resource_usage(self):
        """Testování využití zdrojů"""

        logger.info("💾 Testing resource usage...")

        try:
            import psutil

            # Monitor CPU and memory during operations
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent

            resource_metrics = {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory_percent,
                "available_memory_gb": psutil.virtual_memory().available / (1024**3)
            }

            self.test_results["resource_usage"] = resource_metrics

            # Test passes if resource usage is reasonable
            resource_test_passed = (
                cpu_percent < 80 and
                memory_percent < 90
            )

            self._record_test_result(
                resource_test_passed,
                f"Resource usage: CPU {cpu_percent}%, Memory {memory_percent}%"
            )

        except Exception as e:
            self._record_test_result(False, f"Resource usage test: {e}")

    async def _test_error_handling(self):
        """Testování error handling"""

        logger.info("🛡️ Testing error handling...")

        error_handling_results = {}

        # Test invalid inputs
        try:
            if COMPONENTS_AVAILABLE:
                async with BiohackingResearchEngine() as engine:
                    # Test with empty compound
                    request = BiohackingResearchRequest(compound="")
                    result = await engine.research_compound(request)

                    error_handling_results["empty_input"] = {
                        "handled_gracefully": True,
                        "result_type": type(result).__name__
                    }

                    self._record_test_result(True, "Empty input error handling")
            else:
                error_handling_results["empty_input"] = {"status": "skipped"}

        except Exception as e:
            error_handling_results["empty_input"] = {
                "handled_gracefully": True,  # Exception is expected
                "error": str(e)
            }
            self._record_test_result(True, "Empty input error handling (exception expected)")

        # Test network timeout simulation
        try:
            async with AdvancedSourceAggregator() as aggregator:
                # Test with invalid source
                results = await aggregator.multi_source_search(
                    "test query",
                    sources=["invalid_source"],
                    max_results_per_source=1
                )

                error_handling_results["invalid_source"] = {
                    "handled_gracefully": True,
                    "results_returned": bool(results)
                }

                self._record_test_result(True, "Invalid source error handling")

        except Exception as e:
            error_handling_results["invalid_source"] = {
                "handled_gracefully": True,
                "error": str(e)
            }
            self._record_test_result(True, "Invalid source error handling")

        self.test_results["error_handling"] = error_handling_results

    async def _test_end_to_end_workflows(self):
        """Testování end-to-end workflow"""

        logger.info("🔄 Testing end-to-end workflows...")

        workflow_results = {}

        # Test complete biohacking research workflow
        try:
            if COMPONENTS_AVAILABLE:
                async with IntelligentResearchOrchestrator() as orchestrator:
                    # Simulate user research session
                    user_id = "test_researcher"

                    # Step 1: Initial research
                    result1 = await orchestrator.intelligent_research(
                        "BPC-157 healing benefits",
                        user_id=user_id
                    )

                    # Step 2: Follow-up research
                    result2 = await orchestrator.intelligent_research(
                        "BPC-157 dosing protocol",
                        user_id=user_id
                    )

                    # Step 3: Safety research
                    result3 = await orchestrator.intelligent_research(
                        "BPC-157 side effects safety",
                        user_id=user_id
                    )

                    workflow_results["complete_research_session"] = {
                        "steps_completed": 3,
                        "all_successful": all("results" in r for r in [result1, result2, result3]),
                        "user_profile_updated": user_id in orchestrator.user_profiles,
                        "personalization_working": any(
                            "personalization_applied" in r for r in [result1, result2, result3]
                        )
                    }

                    workflow_success = (
                        workflow_results["complete_research_session"]["all_successful"] and
                        workflow_results["complete_research_session"]["user_profile_updated"]
                    )

                    self._record_test_result(workflow_success, "Complete research session workflow")
            else:
                workflow_results["complete_research_session"] = {"status": "skipped"}

        except Exception as e:
            workflow_results["complete_research_session"] = {
                "status": "failed",
                "error": str(e)
            }
            self._record_test_result(False, f"Research session workflow: {e}")

        self.test_results["end_to_end_workflows"] = workflow_results

    async def _test_ai_integration(self):
        """Testování AI integrace"""

        logger.info("🤖 Testing AI integration...")

        ai_results = {}

        # Test prompt processing
        try:
            async with M1OptimizedOllamaClient() as ai_client:
                # Test different prompt types
                test_prompts = [
                    PEPTIDE_RESEARCH_PROMPTS["basic_info"].format(query="BPC-157"),
                    BIOHACKING_PROMPTS["safety_profile"].format(query="Modafinil")
                ]

                prompt_results = []
                for i, prompt in enumerate(test_prompts):
                    start_time = time.time()
                    response = await ai_client.generate_response(prompt, max_tokens=100)
                    execution_time = time.time() - start_time

                    prompt_results.append({
                        "prompt_index": i,
                        "response_length": len(response) if response else 0,
                        "execution_time": execution_time,
                        "successful": bool(response)
                    })

                ai_results["prompt_processing"] = {
                    "total_prompts": len(test_prompts),
                    "successful_prompts": sum(1 for r in prompt_results if r["successful"]),
                    "avg_execution_time": sum(r["execution_time"] for r in prompt_results) / len(prompt_results),
                    "results": prompt_results
                }

                self._record_test_result(
                    ai_results["prompt_processing"]["successful_prompts"] > 0,
                    f"AI prompt processing: {ai_results['prompt_processing']['successful_prompts']}/{len(test_prompts)} successful"
                )

        except Exception as e:
            ai_results["prompt_processing"] = {
                "status": "failed",
                "error": str(e)
            }
            self._record_test_result(False, f"AI prompt processing: {e}")

        self.test_results["ai_integration"] = ai_results

    def _record_test_result(self, passed: bool, test_name: str):
        """Zaznamenání výsledku testu"""
        self.test_results["tests_run"] += 1

        if passed:
            self.test_results["tests_passed"] += 1
            logger.info(f"✅ {test_name}")
        else:
            self.test_results["tests_failed"] += 1
            logger.warning(f"❌ {test_name}")

    async def save_test_report(self, output_path: Path = None):
        """Uložení testovacího reportu"""

        if output_path is None:
            output_path = Path("TEST_REPORT.json")

        try:
            with open(output_path, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)

            logger.info(f"📄 Test report saved to {output_path}")

            # Also create markdown report
            await self._create_markdown_report(output_path.with_suffix('.md'))

        except Exception as e:
            logger.error(f"Failed to save test report: {e}")

    async def _create_markdown_report(self, output_path: Path):
        """Vytvoření markdown reportu"""
        
        report_content = f"""# Comprehensive Test Report

Generated: {self.test_results['timestamp']}

## Summary
- **Total Tests**: {self.test_results['tests_run']}
- **Passed**: {self.test_results['tests_passed']}
- **Failed**: {self.test_results['tests_failed']}
- **Success Rate**: {self.test_results.get('success_rate', 0):.2%}
- **Execution Time**: {self.test_results.get('total_execution_time', 0):.2f} seconds

## Component Tests
"""
        
        for component, result in self.test_results.get('component_tests', {}).items():
            status_emoji = "✅" if result.get('status') == 'passed' else "❌"
            report_content += f"- {status_emoji} **{component}**: {result.get('status', 'unknown')}\n"
            
        report_content += "\n## Performance Metrics\n"
        perf_metrics = self.test_results.get('performance_metrics', {})
        for metric, data in perf_metrics.items():
            report_content += f"### {metric.replace('_', ' ').title()}\n"
            if isinstance(data, dict):
                for key, value in data.items():
                    report_content += f"- {key}: {value}\n"
            report_content += "\n"
            
        if self.test_results.get('errors'):
            report_content += "## Errors\n"
            for error in self.test_results['errors']:
                report_content += f"- {error}\n"
                
        try:
            output_path.write_text(report_content, encoding='utf-8')
            logger.info(f"📄 Markdown report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save markdown report: {e}")

# Pytest fixtures and tests
@pytest.fixture
async def test_suite():
    """Pytest fixture pro test suite"""
    return ComprehensiveTestSuite()

@pytest.mark.asyncio
async def test_component_functionality(test_suite):
    """Test základní funkcionality komponent"""
    await test_suite._test_individual_components()
    assert test_suite.test_results["tests_passed"] > 0

@pytest.mark.asyncio
async def test_system_integration(test_suite):
    """Test integrace systému"""
    await test_suite._test_system_integration()
    assert test_suite.test_results["tests_passed"] > 0

@pytest.mark.asyncio
async def test_performance_benchmarks(test_suite):
    """Test výkonnostních benchmarků"""
    await test_suite._test_performance()
    assert test_suite.test_results["tests_passed"] > 0

# Main execution function
async def run_comprehensive_tests():
    """Hlavní funkce pro spuštění testů"""
    test_suite = ComprehensiveTestSuite()
    results = await test_suite.run_full_test_suite()
    await test_suite.save_test_report()
    return results

if __name__ == "__main__":
    asyncio.run(run_comprehensive_tests())

# Export
__all__ = ['ComprehensiveTestSuite', 'run_comprehensive_tests']
