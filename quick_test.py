#!/usr/bin/env python3
"""
Quick Test Script - Rychlé testování nových komponent
"""
import sys
import traceback

def test_imports():
    """Test základních importů"""
    print("🧪 Testing component imports...")

    components = [
        ("biohacking_research_engine", "BiohackingResearchEngine"),
        ("advanced_source_aggregator", "AdvancedSourceAggregator"),
        ("intelligent_research_orchestrator", "IntelligentResearchOrchestrator"),
        ("quality_assessment_system", "QualityAssessmentSystem"),
        ("project_cleanup_optimizer", "ProjectCleanupOptimizer"),
        ("peptide_prompts", "PEPTIDE_RESEARCH_PROMPTS")
    ]

    results = {"passed": 0, "failed": 0, "errors": []}

    for module_name, class_name in components:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name} imported successfully")
            results["passed"] += 1
        except Exception as e:
            print(f"❌ {module_name} import failed: {e}")
            results["failed"] += 1
            results["errors"].append(f"{module_name}: {str(e)}")

    return results

def main():
    print(f"Python version: {sys.version}")
    print("=" * 60)

    # Test imports
    import_results = test_imports()

    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"✅ Passed: {import_results['passed']}")
    print(f"❌ Failed: {import_results['failed']}")

    if import_results['errors']:
        print("\n🚨 Errors:")
        for error in import_results['errors']:
            print(f"  - {error}")

    print("\n🎯 Component testing completed!")

    return import_results

if __name__ == "__main__":
    main()
