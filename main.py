#!/usr/bin/env python3
"""
Unified Main Entry Point - Sjednocený vstupní bod pro Advanced Biohacking Research Tool
Integruje všechny funkce: lokální AI, peptidový výzkum, M1 optimalizace a pokročilé analýzy
Senior IT specialist optimalizovaná verze - vše v jednom
"""

import asyncio
import argparse
import logging
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# Unified imports with intelligent fallbacks
try:
    from intelligent_research_orchestrator import IntelligentResearchOrchestrator, UserProfile
    from biohacking_research_engine import BiohackingResearchEngine, BiohackingResearchRequest
    from advanced_source_aggregator import AdvancedSourceAggregator
    from quality_assessment_system import QualityAssessmentSystem
    from local_ai_adapter import M1OptimizedOllamaClient, quick_ai_query
    from unified_config import get_config, Environment
    from peptide_prompts import PEPTIDE_RESEARCH_PROMPTS, BIOHACKING_PROMPTS
    ADVANCED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Advanced components not available: {e}")
    ADVANCED_COMPONENTS_AVAILABLE = False

# Fallback imports for basic functionality
try:
    from academic_scraper import create_scraping_orchestrator
    from cache_manager import get_cache_manager
    BASIC_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Basic components not available: {e}")
    BASIC_COMPONENTS_AVAILABLE = False

# Setup enhanced logging
def setup_logging(verbose: bool = False) -> None:
    """Setup centralized logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('research_tool.log')
        ]
    )

logger = logging.getLogger(__name__)

class UnifiedBiohackingResearchTool:
    """
    Sjednocený nástroj pro pokročilý biohacking výzkum
    Kombinuje všechny funkce v jednom optimalizovaném rozhraní
    """

    def __init__(self, user_id: str = "default_user", verbose: bool = False):
        setup_logging(verbose)
        self.user_id = user_id
        self.verbose = verbose

        # Initialize components
        self.orchestrator = None
        self.research_engine = None
        self.quality_system = None
        self.cache_manager = None

        # Performance tracking
        self.performance_stats = {
            "queries_processed": 0,
            "total_time": 0.0,
            "avg_response_time": 0.0,
            "cache_hits": 0,
            "errors": 0
        }

        logger.info(f"🧬 Unified Biohacking Research Tool initialized for user: {user_id}")

    async def __aenter__(self):
        """Async context manager entry with intelligent component initialization"""
        try:
            if ADVANCED_COMPONENTS_AVAILABLE:
                # Initialize advanced orchestrator
                self.orchestrator = IntelligentResearchOrchestrator()
                await self.orchestrator.__aenter__()

                # Initialize quality assessment
                self.quality_system = QualityAssessmentSystem()
                await self.quality_system.__aenter__()

                logger.info("✅ Advanced components initialized")
            else:
                logger.warning("⚠️ Running in basic mode - advanced features disabled")

            # Initialize cache (always try to enable)
            if BASIC_COMPONENTS_AVAILABLE:
                self.cache_manager = get_cache_manager()

        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup all components"""
        if self.orchestrator:
            await self.orchestrator.__aexit__(exc_type, exc_val, exc_tb)
        if self.quality_system:
            await self.quality_system.__aexit__(exc_type, exc_val, exc_tb)

    async def research(
        self,
        query: str,
        research_type: str = "comprehensive",
        evidence_level: str = "high",
        include_safety: bool = True,
        output_format: str = "detailed"
    ) -> Dict[str, Any]:
        """
        Unified research method with intelligent mode selection

        Args:
            query: Research query (e.g., "BPC-157 dosing protocol")
            research_type: "quick", "balanced", "comprehensive", "safety", "dosage", "stacking"
            evidence_level: "high", "medium", "all"
            include_safety: Include safety assessment
            output_format: "brief", "detailed", "expert", "json"
        """
        start_time = time.time()
        self.performance_stats["queries_processed"] += 1

        logger.info(f"🔬 Starting {research_type} research: '{query}'")

        try:
            # Use advanced orchestrator if available
            if self.orchestrator:
                result = await self._advanced_research(
                    query, research_type, evidence_level, include_safety
                )
            else:
                result = await self._basic_research(query, research_type)

            # Add quality assessment if available
            if self.quality_system and "research_data" in result:
                quality_assessment = await self.quality_system.assess_research_quality(
                    result["research_data"], query
                )
                result["quality_assessment"] = quality_assessment

            # Format output
            formatted_result = self._format_output(result, output_format)

            # Update performance stats
            execution_time = time.time() - start_time
            self.performance_stats["total_time"] += execution_time
            self.performance_stats["avg_response_time"] = (
                self.performance_stats["total_time"] / self.performance_stats["queries_processed"]
            )

            formatted_result["performance"] = {
                "execution_time": execution_time,
                "query_number": self.performance_stats["queries_processed"],
                "avg_response_time": self.performance_stats["avg_response_time"]
            }

            logger.info(f"✅ Research completed in {execution_time:.2f}s")
            return formatted_result

        except Exception as e:
            self.performance_stats["errors"] += 1
            logger.error(f"❌ Research failed: {e}")
            return {
                "error": str(e),
                "query": query,
                "execution_time": time.time() - start_time,
                "success": False
            }

    async def _advanced_research(
        self,
        query: str,
        research_type: str,
        evidence_level: str,
        include_safety: bool
    ) -> Dict[str, Any]:
        """Advanced research using intelligent orchestrator"""

        # Use intelligent orchestrator with personalization
        result = await self.orchestrator.intelligent_research(
            query=query,
            user_id=self.user_id
        )

        return {
            "method": "advanced_orchestrator",
            "research_data": result,
            "personalized": True,
            "ai_enhanced": True
        }

    async def _basic_research(self, query: str, research_type: str) -> Dict[str, Any]:
        """Basic research using simple scraping"""

        if not BASIC_COMPONENTS_AVAILABLE:
            return {
                "error": "No research components available",
                "suggestion": "Please install required dependencies"
            }

        scraper = await create_scraping_orchestrator(max_concurrent=2, timeout=30)

        # Basic sources for fallback
        basic_sources = ["wikipedia", "openalex"]
        results = {}

        for source in basic_sources:
            try:
                source_results = await scraper.scrape_source(
                    source=source,
                    query=query,
                    max_results=3
                )
                results[source] = source_results
            except Exception as e:
                logger.warning(f"Source {source} failed: {e}")

        return {
            "method": "basic_scraping",
            "research_data": {"source_results": results},
            "personalized": False,
            "ai_enhanced": False
        }

    def _format_output(self, result: Dict[str, Any], output_format: str) -> Dict[str, Any]:
        """Format output based on requested format"""

        if output_format == "json":
            return result
        elif output_format == "brief":
            return self._create_brief_summary(result)
        elif output_format == "expert":
            return self._create_expert_report(result)
        else:  # detailed
            return self._create_detailed_report(result)

    def _create_brief_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create brief summary for quick overview"""
        summary = {
            "query": result.get("research_data", {}).get("query", "Unknown"),
            "method": result.get("method", "unknown"),
            "success": "error" not in result,
            "key_findings": [],
            "safety_notes": [],
            "recommendations": []
        }

        # Extract key findings from advanced results
        if "research_data" in result and "results" in result["research_data"]:
            research_results = result["research_data"]["results"]
            if isinstance(research_results, dict) and "ai_synthesis" in research_results:
                summary["key_findings"] = ["AI synthesis available"]

        # Extract safety information
        if "quality_assessment" in result:
            safety_assessment = result["quality_assessment"].get("safety_assessment", {})
            if safety_assessment.get("warnings"):
                summary["safety_notes"] = safety_assessment["warnings"]

        return summary

    def _create_detailed_report(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed research report"""
        return {
            **result,
            "report_type": "detailed",
            "generated_at": datetime.now().isoformat(),
            "user_id": self.user_id,
            "stats": self.performance_stats
        }

    def _create_expert_report(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create expert-level report with technical details"""
        expert_report = self._create_detailed_report(result)

        # Add technical metadata
        expert_report["technical_details"] = {
            "components_used": {
                "advanced_orchestrator": bool(self.orchestrator),
                "quality_assessment": bool(self.quality_system),
                "cache_manager": bool(self.cache_manager)
            },
            "performance_analysis": self.performance_stats,
            "system_capabilities": {
                "advanced_components": ADVANCED_COMPONENTS_AVAILABLE,
                "basic_components": BASIC_COMPONENTS_AVAILABLE
            }
        }

        return expert_report

    async def peptide_research(
        self,
        peptide_name: str,
        research_focus: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Specialized peptide research with enhanced prompts"""

        # Map research focus to specialized queries
        focus_queries = {
            "dosage": f"{peptide_name} dosing protocol administration frequency",
            "safety": f"{peptide_name} safety profile side effects contraindications",
            "mechanisms": f"{peptide_name} mechanism of action receptor binding",
            "stacking": f"{peptide_name} stacking combinations synergistic effects",
            "comprehensive": f"{peptide_name} complete research dosing safety mechanisms benefits"
        }

        query = focus_queries.get(research_focus, focus_queries["comprehensive"])

        logger.info(f"🧬 Specialized peptide research: {peptide_name} (focus: {research_focus})")

        result = await self.research(
            query=query,
            research_type="comprehensive",
            evidence_level="high",
            include_safety=True,
            output_format="detailed"
        )

        # Add peptide-specific metadata
        result["peptide_research"] = {
            "peptide_name": peptide_name,
            "research_focus": research_focus,
            "specialized_analysis": True
        }

        return result

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            **self.performance_stats,
            "components_status": {
                "advanced_available": ADVANCED_COMPONENTS_AVAILABLE,
                "basic_available": BASIC_COMPONENTS_AVAILABLE,
                "orchestrator_active": bool(self.orchestrator),
                "quality_system_active": bool(self.quality_system)
            }
        }

# CLI Interface
async def main():
    """Main CLI interface with comprehensive argument parsing"""
    parser = argparse.ArgumentParser(
        description="Advanced Biohacking Research Tool - Unified Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s research "BPC-157 dosing protocol"
  %(prog)s peptide BPC-157 --focus dosage
  %(prog)s research "Modafinil cognitive enhancement" --type comprehensive --evidence high
  %(prog)s peptide "TB-500" --focus safety --format expert
  %(prog)s research "NAD+ longevity" --output results.json
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Research command
    research_parser = subparsers.add_parser('research', help='General research query')
    research_parser.add_argument('query', help='Research query')
    research_parser.add_argument('--type', choices=['quick', 'balanced', 'comprehensive', 'safety'],
                                default='comprehensive', help='Research type')
    research_parser.add_argument('--evidence', choices=['high', 'medium', 'all'],
                                default='high', help='Evidence level required')
    research_parser.add_argument('--format', choices=['brief', 'detailed', 'expert', 'json'],
                                default='detailed', help='Output format')
    research_parser.add_argument('--no-safety', action='store_true', help='Skip safety assessment')
    research_parser.add_argument('--output', help='Output file path')

    # Peptide command
    peptide_parser = subparsers.add_parser('peptide', help='Specialized peptide research')
    peptide_parser.add_argument('name', help='Peptide name (e.g., BPC-157)')
    peptide_parser.add_argument('--focus', choices=['dosage', 'safety', 'mechanisms', 'stacking', 'comprehensive'],
                               default='comprehensive', help='Research focus')
    peptide_parser.add_argument('--format', choices=['brief', 'detailed', 'expert', 'json'],
                               default='detailed', help='Output format')
    peptide_parser.add_argument('--output', help='Output file path')

    # Performance command
    perf_parser = subparsers.add_parser('performance', help='Show performance statistics')

    # Global options
    parser.add_argument('--user-id', default='cli_user', help='User ID for personalization')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize tool
    async with UnifiedBiohackingResearchTool(args.user_id, args.verbose) as tool:

        if args.command == 'research':
            result = await tool.research(
                query=args.query,
                research_type=args.type,
                evidence_level=args.evidence,
                include_safety=not args.no_safety,
                output_format=args.format
            )

        elif args.command == 'peptide':
            result = await tool.peptide_research(
                peptide_name=args.name,
                research_focus=args.focus
            )

        elif args.command == 'performance':
            result = tool.get_performance_stats()

        else:
            print(f"Unknown command: {args.command}")
            return

        # Output results
        if hasattr(args, 'output') and args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            print(f"📄 Results saved to: {output_path}")
        else:
            if args.format == 'json' or args.command == 'performance':
                print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            else:
                # Pretty print for human consumption
                print("\n" + "="*80)
                print(f"🔬 Research Results for: {result.get('research_data', {}).get('query', 'Unknown')}")
                print("="*80)

                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"✅ Success: Research completed")
                    print(f"📊 Method: {result.get('method', 'unknown')}")
                    print(f"⚡ Time: {result.get('performance', {}).get('execution_time', 0):.2f}s")

                    if result.get('quality_assessment'):
                        quality = result['quality_assessment']
                        print(f"🎯 Quality Score: {quality.get('overall_quality', 0):.1f}/10")
                        print(f"🛡️ Reliability: {quality.get('reliability_score', 0):.1f}/10")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Research interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
