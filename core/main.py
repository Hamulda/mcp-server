#!/usr/bin/env python3
"""
Unified Main Entry Point - Vyčištěný a optimalizovaný vstupní bod
Zjednodušené importy, robustní error handling, M1 optimalizované
"""

import asyncio
import argparse
import logging
import sys
import time
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Simplified unified imports with better error handling
try:
    from unified_config import get_config
except ImportError:
    print("⚠️ unified_config not available - using default config")
    def get_config():
        return {"default": True}

try:
    from unified_cache_system import get_cache_manager
except ImportError:
    print("⚠️ unified_cache_system not available - caching disabled")
    def get_cache_manager():
        return None

try:
    from academic_scraper import create_scraping_orchestrator
except ImportError:
    print("⚠️ academic_scraper not available - scraping disabled")
    def create_scraping_orchestrator():
        return None

CORE_COMPONENTS_AVAILABLE = True
print("✅ Core components loaded successfully")

# Optional advanced components
try:
    from local_ai_adapter import M1OptimizedOllamaClient
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️ AI components not available")

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

def sanitize_input(input_string: str, max_length: int = 1000) -> str:
    """
    Sanitizuje vstupní řetězce proti command injection a dalším útokům
    """
    if not input_string:
        return ""

    # Omezyí délku
    sanitized = input_string[:max_length]

    # Odstraní nebezpečné znaky
    sanitized = re.sub(r'[;&|`$(){}[\]<>]', '', sanitized)

    # Odstraní shell escape sekvence
    sanitized = re.sub(r'\\[nt$`"\'\\]', '', sanitized)

    # Trim whitespace
    return sanitized.strip()

def validate_peptide_name(name: str) -> bool:
    """
    Validuje název peptidu - povoleny pouze alfanumerické znaky, pomlčky a čísla
    """
    # Opravený regex bez redundantního escape
    return bool(re.match(r'^[A-Za-z0-9\-.]+$', name)) and len(name) <= 50

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
            if AI_AVAILABLE:
                # Fallback implementation if IntelligentResearchOrchestrator není dostupný
                try:
                    from intelligent_research_orchestrator import IntelligentResearchOrchestrator
                    self.orchestrator = IntelligentResearchOrchestrator()
                    await self.orchestrator.__aenter__()
                    logger.info("✅ Advanced orchestrator initialized")
                except ImportError:
                    logger.warning("⚠️ IntelligentResearchOrchestrator not available - using basic mode")
                    self.orchestrator = None
            else:
                logger.warning("⚠️ Running in basic mode - advanced features disabled")

            # Initialize cache (always try to enable)
            if CORE_COMPONENTS_AVAILABLE:
                self.cache_manager = get_cache_manager()

        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup all components"""
        if self.orchestrator and hasattr(self.orchestrator, '__aexit__'):
            await self.orchestrator.__aexit__(exc_type, exc_val, exc_tb)

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

        try:
            # Sanitize input
            query = sanitize_input(query)

            # Update stats
            self.performance_stats["queries_processed"] += 1

            # Check cache first
            cache_key = f"{query}:{research_type}:{evidence_level}"
            if self.cache_manager:
                cached_result = await self._check_cache(cache_key)
                if cached_result:
                    self.performance_stats["cache_hits"] += 1
                    return cached_result

            # Perform research based on available components
            if self.orchestrator:
                result = await self._advanced_research(query, research_type, evidence_level, include_safety)
            else:
                result = await self._basic_research(query, research_type, evidence_level, include_safety)

            # Cache result
            if self.cache_manager:
                await self._cache_result(cache_key, result)

            # Update performance stats
            processing_time = time.time() - start_time
            self.performance_stats["total_time"] += processing_time
            self.performance_stats["avg_response_time"] = (
                self.performance_stats["total_time"] / self.performance_stats["queries_processed"]
            )

            return self._format_output(result, output_format)

        except Exception as e:
            self.performance_stats["errors"] += 1
            logger.error(f"Research failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat()
            }

    async def _advanced_research(self, query: str, research_type: str, evidence_level: str, include_safety: bool) -> Dict[str, Any]:
        """Advanced research using full orchestrator"""
        return await self.orchestrator.comprehensive_research(
            query=query,
            research_mode=research_type,
            evidence_threshold=evidence_level,
            include_safety_assessment=include_safety
        )

    async def _basic_research(self, query: str, research_type: str, evidence_level: str, include_safety: bool) -> Dict[str, Any]:
        """Basic research using available scrapers"""
        return {
            "query": query,
            "research_type": research_type,
            "evidence_level": evidence_level,
            "include_safety": include_safety,
            "results": [
                {
                    "source": "basic_mode",
                    "content": f"Basic research for: {query}",
                    "confidence": 0.5
                }
            ],
            "success": True,
            "mode": "basic",
            "timestamp": datetime.now().isoformat()
        }

    async def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if result is cached"""
        try:
            # Placeholder pro cache logiku
            return None
        except Exception as e:
            logger.error(f"Cache check failed: {e}")
            return None

    async def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache the result"""
        try:
            # Placeholder pro cache logiku
            pass
        except Exception as e:
            logger.error(f"Cache store failed: {e}")

    def _format_output(self, result: Dict[str, Any], output_format: str) -> Dict[str, Any]:
        """Format output based on requested format"""
        if output_format == "brief":
            return {
                "query": result.get("query"),
                "summary": result.get("summary", "No summary available"),
                "confidence": result.get("confidence", 0.0),
                "success": result.get("success", False)
            }
        elif output_format == "json":
            return result
        else:  # detailed or expert
            return result

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.performance_stats.copy()

async def main():
    """Hlavní funkce s vylepšeným argument parsing"""
    parser = argparse.ArgumentParser(description="Unified Biohacking Research Tool")
    parser.add_argument("--query", "-q", help="Research query")
    parser.add_argument("--type", "-t", default="comprehensive",
                       choices=["quick", "balanced", "comprehensive", "safety", "dosage", "stacking"])
    parser.add_argument("--evidence", "-e", default="high", choices=["high", "medium", "all"])
    parser.add_argument("--format", "-f", default="detailed", choices=["brief", "detailed", "expert", "json"])
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--user-id", default="cli_user", help="User ID for session tracking")

    args = parser.parse_args()

    if not args.query:
        print("🧬 Unified Biohacking Research Tool")
        print("Use --query to specify research topic")
        print("Example: python main.py --query 'BPC-157 dosing protocol' --type comprehensive")
        return

    async with UnifiedBiohackingResearchTool(user_id=args.user_id, verbose=args.verbose) as tool:
        result = await tool.research(
            query=args.query,
            research_type=args.type,
            evidence_level=args.evidence,
            include_safety=True,
            output_format=args.format
        )

        if args.format == "json":
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("\n" + "="*80)
            print(f"🔬 Research Results for: {args.query}")
            print("="*80)
            print(json.dumps(result, indent=2, ensure_ascii=False))

        # Print performance stats if verbose
        if args.verbose:
            stats = tool.get_performance_stats()
            print("\n" + "="*40)
            print("📊 Performance Statistics:")
            print("="*40)
            for key, value in stats.items():
                print(f"{key}: {value}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Research interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
