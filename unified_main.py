Unified Main Entry Point - Hlavní vstupní bod pro celý Academic Research Tool
Sjednocuje všechny komponenty a poskytuje jednotné CLI rozhraní
"""

import asyncio
import logging
import sys
import argparse
import time
from typing import Optional, List
import json

# Import optimized components
try:
    from unified_config import get_config, create_config, Environment
    from academic_scraper import create_scraping_orchestrator
    UNIFIED_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importing unified components: {e}")
    UNIFIED_AVAILABLE = False
    # Define fallbacks for missing components
    get_config = None
    create_config = None
    Environment = None
    create_scraping_orchestrator = None
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedResearchTool:
    """Sjednocený nástroj pro akademický výzkum"""

    def __init__(self, environment: Optional[Environment] = None):
        """Initialize tool with configuration"""
        try:
            self.config = get_config() if environment is None else create_config(environment)
            logger.info(f"✅ Unified Research Tool initialized in {self.config.environment.value} mode")

            # Validate configuration
            errors = self.config.validate()
            if errors:
                logger.warning(f"Configuration validation warnings: {errors}")

        except Exception as e:
            logger.error(f"Failed to initialize configuration: {e}")
            raise

    async def run_scraping(self, query: str, sources: Optional[List[str]] = None,
                          output_file: Optional[str] = None) -> dict:
        """Run scraping operation from CLI"""
        logger.info(f"Starting scraping operation for query: '{query}'")

        try:
            if create_scraping_orchestrator is None:
                raise ImportError("Academic scraper not available")

            orchestrator = create_scraping_orchestrator()

            # Run scraping
            results = await orchestrator.scrape_all_sources(query, sources)

            # Process results
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]

            output_data = {
                'query': query,
                'timestamp': __import__('datetime').datetime.utcnow().isoformat(),
                'summary': {
                    'total_sources': len(results),
                    'successful_sources': len(successful_results),
                    'failed_sources': len(failed_results)
                },
                'results': [
                    {
                        'source': r.source,
                        'success': r.success,
                        'data': r.data,
                        'error': r.error,
                        'response_time': r.response_time
                    }
                    for r in results
                ]
            }

            # Save to file if requested
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Results saved to: {output_file}")

            # Print summary
            print(f"\n📊 Scraping Summary:")
            print(f"Query: {query}")
            print(f"Successful sources: {len(successful_results)}")
            print(f"Failed sources: {len(failed_results)}")

            for result in successful_results:
                print(f"✅ {result.source}: {len(str(result.data))} characters")

            for result in failed_results:
                print(f"❌ {result.source}: {result.error}")

            # Cleanup
            await orchestrator.cleanup()

            return output_data

        except Exception as e:
            logger.error(f"Scraping operation failed: {e}")
            raise

    def run_server(self, server_type: str = "unified"):
        """Run web server"""
        logger.info(f"Starting {server_type} server...")

        if server_type == "unified":
            # Run FastAPI unified server via uvicorn
            try:
                from unified_server import create_app
                import uvicorn
                app = create_app()
                uvicorn.run(
                    app,
                    host=self.config.api.host,
                    port=self.config.api.port,
                    reload=self.config.api.debug
                )
            except ImportError:
                logger.error("unified_server module not found. Please ensure it's properly configured.")
                raise
        else:
            raise ValueError(f"Unsupported server type: {server_type}. Only 'unified' is supported.")

    def show_config(self):
        """Display current configuration"""
        print(f"\n🔧 Current Configuration:")
        print(f"Environment: {self.config.environment.value}")
        print(f"Database: {self.config.database.type}")
        print(f"API Host: {self.config.api.host}:{self.config.api.port}")
        print(f"Debug Mode: {self.config.api.debug}")
        print(f"CORS Enabled: {self.config.api.cors_enabled}")
        print(f"Cache Enabled: {self.config.cache.enabled}")

        print(f"\n📡 Available Sources:")
        for name, source in self.config.sources.items():
            status = "✅ Enabled" if source.enabled else "❌ Disabled"
            print(f"  {name}: {status} ({source.base_url})")

    def run_tests(self):
        """Spustí test suite"""
        print("\n🧪 Running test suite...")
        import subprocess
        try:
            result = subprocess.run(['python', '-m', 'pytest', '-v'],
                                  capture_output=True, text=True, cwd='.')
            print(result.stdout)
            if result.stderr:
                print(f"Errors: {result.stderr}")
            return result.returncode == 0
        except Exception as e:
            print(f"Failed to run tests: {e}")
            return False

    def show_status(self):
        """Zobrazí status systému s rozšířenými informacemi"""
        print(f"\n📊 System Status:")
        print(f"Environment: {self.config.environment.value}")
        print(f"Cache enabled: {self.config.cache.enabled}")

        # Test AI connectivity
        if self.config.ai.use_local_ai:
            try:
                import requests
                response = requests.get(f"{self.config.ai.local_ai.ollama_host}/api/tags", timeout=5)
                if response.status_code == 200:
                    print("✅ Local AI (Ollama): Connected")
                    models = response.json().get('models', [])
                    print(f"   Available models: {len(models)}")
                else:
                    print("❌ Local AI (Ollama): Not responding")
            except Exception:
                print("❌ Local AI (Ollama): Not available")

        # Test sources
        enabled_sources = [name for name, source in self.config.sources.items() if source.enabled]
        print(f"Enabled sources: {len(enabled_sources)}")

        # Enhanced status with new systems
        try:
            from smart_caching_system import get_intelligent_cache
            cache = get_intelligent_cache()
            cache_stats = cache.get_cache_stats()
            print(f"\n🧠 Intelligent Cache Status:")
            print(f"   Cache hit rate: {cache_stats['cache_hit_rate']:.2%}")
            print(f"   Total entries: {cache_stats['total_entries']}")
            print(f"   Cache size: {cache_stats['total_size_mb']:.1f}MB")
            print(f"   Predictive hits: {cache_stats['predictive_hits']}")
        except ImportError:
            print("⚠️ Smart caching not available")

        try:
            from adaptive_learning_system import get_personalization_engine
            engine = get_personalization_engine()
            learning_stats = engine.get_learning_stats()
            print(f"\n🎯 Personalization Status:")
            print(f"   Learned preferences: {learning_stats['total_preferences']}")
            print(f"   High confidence preferences: {learning_stats['high_confidence_preferences']}")
            if learning_stats['most_researched_peptide']:
                print(f"   Top peptide interest: {learning_stats['most_researched_peptide'][0]}")
        except ImportError:
            print("⚠️ Adaptive learning not available")

    async def run_advanced_research(self, query: str, priority: str = "balanced",
                                  enable_quality_check: bool = True,
                                  enable_personalization: bool = True) -> dict:
        """Advanced research s všemi novými funkcemi"""
        logger.info(f"Starting advanced research for: '{query}'")
        start_time = time.time()

        try:
            # 1. Intelligent source selection a research orchestration
            from intelligent_research_orchestrator import IntelligentResearchOrchestrator
            orchestrator = IntelligentResearchOrchestrator()

            priority_map = {
                "speed": "SPEED",
                "balanced": "ACCURACY",
                "quality": "DEPTH"
            }

            research_priority = getattr(__import__('intelligent_research_orchestrator').ResearchPriority,
                                      priority_map.get(priority, "ACCURACY"))

            # 2. Conduct intelligent research
            research_results = await orchestrator.conduct_intelligent_research(
                query, priority=research_priority, max_sources=6
            )

            # 3. Quality assessment
            if enable_quality_check:
                try:
                    from quality_assessment_system import assess_research_quality
                    quality_report = await assess_research_quality(research_results)
                    research_results['quality_assessment'] = quality_report['quality_assessment']
                    logger.info("✅ Quality assessment completed")
                except ImportError:
                    logger.warning("Quality assessment not available")

            # 4. Personalization learning
            if enable_personalization:
                try:
                    from adaptive_learning_system import learn_from_research_session
                    sources_used = research_results.get('sources_used', [])
                    time_spent = time.time() - start_time
                    await learn_from_research_session(query, sources_used, time_spent)
                    logger.info("📚 Learning from research session completed")
                except ImportError:
                    logger.warning("Personalization learning not available")

            # 5. Performance metrics
            total_time = time.time() - start_time
            research_results.update({
                'advanced_features': {
                    'intelligent_orchestration': True,
                    'quality_assessment': enable_quality_check,
                    'personalization_learning': enable_personalization,
                    'total_processing_time': total_time
                }
            })

            print(f"\n🎯 Advanced Research Summary:")
            print(f"Query: {query}")
            print(f"Processing time: {total_time:.2f}s")
            print(f"Sources analyzed: {len(research_results.get('sources_used', []))}")

            if 'quality_assessment' in research_results:
                qa = research_results['quality_assessment']['overall_metrics']
                print(f"Quality: {qa['high_quality_sources']}/{qa['total_sources']} high-quality sources")

            return research_results

        except Exception as e:
            logger.error(f"Advanced research failed: {e}")
            # Fallback to basic scraping
            return await self.run_scraping(query, None, None)

    async def run_peptide_research(self, peptide_name: str, research_type: str = "comprehensive") -> dict:
        """Specializovaný peptide research s AI optimalizacemi"""
        research_queries = {
            "comprehensive": f"comprehensive analysis of {peptide_name} including mechanism, dosage, safety, clinical evidence",
            "dosage": f"{peptide_name} dosage protocol administration timing cycling",
            "safety": f"{peptide_name} side effects contraindications safety profile risks",
            "mechanism": f"{peptide_name} mechanism of action pathways receptors pharmacokinetics",
            "clinical": f"{peptide_name} clinical trials research studies evidence efficacy"
        }

        query = research_queries.get(research_type, research_queries["comprehensive"])

        print(f"\n🧬 Peptide Research: {peptide_name}")
        print(f"Focus: {research_type}")

        return await self.run_advanced_research(
            query,
            priority="quality",  # Always use high quality for peptide research
            enable_quality_check=True,
            enable_personalization=True
        )

async def main():
    """Hlavní async CLI funkce"""
    parser = argparse.ArgumentParser(description='Unified Academic Research Tool')
    parser.add_argument('--env', choices=['development', 'testing', 'production'],
                       default='development', help='Environment to run in')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Scraping command
    scrape_parser = subparsers.add_parser('scrape', help='Run scraping operation')
    scrape_parser.add_argument('query', help='Search query')
    scrape_parser.add_argument('--sources', nargs='+', help='Specific sources to use')
    scrape_parser.add_argument('--output', '-o', help='Output file for results')

    # Server command
    server_parser = subparsers.add_parser('server', help='Run web server')
    server_parser.add_argument('--type', choices=['unified', 'flask'],
                              default='unified', help='Server type')

    # Config commands
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('action', choices=['show', 'validate'],
                              help='Configuration action')

    # Test command
    test_parser = subparsers.add_parser('test', help='Run test suite')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')

    # Advanced research command
    research_parser = subparsers.add_parser('research', help='Advanced AI-powered research')
    research_parser.add_argument('query', help='Research query')
    research_parser.add_argument('--priority', choices=['speed', 'balanced', 'quality'],
                                default='balanced', help='Research priority')
    research_parser.add_argument('--output', '-o', help='Output file for results')
    research_parser.add_argument('--no-quality-check', action='store_true',
                                help='Disable quality assessment')
    research_parser.add_argument('--no-learning', action='store_true',
                                help='Disable personalization learning')

    # Peptide research command
    peptide_parser = subparsers.add_parser('peptide', help='Specialized peptide research')
    peptide_parser.add_argument('name', help='Peptide name (e.g., BPC-157, TB-500)')
    peptide_parser.add_argument('--type', choices=['comprehensive', 'dosage', 'safety', 'mechanism', 'clinical'],
                               default='comprehensive', help='Research focus')
    peptide_parser.add_argument('--output', '-o', help='Output file for results')

    # Cache management
    cache_parser = subparsers.add_parser('cache', help='Cache management')
    cache_parser.add_argument('action', choices=['stats', 'clean', 'preload'],
                             help='Cache action')
    cache_parser.add_argument('--query', help='Query for preload action')

    # Learning insights
    insights_parser = subparsers.add_parser('insights', help='Show personalization insights')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize tool
    try:
        environment = Environment(args.env) if hasattr(args, 'env') else Environment.DEVELOPMENT
        tool = UnifiedResearchTool(environment)

        # Execute command
        if args.command == 'scrape':
            await tool.run_scraping(args.query, args.sources, args.output)

        elif args.command == 'research':
            result = await tool.run_advanced_research(
                args.query,
                priority=args.priority,
                enable_quality_check=not args.no_quality_check,
                enable_personalization=not args.no_learning
            )
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"📄 Advanced research results saved to: {args.output}")

        elif args.command == 'peptide':
            result = await tool.run_peptide_research(args.name, args.type)
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"🧬 Peptide research results saved to: {args.output}")

        elif args.command == 'cache':
            if args.action == 'stats':
                try:
                    from smart_caching_system import get_intelligent_cache
                    cache = get_intelligent_cache()
                    stats = cache.get_cache_stats()
                    print(f"\n📊 Cache Statistics:")
                    print(f"Hit rate: {stats['cache_hit_rate']:.2%}")
                    print(f"Total entries: {stats['total_entries']}")
                    print(f"Cache size: {stats['total_size_mb']:.1f}MB")
                    print(f"Predictive hits: {stats['predictive_hits']}")
                    print(f"Hot cache entries: {stats['hot_cache_size']}")
                except ImportError:
                    print("❌ Smart caching not available")

            elif args.action == 'clean':
                try:
                    from smart_caching_system import get_intelligent_cache
                    cache = get_intelligent_cache()
                    await cache.cleanup_old_cache()
                    print("✅ Cache cleanup completed")
                except ImportError:
                    print("❌ Smart caching not available")

            elif args.action == 'preload':
                if not args.query:
                    print("❌ --query required for preload action")
                    return
                try:
                    from smart_caching_system import get_intelligent_cache
                    cache = get_intelligent_cache()
                    await cache.predictive_preload(args.query)
                    print(f"🔮 Predictive preload initiated for: {args.query}")
                except ImportError:
                    print("❌ Smart caching not available")

        elif args.command == 'insights':
            try:
                from adaptive_learning_system import get_personalization_engine
                engine = get_personalization_engine()
                insights = await engine.generate_insights()
                print(f"\n🎯 Personalization Insights:")
                print(insights.get('insights', 'No insights available'))

                stats = engine.get_learning_stats()
                print(f"\n📊 Learning Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            except ImportError:
                print("❌ Adaptive learning not available")
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)

def cli_main():
    """Synchronní wrapper pro CLI"""
    if not UNIFIED_AVAILABLE:
        print("❌ Unified components not available. Please install dependencies.")
        sys.exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cli_main()
