#!/usr/bin/env python3
"""
Unified Main Entry Point - Hlavní vstupní bod pro celý Academic Research Tool
Sjednocuje všechny komponenty a poskytuje jednotné CLI rozhraní
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path
from typing import Optional, List
import json

# Import optimized components
try:
    from unified_config import get_config, create_config, Environment
    from unified_server import create_unified_server
    from academic_scraper import create_scraping_orchestrator
    UNIFIED_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importing unified components: {e}")
    UNIFIED_AVAILABLE = False
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
            server = create_unified_server()
            server.run()
        elif server_type == "flask":
            from app import create_app
            app = create_app()
            app.run(
                host=self.config.api.host,
                port=self.config.api.port,
                debug=self.config.api.debug
            )
        else:
            raise ValueError(f"Unknown server type: {server_type}")

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

    def run_diagnostics(self):
        """Run system diagnostics"""
        print(f"\n🔍 System Diagnostics:")

        # Check configuration
        errors = self.config.validate()
        if errors:
            print("❌ Configuration Issues:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("✅ Configuration: OK")

        # Check dependencies
        try:
            import requests, aiohttp, beautifulsoup4
            print("✅ Core Dependencies: OK")
        except ImportError as e:
            print(f"❌ Missing Dependencies: {e}")

        # Check API keys
        import os
        api_keys = {}
        for source_name, source_config in self.config.sources.items():
            if source_config.api_key_env:
                key_present = bool(os.getenv(source_config.api_key_env))
                api_keys[source_name] = key_present

        if api_keys:
            print("🔑 API Keys:")
            for source, present in api_keys.items():
                status = "✅ Present" if present else "❌ Missing"
                print(f"  {source}: {status}")

        # Test basic scraping
        print("\n🧪 Testing Basic Scraping:")
        asyncio.run(self._test_scraping())

    async def _test_scraping(self):
        """Test basic scraping functionality"""
        try:
            orchestrator = create_scraping_orchestrator()

            # Quick test with Wikipedia only
            results = await orchestrator.scrape_all_sources("test", ["wikipedia"])

            if results and results[0].success:
                print("✅ Basic Scraping: OK")
            else:
                print("❌ Basic Scraping: Failed")
                if results:
                    print(f"  Error: {results[0].error}")

            await orchestrator.cleanup()

        except Exception as e:
            print(f"❌ Basic Scraping: Error - {e}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Unified Academic Research Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_main.py scrape "machine learning" --sources wikipedia openalex
  python unified_main.py server --type unified
  python unified_main.py config
  python unified_main.py diagnostics
        """
    )

    # Global options
    parser.add_argument('--env', choices=['development', 'testing', 'production'],
                       help='Environment to use')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Scrape command
    scrape_parser = subparsers.add_parser('scrape', help='Run scraping operation')
    scrape_parser.add_argument('query', help='Search query')
    scrape_parser.add_argument('--sources', nargs='+',
                              choices=['wikipedia', 'openalex', 'semantic_scholar', 'pubmed'],
                              help='Sources to scrape')
    scrape_parser.add_argument('--output', '-o', help='Output file path')

    # Server command
    server_parser = subparsers.add_parser('server', help='Start web server')
    server_parser.add_argument('--type', choices=['unified', 'flask'],
                              default='unified', help='Server type')

    # Config command
    subparsers.add_parser('config', help='Show current configuration')

    # Diagnostics command
    subparsers.add_parser('diagnostics', help='Run system diagnostics')

    # Parse arguments
    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine environment
    environment = None
    if args.env:
        environment = Environment(args.env)

    try:
        # Initialize tool
        tool = UnifiedResearchTool(environment)

        # Execute command
        if args.command == 'scrape':
            asyncio.run(tool.run_scraping(args.query, args.sources, args.output))

        elif args.command == 'server':
            tool.run_server(args.type)

        elif args.command == 'config':
            tool.show_config()

        elif args.command == 'diagnostics':
            tool.run_diagnostics()

        else:
            parser.print_help()

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
