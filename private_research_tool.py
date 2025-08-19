M1 Private Research Tool - Finální optimalizovaná verze pro soukromé použití
Maximálně jednoduchý a efektivní nástroj pro MacBook Air M1 s Phi-3 Mini
"""

import asyncio
import argparse
import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Zjednodušené importy pro soukromé použití
try:
    from local_ai_adapter import quick_ai_query, M1OptimizedOllamaClient
    from cache_manager import get_cache_manager
    from academic_scraper import create_scraping_orchestrator
    LOCAL_IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    LOCAL_IMPORTS_OK = False

class PrivateResearchTool:
    """Jednoduchý research tool optimalizovaný pro soukromé použití na M1"""

    def __init__(self):
        self.cache = get_cache_manager() if LOCAL_IMPORTS_OK else None
        print("🔒 Private Research Tool inicializován (100% lokální)")

    async def simple_research(self, query: str, use_ai: bool = True) -> Dict:
        """Jednoduchý research s minimálními závislostmi"""
        print(f"🔍 Výzkum: '{query}'")
        start_time = time.time()

        results = []
        sources_used = []

        # Pokus o scraping (bez složitých konfigurací)
        try:
            scraper = await create_scraping_orchestrator(max_concurrent=2, timeout=30)

            # Jen nejzákladnější zdroje
            basic_sources = ["wikipedia", "openalex"]

            for source in basic_sources:
                try:
                    source_results = await scraper.scrape_source(
                        source=source,
                        query=query,
                        max_results=3
                    )
                    if source_results:
                        results.extend(source_results)
                        sources_used.append(source)
                        print(f"✅ {source}: {len(source_results)} výsledků")
                except Exception as e:
                    print(f"⚠️ {source}: {e}")

            await scraper.cleanup()

        except Exception as e:
            print(f"⚠️ Scraping error: {e}")
            results = [{"title": "Scraping nedostupný", "summary": "Zkuste později"}]

        # AI analýza pokud je požadována a dostupná
        ai_summary = None
        if use_ai and LOCAL_IMPORTS_OK:
            try:
                print("🧠 Generuji AI souhrn...")

                # Připrav jednoduchý kontext
                context = ""
                for i, result in enumerate(results[:3], 1):
                    if isinstance(result, dict):
                        title = result.get('title', 'Bez názvu')[:100]
                        summary = result.get('summary', result.get('abstract', ''))[:200]
                        context += f"{i}. {title}: {summary}\n"

                if context:
                    prompt = f"""Na základě těchto informací o "{query}" vytvoř stručný souhrn:

{context}

Souhrn (max 100 slov):"""

                    ai_summary = await quick_ai_query(prompt, max_tokens=150)
                    print("✅ AI souhrn vygenerován")
                else:
                    ai_summary = "Nedostatek dat pro AI analýzu"

            except Exception as e:
                print(f"⚠️ AI error: {e}")
                ai_summary = f"AI nedostupná: {e}"

        execution_time = time.time() - start_time

        result = {
            'query': query,
            'results': results,
            'ai_summary': ai_summary,
            'sources_used': sources_used,
            'execution_time': execution_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        print(f"✅ Dokončeno za {execution_time:.1f}s")
        return result

    async def test_system(self) -> bool:
        """Test celého systému"""
        print("🧪 Testuji systém...")

        # Test AI
        try:
            response = await quick_ai_query("Test connection", max_tokens=20)
            if response and "❌" not in response:
                print("✅ Phi-3 Mini funguje")
                ai_ok = True
            else:
                print("❌ Phi-3 Mini nefunguje")
                ai_ok = False
        except Exception as e:
            print(f"❌ AI test failed: {e}")
            ai_ok = False

        # Test cache
        try:
            if self.cache:
                self.cache.set("test", {"working": True})
                cached = self.cache.get("test")
                if cached and cached.get("working"):
                    print("✅ Cache funguje")
                    cache_ok = True
                else:
                    print("❌ Cache nefunguje")
                    cache_ok = False
            else:
                print("⚠️ Cache nedostupná")
                cache_ok = False
        except Exception as e:
            print(f"❌ Cache test failed: {e}")
            cache_ok = False

        # Test memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            print(f"💾 Dostupná RAM: {available_gb:.1f}GB")
            memory_ok = available_gb > 1.0
        except Exception as e:
            print(f"⚠️ Memory check failed: {e}")
            memory_ok = True  # Assume OK if can't check

        return ai_ok and cache_ok and memory_ok

def show_help():
    """Zobrazí nápovědu"""
    print("""
🆘 M1 Private Research Tool - Nápověda

Základní použití:
  python private_research_tool.py "machine learning"
  python private_research_tool.py "AI ethics" --no-ai
  python private_research_tool.py --test
  python private_research_tool.py --interactive

Příklady dotazů:
  • "quantum computing applications"
  • "COVID-19 vaccines efficacy"  
  • "sustainable energy technologies"
  • "machine learning in healthcare"

Tipy pro M1:
  • Krátké dotazy jsou rychlejší
  • Vypněte ostatní aplikace pro více RAM
  • Cache se automaticky optimalizuje
  • AI odpovědi jsou 100% lokální
    """)

async def interactive_mode():
    """Interaktivní mód"""
    tool = PrivateResearchTool()
    print("🎯 Interaktivní mód - zadej 'exit' pro ukončení")

    while True:
        try:
            query = input("\n🔍 Research dotaz: ").strip()

            if query.lower() in ['exit', 'quit', 'q']:
                break
            elif query.lower() in ['help', 'h']:
                show_help()
                continue
            elif query.lower() == 'test':
                await tool.test_system()
                continue
            elif not query:
                continue

            result = await tool.simple_research(query)

            # Zobraz výsledky
            print(f"\n📚 Nalezeno: {len(result['results'])} výsledků")
            print(f"📊 Zdroje: {', '.join(result['sources_used'])}")

            if result['ai_summary']:
                print(f"\n🤖 AI Souhrn:\n{result['ai_summary']}")

            print(f"\n📖 Top výsledky:")
            for i, res in enumerate(result['results'][:3], 1):
                if isinstance(res, dict):
                    title = res.get('title', 'Bez názvu')[:60]
                    print(f"   {i}. {title}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    print("👋 Končím...")

async def main():
    parser = argparse.ArgumentParser(description="M1 Private Research Tool")
    parser.add_argument("query", nargs="?", help="Research dotaz")
    parser.add_argument("--no-ai", action="store_true", help="Bez AI analýzy")
    parser.add_argument("--test", action="store_true", help="Test systému")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interaktivní mód")
    parser.add_argument("--output", help="Soubor pro uložení výsledků")

    args = parser.parse_args()

    if not LOCAL_IMPORTS_OK:
        print("❌ Kritická chyba při importu modulů")
        return 1

    tool = PrivateResearchTool()

    if args.test:
        success = await tool.test_system()
        return 0 if success else 1

    if args.interactive:
        await interactive_mode()
        return 0

    if args.query:
        result = await tool.simple_research(args.query, use_ai=not args.no_ai)

        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"💾 Uloženo do: {output_path}")

        return 0
    else:
        show_help()
        return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Přerušeno")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Kritická chyba: {e}")
        sys.exit(1)
