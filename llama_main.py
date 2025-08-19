Llama 3.1 8B Main - Pokročilý research tool optimalizovaný pro MacBook Air M1
Využívá Llama 3.1 8B pro vysokou kvalitu výzkumu s inteligentním fallbackem
"""

import asyncio
import argparse
import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Llama 3.1 8B optimized imports
try:
    from unified_config import get_config
    from unified_research_engine import M1OptimizedResearchEngine, M1OptimizedResearchRequest
    from local_ai_adapter import M1OptimizedLlamaClient, quick_ai_query
    from cache_manager import get_cache_manager
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Chyba při importu: {e}")
    IMPORTS_OK = False
    sys.exit(1)

class LlamaResearchTool:
    """Pokročilý research tool s Llama 3.1 8B a inteligentním model managementem"""

    def __init__(self):
        self.config = get_config()
        print(f"🧠 Llama 3.1 8B Research Tool inicializován v {self.config.environment.value} módu")

        # Ověř konfigurace
        errors = self.config.validate()
        if errors:
            print("⚠️  Varování konfigurace:")
            for error in errors:
                print(f"   • {error}")

    async def research_query(
        self,
        query: str,
        strategy: str = "balanced",
        sources: Optional[list] = None,
        require_quality: bool = False,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Provede pokročilý research s Llama 3.1 8B"""

        print(f"🔍 Zahajuji high-quality research: '{query}'")
        print(f"📊 Strategie: {strategy}")
        print(f"🎯 High-quality mode: {'Ano' if require_quality else 'Auto-detect'}")

        start_time = time.time()

        # Automatická detekce kvality podle dotazu
        auto_quality = self._should_use_quality_mode(query, require_quality)

        # Vytvoř research request s enhanced parametry
        request = M1OptimizedResearchRequest(
            query=query,
            strategy=strategy,
            sources=sources,
            use_ai_analysis=True,
            max_sources=3 if strategy == "fast" else 4,
            timeout=45 if auto_quality else 30
        )

        # Spusť research engine s Llama optimalizacemi
        async with M1OptimizedResearchEngine() as engine:
            result = await engine.research(request)

        execution_time = time.time() - start_time

        # Enhanced výsledky s model info
        await self._display_enhanced_results(result, execution_time, auto_quality)

        # Ulož do souboru pokud požadováno
        if output_file:
            await self._save_enhanced_results(result, output_file, auto_quality)
            print(f"💾 Výsledky uloženy do: {output_file}")

        return result.to_dict()

    def _should_use_quality_mode(self, query: str, force_quality: bool) -> bool:
        """Inteligentně rozhodne o použití quality módu"""
        if force_quality:
            return True

        # Automatická detekce podle komplexity dotazu
        quality_indicators = [
            len(query) > 100,
            any(word in query.lower() for word in [
                'analyze', 'compare', 'explain', 'research', 'detailed',
                'comprehensive', 'academic', 'scientific', 'technical',
                'trends', 'future', 'implications', 'impact'
            ]),
            '?' in query and len(query.split('?')) > 1,  # Více otázek
            query.count(',') > 2,  # Složitější struktura
        ]

        return sum(quality_indicators) >= 2

    async def _display_enhanced_results(self, result, execution_time: float, quality_mode: bool):
        """Zobrazí enhanced výsledky s model statistikami"""

        print(f"\n✅ Research dokončen za {execution_time:.2f}s")
        print(f"📚 Použité zdroje: {', '.join(result.sources_used)}")
        print(f"📝 Nalezeno výsledků: {len(result.results)}")
        print(f"💾 Cache hit: {'Ano' if result.cache_hit else 'Ne'}")
        print(f"🧠 Spotřeba paměti: {result.memory_usage_mb:.1f}MB")
        print(f"🎯 Quality mode: {'Aktivní' if quality_mode else 'Standard'}")

        if result.ai_analysis:
            print(f"\n🤖 AI Analýza (Llama 3.1 8B):")
            print("=" * 60)
            print(f"{result.ai_analysis}")
            print("=" * 60)

        # Zobraz top výsledky s enhanced formátováním
        print(f"\n📖 Nejlepší výsledky:")
        for i, res in enumerate(result.results[:5], 1):
            if isinstance(res, dict) and 'title' in res:
                title = res.get('title', 'Bez názvu')[:70]
                source = res.get('source', 'Neznámý')
                print(f"   {i}. [{source}] {title}")

                # Zobraz summary pokud existuje
                summary = res.get('summary', res.get('abstract', ''))
                if summary:
                    summary_short = summary[:150] + "..." if len(summary) > 150 else summary
                    print(f"      📄 {summary_short}")

    async def _save_enhanced_results(self, result, output_file: str, quality_mode: bool):
        """Uloží enhanced výsledky s metadaty"""
        output_path = Path(output_file)

        enhanced_data = {
            **result.to_dict(),
            'enhanced_metadata': {
                'quality_mode_used': quality_mode,
                'model_info': await self._get_model_info(),
                'optimization_level': 'llama_3_1_8b',
                'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'tool_version': '2.0.0'
            }
        }

        if output_path.suffix.lower() == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
        else:
            # Enhanced text format
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"🧠 LLAMA 3.1 8B RESEARCH REPORT\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"Query: {result.query}\n")
                f.write(f"Datum: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Strategie: {result.strategy_used}\n")
                f.write(f"Quality Mode: {'Aktivní' if quality_mode else 'Standard'}\n")
                f.write(f"Zdroje: {', '.join(result.sources_used)}\n")
                f.write(f"Doba zpracování: {result.execution_time:.2f}s\n")
                f.write(f"Spotřeba paměti: {result.memory_usage_mb:.1f}MB\n\n")

                if result.ai_analysis:
                    f.write(f"🤖 AI ANALÝZA (Llama 3.1 8B):\n")
                    f.write(f"-" * 40 + "\n")
                    f.write(f"{result.ai_analysis}\n\n")

                f.write("📚 DETAILNÍ VÝSLEDKY:\n")
                f.write(f"-" * 40 + "\n")
                for i, res in enumerate(result.results, 1):
                    if isinstance(res, dict):
                        f.write(f"{i}. {res.get('title', 'Bez názvu')}\n")
                        if 'source' in res:
                            f.write(f"   Zdroj: {res['source']}\n")
                        if 'url' in res:
                            f.write(f"   URL: {res['url']}\n")
                        if 'summary' in res:
                            f.write(f"   Souhrn: {res['summary']}\n")
                        f.write("\n")

    async def _get_model_info(self) -> Dict:
        """Získá informace o aktuálně používaných modelech"""
        try:
            async with M1OptimizedLlamaClient() as client:
                return client.get_model_stats()
        except Exception:
            return {"error": "Model info nedostupné"}

    async def test_llama_connection(self):
        """Otestuje připojení k Llama 3.1 8B a fallback modelům"""
        print("🧠 Testuji Llama 3.1 8B ecosystem...")

        tests = [
            ("Llama 3.1 8B (high-quality)", "Analyzuj současné trendy v AI", 100, True),
            ("Auto-select model", "Co je machine learning?", 50, False),
            ("Fallback test", "Rychlá odpověď", 30, False)
        ]

        results = []

        for test_name, query, max_tokens, require_quality in tests:
            print(f"\n🧪 Test: {test_name}")
            try:
                start_time = time.time()
                response = await quick_ai_query(
                    query,
                    max_tokens=max_tokens,
                    require_quality=require_quality
                )

                test_time = time.time() - start_time

                if response and "❌" not in response:
                    print(f"✅ {test_name}: {test_time:.2f}s")
                    print(f"   📝 Odpověď: {response[:100]}...")
                    results.append(True)
                else:
                    print(f"❌ {test_name}: Neplatná odpověď")
                    results.append(False)

            except Exception as e:
                print(f"❌ {test_name}: {e}")
                results.append(False)

        success_rate = sum(results) / len(results) * 100
        print(f"\n📊 Celková úspěšnost: {success_rate:.0f}%")

        if success_rate >= 66:
            print("✅ Llama 3.1 8B ecosystem je funkční!")
            return True
        else:
            print("⚠️ Některé testy selhaly. Zkontroluj konfiguraci.")
            return False

    def show_enhanced_system_info(self):
        """Zobrazí enhanced systémové informace včetně model stats"""
        print("🖥️  Enhanced Systémové Informace - Llama 3.1 8B Edition")
        print("=" * 60)

        try:
            import psutil

            # CPU info s M1 optimalizacemi
            cpu_count = psutil.cpu_count()
            cpu_usage = psutil.cpu_percent(interval=1)
            print(f"   🖥️  CPU: {cpu_count} jader, {cpu_usage}% využití")

            # Memory info s doporučeními pro Llama
            memory = psutil.virtual_memory()
            memory_total = memory.total / (1024**3)
            memory_available = memory.available / (1024**3)
            memory_percent = memory.percent

            print(f"   💾 RAM: {memory_available:.1f}GB volné z {memory_total:.1f}GB ({memory_percent}% využito)")

            # Doporučení podle RAM
            if memory_available >= 4.5:
                print(f"   ✅ Dostatek RAM pro Llama 3.1 8B")
            elif memory_available >= 3.0:
                print(f"   ⚠️  Omezená RAM - doporučuji menší modely")
            else:
                print(f"   ❌ Nízká RAM - použij pouze Phi-3 Mini")

            # Disk info
            disk = psutil.disk_usage('/')
            disk_free = disk.free / (1024**3)
            disk_total = disk.total / (1024**3)
            print(f"   💿 Disk: {disk_free:.1f}GB volné z {disk_total:.1f}GB")

        except Exception as e:
            print(f"   ❌ Nelze získat systémové informace: {e}")

        # Cache info
        try:
            cache_manager = get_cache_manager()
            cache_stats = cache_manager.get_stats()
            print(f"   💾 Cache: {cache_stats['size']} položek, {cache_stats['hit_rate']} hit rate")
            print(f"   📊 Cache memory: {cache_stats.get('memory_usage_mb', 0):.1f}MB")
        except Exception as e:
            print(f"   💾 Cache: Nedostupná ({e})")

        # Model info (async - zobrazíme placeholder)
        print(f"   🧠 AI Models: Llama 3.1 8B + fallbacks (spusť --test-ai pro detaily)")

async def main():
    """Hlavní funkce s enhanced CLI rozhraním"""
    parser = argparse.ArgumentParser(
        description="Llama 3.1 8B MacBook Research Tool - High-Quality AI Research"
    )

    parser.add_argument("query", nargs="?", help="Research dotaz")
    parser.add_argument("--strategy", choices=["fast", "balanced", "thorough"],
                       default="balanced", help="Strategie výzkumu")
    parser.add_argument("--sources", nargs="+",
                       choices=["wikipedia", "pubmed", "openalex"],
                       help="Konkrétní zdroje")
    parser.add_argument("--require-quality", action="store_true",
                       help="Vynutit high-quality mode (Llama 3.1 8B)")
    parser.add_argument("--output", help="Soubor pro uložení výsledků")
    parser.add_argument("--test-ai", action="store_true",
                       help="Otestovat Llama 3.1 8B ecosystem")
    parser.add_argument("--system-info", action="store_true",
                       help="Zobrazit enhanced systémové informace")
    parser.add_argument("--interactive", action="store_true",
                       help="Interaktivní mód s pokročilými funkcemi")

    args = parser.parse_args()

    if not IMPORTS_OK:
        print("❌ Nelze pokračovat kvůli chybám při importu")
        return 1

    # Vytvoř tool
    tool = LlamaResearchTool()

    # Test AI pokud požadováno
    if args.test_ai:
        success = await tool.test_llama_connection()
        return 0 if success else 1

    # Enhanced systémové info
    if args.system_info:
        tool.show_enhanced_system_info()
        return 0

    # Research dotaz
    if args.query:
        try:
            await tool.research_query(
                query=args.query,
                strategy=args.strategy,
                sources=args.sources,
                require_quality=args.require_quality,
                output_file=args.output
            )
            return 0
        except Exception as e:
            print(f"❌ Chyba při research: {e}")
            return 1
    elif args.interactive:
        # Enhanced interaktivní mód
        print("🎯 Llama 3.1 8B Research Tool - Enhanced Interaktivní Mód")
        print("💡 Příkazy: 'exit', 'help', 'stats', 'clear', 'models'")

        while True:
            try:
                query = input("\n🧠 Research dotaz (Llama 3.1 8B): ").strip()

                if query.lower() in ['exit', 'quit', 'q']:
                    print("👋 Ukončuji...")
                    break
                elif query.lower() in ['help', 'h']:
                    print("""
🆘 Enhanced Nápověda:
   • Zadej jakýkoli výzkumný dotaz pro high-quality analýzu
   • Příklady: "AI trendy 2024", "quantum computing applications"
   • Příkazy: 
     - 'exit' (ukončit)
     - 'stats' (systémové info)
     - 'clear' (vyčistit cache)
     - 'models' (model statistiky)
     - 'quality on/off' (zapnout/vypnout quality mode)
   • Pro komplexní dotazy je automaticky aktivován Llama 3.1 8B
                    """)
                    continue
                elif query.lower() == 'stats':
                    tool.show_enhanced_system_info()
                    continue
                elif query.lower() == 'models':
                    await tool.test_llama_connection()
                    continue
                elif query.lower() == 'clear':
                    cache_manager = get_cache_manager()
                    cache_manager.clear()
                    print("🗑️  Cache vyčištěna")
                    continue
                elif not query:
                    continue

                # Spusť enhanced research
                await tool.research_query(
                    query,
                    strategy="balanced",
                    require_quality=False  # Auto-detect
                )

            except KeyboardInterrupt:
                print("\n👋 Ukončuji...")
                break
            except Exception as e:
                print(f"❌ Chyba: {e}")

        return 0
    else:
        # Zobraz help a model info
        parser.print_help()
        print(f"\n🧠 Dostupné AI modely:")
        print(f"   • Llama 3.1 8B - High-quality research a analýzy")
        print(f"   • Qwen2 7B - Kódování a technické dotazy")
        print(f"   • Phi-3 Mini - Rychlé odpovědi a fallback")
        print(f"\n💡 Quick start: python {sys.argv[0]} 'your research query'")
        return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Ukončeno uživatelem")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Kritická chyba: {e}")
        sys.exit(1)
