"""
Optimalizovaný main.py pro soukromé použití
Rychlé spuštění, minimální overhead
"""
import asyncio
import os
from pathlib import Path
from config_personal import *
from high_performance_cache import high_perf_cache

def print_banner():
    """Zkrácený banner"""
    print("🔬 AI Research Tool - Personal Edition")
    print("Optimalizováno pro rychlost a soukromé použití\n")

async def run_fast_cli_mode():
    """Ultraúsporný CLI režim - levnější než Perplexity"""
    print("💰 Spouštím v úsporném režimu (cíl: <$15/měsíc)...\n")

    # Lazy import pro rychlejší start
    from research_engine import UltraCheapResearchEngine
    from cost_optimizer import cost_optimizer

    # Úsporná inicializace
    engine = UltraCheapResearchEngine()

    # Zobraz aktuální rozpočet
    if hasattr(engine.gemini_manager, 'get_daily_cost_summary'):
        cost_summary = engine.gemini_manager.get_daily_cost_summary()
        print(f"📊 Denní rozpočet: ${cost_summary['remaining_budget']:.2f} zbývá z ${DAILY_COST_LIMIT}")
        print(f"🔥 API volání: {cost_summary['calls_remaining']} zbývá z 50")

    # Test dotaz s optimalizací nákladů
    from research_engine import ResearchQuery
    original_query = "artificial intelligence trends 2024"
    optimized_query = cost_optimizer.optimize_query_for_cost(original_query)

    query = ResearchQuery(
        query=optimized_query,
        sources=["academic"],
        depth="medium",
        max_results=15  # Sníženo pro úsporu
    )

    print(f"🔍 Úsporný research: '{query.query}'")
    print("⏳ Preferuji cache před API voláními...\n")

    try:
        # Úsporné zpracování
        result = await engine.ultra_cheap_research(query)

        print(f"✅ Hotovo za {result.processing_time:.1f}s")
        print(f"📊 Nalezeno: {result.sources_found} zdrojů")
        print(f"💰 Náklady: ${result.cost_info.get('estimated_cost_usd', 0):.4f}")

        # Zobraz úspory oproti konkurenci
        savings = engine.get_savings_report()
        print(f"💵 Měsíční úspora vs Perplexity: ${savings['monthly_savings']:.2f} ({savings['savings_percentage']:.1f}%)")

        print(f"\n📄 Shrnutí:\n{result.summary[:300]}...")

        if result.key_findings:
            print(f"\n🔑 Klíčová slova: {', '.join(result.key_findings[:5])}")

    except Exception as e:
        print(f"❌ Chyba: {e}")

def run_fast_web_mode():
    """Rychlé webové rozhraní"""
    print("🌐 Spouštím optimalizované webové rozhraní...")

    import subprocess
    import sys

    # Optimalizované Streamlit nastavení
    cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.maxUploadSize", "200",  # Menší pro rychlost
        "--server.maxMessageSize", "200",
        "--global.disableWatchdogWarning", "true",
        "--browser.gatherUsageStats", "false"
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Ukončeno")

def setup_fast_mode():
    """Rychlé nastavení pro personal use"""
    print("⚡ Konfiguruji pro maximální rychlost...")

    # Vytvoř adresáře
    for directory in [DATA_DIR, CACHE_DIR, REPORTS_DIR]:
        directory.mkdir(exist_ok=True)

    # Inicializuj cache
    high_perf_cache._cleanup_disk_cache()

    print("✅ Rychlá konfigurace dokončena!")

def main():
    """Rychlá hlavní funkce"""
    print_banner()

    # Rychlá kontrola API klíče
    if not GEMINI_API_KEY:
        print("⚠️  GEMINI_API_KEY není nastaven!")
        print("Nastavte v .env souboru nebo environment variable")
        return

    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--web":
            run_fast_web_mode()
        elif sys.argv[1] == "--setup":
            setup_fast_mode()
        elif sys.argv[1] == "--cli":
            asyncio.run(run_fast_cli_mode())
    else:
        # Rychlá volba
        print("Vyberte režim:")
        print("1. 🌐 Web (doporučeno)")
        print("2. 💻 CLI")
        print("3. ⚡ Setup")

        try:
            choice = input("Volba (1-3): ").strip()

            if choice == "1":
                run_fast_web_mode()
            elif choice == "2":
                asyncio.run(run_fast_cli_mode())
            elif choice == "3":
                setup_fast_mode()
            else:
                print("❌ Neplatná volba")

        except KeyboardInterrupt:
            print("\n👋 Ukončeno")

if __name__ == '__main__':
    main()
