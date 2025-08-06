"""
AI Research Tool - Hlavní spouštěcí soubor
Komplexní nástroj pro hloubkové výzkumy na internetu, v databázích a akademických zdrojích
"""
import asyncio
import sys
import argparse
from pathlib import Path
import websocket
import json

# Import našich modulů
from research_engine import ResearchEngine, ResearchQuery
from database_manager import DatabaseManager
from streamlit_app import ResearchUI
import config

def print_banner():
    """Výpis úvodního banneru"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    🔬 AI Research Tool 🔬                     ║
    ║                                                               ║
    ║  Komplexní nástroj pro hloubkové výzkumy a analýzy           ║
    ║  • Web scraping z různých zdrojů                             ║
    ║  • Akademické databáze (arXiv, PubMed, Google Scholar)       ║
    ║  • AI analýza textu a sentiment                              ║
    ║  • Automatické generování reportů                            ║
    ║  • Webové rozhraní pro pohodlné ovládání                     ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

async def run_cli_mode():
    """Spuštění v CLI režimu pro rychlý test"""
    print("🚀 Spouštím Research Tool v CLI režimu...\n")

    # Inicializace komponent
    engine = ResearchEngine()
    db_manager = DatabaseManager()
    await db_manager.initialize()

    # Ukázkový research
    query = ResearchQuery(
        query="artificial intelligence trends 2024",
        sources=["web", "academic"],
        depth="medium",
        max_results=20
    )

    print(f"🔍 Provádím research pro dotaz: '{query.query}'")
    print("⏳ Toto může trvat několik minut...\n")

    # Optimalizace pro lékařské dotazy
    medical_keywords = ["nootropika", "peptidy", "medikace", "psychické poruchy"]
    optimized_query = engine.optimize_query(query.query, medical_keywords)

    # Spuštění výzkumu
    results = await engine.perform_research(optimized_query)

    # Generování reportu
    report = engine.generate_report(results)
    print("📄 Generovaný report:\n")
    print(report)

    # Uložení výsledků do databáze
    await db_manager.save_results(results)
    print("✅ Výsledky byly úspěšně uloženy do databáze.")

def run_web_mode():
    """Spuštění webového rozhraní"""
    print("🌐 Spouštím webové rozhraní...")
    print("📱 Otevře se v prohlížeči na adrese http://localhost:8501")
    print("🛑 Pro ukončení stiskněte Ctrl+C\n")

    # Spuštění Streamlit aplikace
    import subprocess
    import os

    script_path = Path(__file__).parent / "streamlit_app.py"

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(script_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Ukončuji aplikaci...")
    except Exception as e:
        print(f"❌ Chyba při spuštění webového rozhraní: {e}")
        print("💡 Zkuste nainstalovat Streamlit: pip install streamlit")

def check_dependencies():
    """Kontrola závislostí"""
    missing_deps = []

    try:
        import requests
    except ImportError:
        missing_deps.append("requests")

    try:
        import beautifulsoup4
    except ImportError:
        missing_deps.append("beautifulsoup4")

    try:
        import streamlit
    except ImportError:
        missing_deps.append("streamlit")

    if missing_deps:
        print("⚠️  Chybí některé závislosti:")
        for dep in missing_deps:
            print(f"   • {dep}")
        print("\n💡 Nainstalujte je pomocí:")
        print("   pip install -r requirements.txt")
        return False

    return True

async def setup_initial_data():
    """Nastavení počátečních dat"""
    db_manager = DatabaseManager()
    await db_manager.initialize()

    # Vytvoření datových složek
    for dir_path in [config.DATA_DIR, config.RAW_DATA_DIR,
                     config.PROCESSED_DATA_DIR, config.REPORTS_DIR]:
        dir_path.mkdir(exist_ok=True)

    print("📁 Datové složky vytvořeny")
    print("🗄️  Databáze inicializována")

def connect_to_mcp_server():
    server_url = "ws://127.0.0.1:65432"
    try:
        ws = websocket.create_connection(server_url)
        print(f"Připojeno k serveru MCP na {server_url}")
        # Testovací zpráva
        ws.send(json.dumps({"action": "test_connection"}))
        response = ws.recv()
        print(f"Odpověď serveru: {response}")
        ws.close()
    except Exception as e:
        print(f"Chyba při připojení k serveru MCP: {e}")

def main():
    """Hlavní funkce"""
    print_banner()

    parser = argparse.ArgumentParser(description="AI Research Tool")
    parser.add_argument("--web", action="store_true", help="Spustit webové rozhraní")
    parser.add_argument("--cli", action="store_true", help="Spustit v CLI režimu")
    parser.add_argument("--setup", action="store_true", help="Nastavit počáteční konfiguraci")

    args = parser.parse_args()

    # Kontrola závislostí
    if not check_dependencies():
        sys.exit(1)

    if args.setup:
        print("🔧 Nastavuji počáteční konfiguraci...")
        asyncio.run(setup_initial_data())
        print("✅ Konfigurace dokončena!")

    elif args.web:
        run_web_mode()

    elif args.cli:
        asyncio.run(run_cli_mode())

    else:
        # Výchozí režim - nabídka možností
        print("🎯 Vyberte režim spuštění:")
        print("   1. 🌐 Webové rozhraní (doporučeno)")
        print("   2. 💻 CLI režim")
        print("   3. 🔧 Nastavení")
        print("   4. ❌ Ukončit")

        try:
            choice = input("\nVaše volba (1-4): ").strip()

            if choice == "1":
                run_web_mode()
            elif choice == "2":
                asyncio.run(run_cli_mode())
            elif choice == "3":
                asyncio.run(setup_initial_data())
            elif choice == "4":
                print("👋 Na shledanou!")
            else:
                print("❌ Neplatná volba")

        except KeyboardInterrupt:
            print("\n👋 Ukončeno uživatelem")

if __name__ == '__main__':
    main()
    connect_to_mcp_server()
