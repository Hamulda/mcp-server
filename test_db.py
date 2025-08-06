"""
Test databáze - kontrola funkcionality
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from database_manager import DatabaseManager
from research_engine import ResearchQuery, ResearchEngine

async def test_database():
    """Test základní funkcionality databáze"""
    print("🧪 Testuji databázovou funkcionalitu...")

    # Inicializace
    db = DatabaseManager()
    await db.initialize()
    print("✅ Databáze inicializována")

    # Test uložení dotazu
    query_params = {
        "sources": ["web", "academic"],
        "depth": "medium",
        "max_results": 50
    }

    query_id = await db.save_research_query("test artificial intelligence", query_params)
    print(f"✅ Dotaz uložen s ID: {query_id}")

    # Test uložení zdrojů
    mock_sources = [
        {
            "title": "AI Research Paper",
            "content": "Lorem ipsum dolor sit amet...",
            "source": "arXiv",
            "type": "academic_paper",
            "url": "https://arxiv.org/abs/test",
            "authors": ["John Doe", "Jane Smith"]
        },
        {
            "title": "AI News Article",
            "content": "Recent developments in AI...",
            "source": "TechCrunch",
            "type": "web",
            "url": "https://techcrunch.com/test"
        }
    ]

    await db.save_sources(query_id, mock_sources)
    print("✅ Zdroje uloženy")

    # Test statistik
    stats = await db.get_statistics()
    print(f"✅ Statistiky: {stats}")

    # Test historie
    history = await db.get_research_history(limit=5)
    print(f"✅ Historie dotazů: {len(history)} záznamů")

    print("\n🎉 Všechny testy databáze prošly!")

async def test_health_research():
    """
    Testování zdravotního výzkumu pomocí ResearchEngine.
    """
    print("🧪 Testuji zdravotní výzkum...")

    # Inicializace enginu
    engine = ResearchEngine()

    # Vytvoření dotazu zaměřeného na zdraví
    query = ResearchQuery(
        query="peptides in clinical trials",
        sources=["academic"],
        depth="medium",
        max_results=10
    )

    try:
        # Spuštění zdravotního výzkumu
        result = await engine.conduct_health_research(query)

        # Výpis výsledků
        print("✅ Zdravotní výzkum dokončen!")
        print(f"Dotaz: {result.query}")
        print(f"Počet nalezených zdrojů: {result.sources_found}")
        print(f"Shrnutí: {result.summary}")
        print(f"Klíčová zjištění: {result.key_findings}")

    except Exception as e:
        print(f"❌ Chyba při zdravotním výzkumu: {e}")

if __name__ == "__main__":
    asyncio.run(test_database())
    asyncio.run(test_health_research())
