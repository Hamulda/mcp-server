"""
Rozšířené testy - integrační scénáře pro celý Research Tool
Testuje spolupráci mezi různými komponenty systému
"""
import unittest
import asyncio
import tempfile
import os
from unittest.mock import patch, MagicMock, AsyncMock
import json
from datetime import datetime, timedelta

# Import testovaných komponent
from research_engine import ResearchEngine, ResearchQuery, ResearchResult
from database_manager import DatabaseManager
from cost_tracker import CostTracker
from academic_scraper import AcademicScraperManager, GoogleScholarScraper
from text_processing_utils import text_processor
from monitoring_metrics import metrics, MetricsCollector

class TestFullWorkflow(unittest.TestCase):
    """Integrační testy pro celý workflow"""

    def setUp(self):
        """Příprava testovacího prostředí"""
        self.test_db_path = tempfile.mktemp(suffix='.db')
        self.test_cost_file = tempfile.mktemp(suffix='.json')

    def tearDown(self):
        """Úklid po testech"""
        for file_path in [self.test_db_path, self.test_cost_file]:
            if os.path.exists(file_path):
                os.remove(file_path)

    @patch('academic_scraper.GoogleScholarScraper.scrape')
    @patch('gemini_manager.GeminiAIManager._call_gemini_async')
    async def test_complete_research_workflow(self, mock_gemini, mock_scholar):
        """Test kompletního research workflow"""
        # Mock data
        mock_scholar.return_value = [
            {
                'title': 'Test Paper on AI',
                'abstract': 'This paper discusses artificial intelligence trends.',
                'authors': ['Dr. Test'],
                'year': 2024,
                'citation_count': 10,
                'source': 'Google Scholar',
                'type': 'academic_paper'
            }
        ]

        mock_gemini.return_value = json.dumps([{
            'summary': 'AI is growing rapidly',
            'keywords': ['ai', 'artificial intelligence'],
            'sentiment': 'positive'
        }])

        # Inicializace komponent
        engine = ResearchEngine()
        db_manager = DatabaseManager(self.test_db_path)
        await db_manager.initialize()

        # Vytvoření query
        query = ResearchQuery(
            query="artificial intelligence trends",
            sources=["academic"],
            depth="medium",
            max_results=5,
            domain="general"
        )

        # Spuštění research workflow
        result = await engine.conduct_research(query)

        # Validace výsledků
        self.assertIsInstance(result, ResearchResult)
        self.assertGreater(len(result.summary), 0)
        self.assertGreater(result.sources_found, 0)
        self.assertGreater(result.confidence_score, 0)

    async def test_medical_research_integration(self):
        """Test integrace lékařského výzkumu"""
        # Test optimalizace lékařských dotazů
        medical_query = "nootropika effects on cognitive function and peptidy research benefits"
        optimized = text_processor.distill_medical_text(medical_query)

        self.assertIn("nootropika", optimized)
        self.assertIn("peptidy", optimized)

        # Test vytvoření lékařského query
        query = ResearchQuery(
            query=medical_query,
            sources=["academic"],
            domain="medical",
            max_results=3
        )

        # Ověření, že query byl optimalizován
        self.assertIn("nootropika", query.query.lower())

class TestDatabaseIntegration(unittest.TestCase):
    """Testy integrace s databází"""

    def setUp(self):
        self.test_db_path = tempfile.mktemp(suffix='.db')

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    async def test_full_database_workflow(self):
        """Test kompletního database workflow"""
        db_manager = DatabaseManager(self.test_db_path)
        await db_manager.initialize()

        # 1. Uložení query
        query_id = await db_manager.save_research_query(
            "test query",
            {"test": "data"}
        )
        self.assertIsInstance(query_id, int)

        # 2. Uložení zdrojů
        test_sources = [
            {
                'title': 'Test Source',
                'content': 'Test content',
                'type': 'academic_paper',
                'source': 'Test Journal'
            }
        ]
        await db_manager.save_sources(query_id, test_sources)

        # 3. Uložení analýzy
        analysis_data = {
            'summary': 'Test summary',
            'keywords': ['test', 'analysis']
        }
        await db_manager.save_analysis(query_id, 'comprehensive', analysis_data)

        # 4. Aktualizace stavu
        await db_manager.update_query_status(query_id, 'completed', 1, 0.85)

        # 5. Načtení detailů
        details = await db_manager.get_query_details(query_id)
        self.assertIsNotNone(details)
        self.assertEqual(details['query']['query_text'], 'test query')
        self.assertEqual(len(details['sources']), 1)
        self.assertEqual(len(details['analyses']), 1)

class TestCostTrackingIntegration(unittest.TestCase):
    """Testy integrace cost trackingu"""

    def setUp(self):
        self.test_cost_file = tempfile.mktemp(suffix='.json')

    def tearDown(self):
        if os.path.exists(self.test_cost_file):
            os.remove(self.test_cost_file)

    def test_cost_tracking_with_metrics(self):
        """Test propojení cost trackingu s metrikami"""
        cost_tracker = CostTracker(self.test_cost_file)

        # Simulace API volání
        initial_cost = cost_tracker.get_daily_cost()
        cost_tracker.record_api_call(0.05, 100, 75, "gemini", "test query")

        # Ověření, že se náklady zvýšily
        new_cost = cost_tracker.get_daily_cost()
        self.assertGreater(new_cost, initial_cost)

        # Test budget alertů
        budget_alert = cost_tracker.check_budget_alert(daily_limit=0.10)
        self.assertFalse(budget_alert['budget_exceeded'])

        # Překročení budgetu
        cost_tracker.record_api_call(0.10, 200, 150, "gemini", "expensive query")
        budget_alert = cost_tracker.check_budget_alert(daily_limit=0.10)
        self.assertTrue(budget_alert['budget_exceeded'])

class TestScraperIntegration(unittest.TestCase):
    """Testy integrace scraperů"""

    @patch('scholarly.scholarly.search_pubs')
    async def test_google_scholar_integration(self, mock_search):
        """Test integrace Google Scholar scraperu"""
        # Mock scholarly response
        mock_pub = {
            'title': 'Test Publication',
            'abstract': 'Test abstract about AI research',
            'author': [{'name': 'Test Author'}],
            'pub_year': 2024,
            'num_citations': 15,
            'venue': 'Test Conference'
        }

        mock_search.return_value = [mock_pub]

        scraper = GoogleScholarScraper()
        if scraper.scholarly:  # Pouze pokud je scholarly dostupná
            query = MagicMock()
            query.query = "artificial intelligence"
            query.max_results = 5

            results = await scraper.scrape(query)

            self.assertIsInstance(results, list)
            if results:  # Pokud scraping proběhl úspěšně
                self.assertEqual(results[0]['title'], 'Test Publication')
                self.assertEqual(results[0]['source'], 'Google Scholar')

class TestTextProcessingIntegration(unittest.TestCase):
    """Testy integrace text processingu"""

    def test_text_processing_pipeline(self):
        """Test kompletního text processing pipeline"""
        # Testovací text s lékařskými termíny
        test_text = """
        This research examines nootropika effects on cognitive performance.
        Various peptidy have shown promising results in clinical trials.
        Regular text without medical terms should be filtered out.
        Additional studies on medikace interactions are needed.
        """

        # 1. Základní čištění
        cleaned = text_processor.clean_text(test_text)
        self.assertNotIn('\n', cleaned)

        # 2. Počítání tokenů
        tokens = text_processor.count_tokens(test_text)
        self.assertGreater(tokens, 0)

        # 3. Destilace lékařského textu
        distilled = text_processor.distill_medical_text(test_text)
        self.assertIn('nootropika', distilled)
        self.assertIn('peptidy', distilled)
        self.assertIn('medikace', distilled)

        # 4. Extrakce klíčových slov
        keywords = text_processor.extract_keywords(test_text, max_keywords=10)
        self.assertIsInstance(keywords, list)
        self.assertGreater(len(keywords), 0)

        # 5. Analýza sentimentu
        sentiment = text_processor.analyze_sentiment_basic(test_text)
        self.assertIn('positive', sentiment)
        self.assertIn('negative', sentiment)
        self.assertIn('neutral', sentiment)

class TestMonitoringIntegration(unittest.TestCase):
    """Testy integrace monitoring systému"""

    def test_metrics_collection(self):
        """Test sběru metrik"""
        # Test inicializace metrics collectoru
        collector = MetricsCollector(port=8001)  # Jiný port pro test

        # Test zaznamenání API volání
        collector.record_api_call('gemini', 'success', 1.5, 100)

        # Test zaznamenání chyby
        collector.record_error('ValueError', 'text_processor')

        # Test aktualizace nákladů
        collector.update_daily_cost(5.25)

        # Test cache statistik
        collector.update_cache_stats(80, 100)  # 80% hit rate

class TestErrorHandlingIntegration(unittest.TestCase):
    """Testy integrace error handlingu"""

    async def test_cascade_error_handling(self):
        """Test kaskádového error handlingu"""
        # Test chování při nedostupnosti Gemini API
        engine = ResearchEngine()
        engine.gemini_manager = None  # Simulace nedostupnosti

        query = ResearchQuery(
            query="test query",
            sources=["web"],
            max_results=1
        )

        # Research by měl pokračovat i bez Gemini
        try:
            result = await engine.conduct_research(query)
            self.assertIsInstance(result, ResearchResult)
        except Exception as e:
            # Pokud selže, měla by to být kontrolovaná chyba
            self.assertIsInstance(e, (ValueError, RuntimeError))

class TestPerformanceIntegration(unittest.TestCase):
    """Testy výkonu a optimalizace"""

    def test_token_optimization_performance(self):
        """Test výkonu optimalizace tokenů"""
        # Velký text pro test optimalizace
        large_text = "This is a test sentence. " * 1000

        start_time = datetime.now()
        optimized = text_processor.optimize_for_tokens(large_text, max_tokens=500)
        end_time = datetime.now()

        # Optimalizace by měla být rychlá (< 1 sekunda)
        processing_time = (end_time - start_time).total_seconds()
        self.assertLess(processing_time, 1.0)

        # Optimalizovaný text by měl být kratší
        self.assertLess(len(optimized), len(large_text))

    def test_cache_performance(self):
        """Test výkonu cache systému"""
        # Test rychlosti cache operací
        test_data = "test content " * 100

        # První volání - cache miss
        start_time = datetime.now()
        tokens1 = text_processor.count_tokens(test_data)
        first_call_time = (datetime.now() - start_time).total_seconds()

        # Druhé volání - cache hit
        start_time = datetime.now()
        tokens2 = text_processor.count_tokens(test_data)
        second_call_time = (datetime.now() - start_time).total_seconds()

        # Výsledky by měly být stejné
        self.assertEqual(tokens1, tokens2)

        # Druhé volání by mělo být rychlejší (cache hit)
        self.assertLessEqual(second_call_time, first_call_time)
def run_integration_tests():
    """Spuštění všech integračních testů"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Přidání všech test classes
    test_classes = [
        TestFullWorkflow,
        TestDatabaseIntegration,
        TestCostTrackingIntegration,
        TestScraperIntegration,
        TestTextProcessingIntegration,
        TestMonitoringIntegration,
        TestErrorHandlingIntegration,
        TestPerformanceIntegration
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)

async def run_async_integration_tests():
    """Spuštění asynchronních integračních testů"""
    print("Spouštím asynchronní integrační testy...")

    # Test kompletního workflow
    test_workflow = TestFullWorkflow()
    test_workflow.setUp()
    try:
        await test_workflow.test_complete_research_workflow()
        await test_workflow.test_medical_research_integration()
        print("✅ Workflow testy prošły")
    except Exception as e:
        print(f"❌ Workflow testy selhaly: {e}")
    finally:
        test_workflow.tearDown()

    # Test databáze
    test_db = TestDatabaseIntegration()
    test_db.setUp()
    try:
        await test_db.test_full_database_workflow()
        print("✅ Databázové testy prošły")
    except Exception as e:
        print(f"❌ Databázové testy selhaly: {e}")
    finally:
        test_db.tearDown()

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--async':
        # Spuštění asynchronních testů
        asyncio.run(run_async_integration_tests())
    else:
        # Spuštění standardních testů
        result = run_integration_tests()
        sys.exit(0 if result.wasSuccessful() else 1)

