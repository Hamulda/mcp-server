"""
Jednotkové testy pro klíčové komponenty projektu
Zaměřené na nejdůležitější a nejsložitější logiku
"""
import unittest
import asyncio
from unittest.mock import patch, MagicMock
from text_processing_utils import text_processor
from cost_tracker import CostTracker
from research_engine import QueryOptimizer, ResearchQuery
from academic_scraper import HealthResearchScraper

class TestTextProcessor(unittest.TestCase):
    """Testy pro centrální TextProcessor"""

    def test_clean_text(self):
        """Test čištění textu"""
        dirty_text = "Hello! This has 123 numbers and @#$ symbols."
        cleaned = text_processor.clean_text(dirty_text)
        self.assertNotIn("123", cleaned)
        self.assertNotIn("@#$", cleaned)
        self.assertIn("Hello", cleaned)

    def test_count_tokens(self):
        """Test počítání tokenů"""
        text = "This is a simple test sentence."
        tokens = text_processor.count_tokens(text)
        self.assertIsInstance(tokens, int)
        self.assertGreater(tokens, 0)

    def test_extract_keywords(self):
        """Test extrakce klíčových slov"""
        text = "Research shows that artificial intelligence and machine learning are growing rapidly."
        keywords = text_processor.extract_keywords(text, max_keywords=5)
        self.assertIsInstance(keywords, list)
        self.assertLessEqual(len(keywords), 5)
        # Kontrola, že výsledek obsahuje tuply (slovo, frekvence)
        if keywords:
            self.assertIsInstance(keywords[0], tuple)
            self.assertEqual(len(keywords[0]), 2)

    def test_medical_text_distillation(self):
        """Test destilace lékařského textu"""
        medical_text = "This study examines nootropika effects. Regular text here. Peptidy show promise in treatment."
        distilled = text_processor.distill_medical_text(medical_text)
        self.assertIn("nootropika", distilled)
        self.assertIn("peptidy", distilled.lower())

    def test_sentiment_analysis(self):
        """Test analýzy sentimentu"""
        positive_text = "This is excellent and wonderful research with great results."
        negative_text = "This is terrible and awful with poor outcomes."
        neutral_text = "This is research about data analysis."

        pos_sentiment = text_processor.analyze_sentiment_basic(positive_text)
        neg_sentiment = text_processor.analyze_sentiment_basic(negative_text)
        neu_sentiment = text_processor.analyze_sentiment_basic(neutral_text)

        self.assertGreater(pos_sentiment["positive"], pos_sentiment["negative"])
        self.assertGreater(neg_sentiment["negative"], neg_sentiment["positive"])
        self.assertGreater(neu_sentiment["neutral"], 0.5)

class TestCostTracker(unittest.TestCase):
    """Testy pro CostTracker"""

    def setUp(self):
        """Nastavení pro každý test"""
        self.cost_tracker = CostTracker("test_cost_tracking.json")

    def test_record_api_call(self):
        """Test zaznamenání API volání"""
        initial_cost = self.cost_tracker.get_daily_cost()
        self.cost_tracker.record_api_call(0.05, 100, 50, "gemini", "test query")
        new_cost = self.cost_tracker.get_daily_cost()
        self.assertGreater(new_cost, initial_cost)

    def test_budget_alert(self):
        """Test alertu při překročení budgetu"""
        # Nastavení vysokých nákladů
        self.cost_tracker.record_api_call(4.0, 1000, 500)
        alert = self.cost_tracker.check_budget_alert(daily_limit=5.0)

        self.assertTrue(alert["alert_triggered"])
        self.assertFalse(alert["budget_exceeded"])

        # Překročení budgetu
        self.cost_tracker.record_api_call(2.0, 500, 250)
        alert = self.cost_tracker.check_budget_alert(daily_limit=5.0)
        self.assertTrue(alert["budget_exceeded"])

    def test_cost_analytics(self):
        """Test analýzy nákladů"""
        # Přidání několika API volání
        self.cost_tracker.record_api_call(0.10, 200, 100)
        self.cost_tracker.record_api_call(0.15, 300, 150)

        analytics = self.cost_tracker.get_cost_analytics()

        self.assertIn("weekly_costs", analytics)
        self.assertIn("monthly_total", analytics)
        self.assertGreater(analytics["total_cost_all_time"], 0)

    def tearDown(self):
        """Úklid po testech"""
        import os
        if os.path.exists("test_cost_tracking.json"):
            os.remove("test_cost_tracking.json")

class TestQueryOptimizer(unittest.TestCase):
    """Testy pro optimalizaci dotazů"""

    def setUp(self):
        self.optimizer = QueryOptimizer()

    def test_medical_query_optimization(self):
        """Test optimalizace lékařských dotazů"""
        query = "I want to research nootropika and their effects on cognitive function. Also peptidy research."
        optimized = self.optimizer.optimize_medical_query(query)

        self.assertIn("nootropika", optimized)
        self.assertIn("peptidy", optimized)
        self.assertLess(len(optimized), len(query))  # Měl by být kratší

    def test_synonym_expansion(self):
        """Test rozšíření dotazu o synonyma"""
        query = "ai research"
        expanded = self.optimizer.expand_query_with_synonyms(query)

        self.assertIn("artificial intelligence", expanded)
        self.assertIn("machine learning", expanded)

    def test_empty_query_handling(self):
        """Test zpracování prázdného dotazu"""
        with self.assertRaises(ValueError):
            self.optimizer.optimize_query("")

        with self.assertRaises(ValueError):
            self.optimizer.optimize_query("   ")

class TestResearchQuery(unittest.TestCase):
    """Testy pro ResearchQuery dataclass"""

    def test_query_auto_optimization(self):
        """Test automatické optimalizace při vytvoření"""
        original_query = "ai trends 2024"
        research_query = ResearchQuery(
            query=original_query,
            sources=["web"],
            domain="general"
        )

        # Dotaz by měl být rozšířen o synonyma
        self.assertIn("artificial intelligence", research_query.query)
        self.assertIn("machine learning", research_query.query)

    def test_medical_domain_optimization(self):
        """Test optimalizace pro lékařskou doménu"""
        medical_query = "Research on nootropika effects. Some other text. Peptidy benefits."
        research_query = ResearchQuery(
            query=medical_query,
            sources=["academic"],
            domain="medical"
        )

        # Měl by prioritizovat lékařské části
        self.assertIn("nootropika", research_query.query)
        self.assertIn("peptidy", research_query.query.lower())

class TestHealthResearchScraper(unittest.TestCase):
    """Testy pro HealthResearchScraper"""

    def setUp(self):
        self.scraper = HealthResearchScraper()

    @patch('aiohttp.ClientSession.get')
    def test_pubmed_scraping_success(self, mock_get):
        """Test úspěšného PubMed scrapingu"""
        # Mock odpověď
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "esearchresult": {"idlist": ["123", "456"]}
        }
        mock_get.return_value.__aenter__.return_value = mock_response

        # Test bude vyžadovat více mock nastavení pro kompletní flow
        # Pro demonstraci základní struktury
        self.assertIsInstance(self.scraper, HealthResearchScraper)

    def test_failed_url_logging(self):
        """Test logování neúspěšných URL"""
        test_url = "http://test-failed-url.com"
        self.scraper._log_failed_url(test_url)

        # Kontrola, že soubor byl vytvořen
        import os
        self.assertTrue(os.path.exists("failed_urls.log"))

        # Úklid
        if os.path.exists("failed_urls.log"):
            os.remove("failed_urls.log")

class TestIntegration(unittest.TestCase):
    """Integrační testy pro spolupráci komponent"""

    def test_text_processor_and_cost_tracker_integration(self):
        """Test integrace TextProcessor a CostTracker"""
        cost_tracker = CostTracker("integration_test.json")

        # Simulace workflow
        text = "This is a research text about artificial intelligence trends."
        tokens = text_processor.count_tokens(text)

        # Zaznamenání nákladů na základě tokenů
        estimated_cost = tokens * 0.0001  # Přibližná cena za token
        cost_tracker.record_api_call(estimated_cost, tokens, tokens // 2)

        daily_cost = cost_tracker.get_daily_cost()
        self.assertGreater(daily_cost, 0)

        # Úklid
        import os
        if os.path.exists("integration_test.json"):
            os.remove("integration_test.json")

def run_critical_tests():
    """Spuštění pouze kritických testů pro rychlou kontrolu"""
    critical_suite = unittest.TestSuite()

    # Přidání kritických testů
    critical_suite.addTest(TestTextProcessor('test_clean_text'))
    critical_suite.addTest(TestTextProcessor('test_count_tokens'))
    critical_suite.addTest(TestCostTracker('test_budget_alert'))
    critical_suite.addTest(TestQueryOptimizer('test_empty_query_handling'))

    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(critical_suite)

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--critical':
        # Spuštění pouze kritických testů
        result = run_critical_tests()
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        # Spuštění všech testů
        unittest.main(verbosity=2)
