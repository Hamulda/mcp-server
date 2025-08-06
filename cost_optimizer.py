"""
Úsporný manager pro minimalizaci nákladů
Cíl: Být levnější než Perplexity ($20/měsíc)
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from high_performance_cache import high_perf_cache
from config_personal import *

class CostOptimizer:
    """Inteligentní optimalizace nákladů"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.daily_spending = 0.0
        self.monthly_spending = 0.0

    def should_use_api(self, estimated_cost: float) -> bool:
        """Rozhodnutí zda použít API nebo cache/fallback"""
        # Kontrola denního limitu
        if self.daily_spending + estimated_cost > DAILY_COST_LIMIT:
            self.logger.warning(f"💰 Denní limit dosažen, používám cache")
            return False

        # Kontrola měsíčního trendu
        daily_average = self.monthly_spending / max(1, datetime.now().day)
        projected_monthly = daily_average * 30

        if projected_monthly > MONTHLY_TARGET_COST:
            self.logger.warning(f"📊 Měsíční trend vysoký, šetřím")
            return False

        return True

    def optimize_query_for_cost(self, query: str) -> str:
        """Optimalizace dotazu pro minimální náklady"""
        # Zkrať dotaz pokud je příliš dlouhý
        words = query.split()
        if len(words) > 10:
            # Ponech pouze klíčová slova
            important_words = [w for w in words if len(w) > 3 and w.lower() not in {'with', 'from', 'that', 'this'}]
            query = ' '.join(important_words[:8])

        return query

    def get_cost_savings_report(self) -> Dict[str, Any]:
        """Report úspor oproti komerčním službám"""
        perplexity_monthly = 20.0  # $20/měsíc
        chatgpt_plus_monthly = 20.0
        claude_pro_monthly = 20.0

        your_monthly = self.monthly_spending

        savings_perplexity = perplexity_monthly - your_monthly
        savings_percentage = (savings_perplexity / perplexity_monthly) * 100

        return {
            "your_monthly_cost": your_monthly,
            "perplexity_cost": perplexity_monthly,
            "monthly_savings": savings_perplexity,
            "savings_percentage": savings_percentage,
            "yearly_savings": savings_perplexity * 12
        }

class SmartCacheManager:
    """Inteligentní cache pro maximální úspory"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.hit_rate_target = MIN_CACHE_HIT_RATE

    def should_cache_aggressively(self, query: str) -> bool:
        """Rozhodnutí o agresivním cachování"""
        # Podobné dotazy se cachují déle
        similar_queries = self.find_similar_cached_queries(query)
        if len(similar_queries) > 2:
            return True

        # Časté termíny se cachují déle
        common_terms = ['ai', 'artificial intelligence', 'research', 'study', 'analysis']
        if any(term in query.lower() for term in common_terms):
            return True

        return False

    def find_similar_cached_queries(self, query: str) -> List[str]:
        """Najdi podobné cached dotazy"""
        query_words = set(query.lower().split())
        similar = []

        # Zde by byla implementace hledání v cache
        # Pro demo vracím prázdný seznam
        return similar

# Globální instance pro celou aplikaci
cost_optimizer = CostOptimizer()
smart_cache = SmartCacheManager()
