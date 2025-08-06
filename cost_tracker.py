"""
Cost Tracker pro monitoring nákladů na Gemini API
"""
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable
from pathlib import Path
import logging
import hashlib
import time
import threading
from benchmarking_tools import BenchmarkingTools
from monitoring_metrics import metrics, MetricsMiddleware, health_checker

class CostTracker:
    """Sledování nákladů s monitoring metrikami"""

    def __init__(self, cost_file: str = "cost_tracking.json"):
        self.cost_file = Path(cost_file)
        self.logger = logging.getLogger(__name__)
        self.costs_data = self._load_costs()
        self.benchmarking_tools = BenchmarkingTools()
        self.metrics_middleware = MetricsMiddleware("cost_tracker")

    def _load_costs(self) -> Dict[str, Any]:
        """Načtení historických nákladů"""
        if self.cost_file.exists():
            try:
                with open(self.cost_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Chyba při načítání cost dat: {e}")

        return {
            "daily_costs": {},
            "monthly_totals": {},
            "api_usage": {},
            "token_usage": {}
        }

    def _save_costs(self):
        """Uložení cost dat"""
        try:
            with open(self.cost_file, 'w') as f:
                json.dump(self.costs_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Chyba při ukládání cost dat: {e}")

    @MetricsMiddleware("cost_tracker").track_operation("record_api_call")
    def record_api_call(self,
                       cost: float,
                       tokens_input: int,
                       tokens_output: int,
                       api_type: str = "gemini",
                       query: str = ""):
        """Zaznamenání API call s náklady a metrikami"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Denní náklady
        if today not in self.costs_data["daily_costs"]:
            self.costs_data["daily_costs"][today] = {
                "total_cost": 0,
                "api_calls": 0,
                "tokens_input": 0,
                "tokens_output": 0,
                "queries": []
            }
        
        daily = self.costs_data["daily_costs"][today]
        daily["total_cost"] += cost
        daily["api_calls"] += 1
        daily["tokens_input"] += tokens_input
        daily["tokens_output"] += tokens_output
        daily["queries"].append({
            "query": query[:100],  # Zkrácený dotaz
            "cost": cost,
            "timestamp": datetime.now().isoformat(),
            "tokens": {"input": tokens_input, "output": tokens_output}
        })
        
        # API usage tracking
        if api_type not in self.costs_data["api_usage"]:
            self.costs_data["api_usage"][api_type] = {
                "total_calls": 0,
                "total_cost": 0,
                "total_tokens": 0
            }
        
        api_usage = self.costs_data["api_usage"][api_type]
        api_usage["total_calls"] += 1
        api_usage["total_cost"] += cost
        api_usage["total_tokens"] += tokens_input + tokens_output

        # Optimalizace kvality výzkumu
        self._optimize_quality(query, tokens_input, tokens_output)
        
        self._save_costs()

        # Aktualizace metrik
        metrics.record_api_call(api_type, "success", 0, tokens_input + tokens_output)
        metrics.update_daily_cost(self.get_daily_cost())

        self.logger.info(f"💰 Zaznamenán API call: ${cost:.4f} | Tokens: {tokens_input + tokens_output}")

    def _optimize_quality(self, query: str, tokens_input: int, tokens_output: int):
        """Optimalizace kvality výzkumu a výsledků"""
        # Zlepšení kvality shrnutí na základě tokenů
        if tokens_output > tokens_input * 0.3:
            self.logger.info("📊 Výstupní tokeny jsou vyšší než očekávané - zvažte zkrácení dotazu.")
        
        # Doporučení pro zlepšení kvality
        if len(query.split()) < 5:
            self.logger.warning("⚠️ Dotaz je příliš krátký - doporučujeme přidat více klíčových slov.")
        elif len(query.split()) > 20:
            self.logger.warning("⚠️ Dotaz je příliš dlouhý - zvažte jeho zjednodušení.")
        
        # Analýza efektivity tokenů
        efficiency = tokens_output / tokens_input if tokens_input > 0 else 0
        if efficiency < 0.5:
            self.logger.info("📉 Efektivita tokenů je nízká - zvažte použití více batch zpracování.")
        elif efficiency > 0.8:
            self.logger.info("📈 Efektivita tokenů je vysoká - výzkum je optimalizován.")

    def get_daily_cost(self, date: str = None) -> float:
        """Získání denních nákladů"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        return self.costs_data["daily_costs"].get(date, {}).get("total_cost", 0.0)

    def get_weekly_costs(self) -> Dict[str, float]:
        """Náklady za posledních 7 dní"""
        weekly_costs = {}

        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            weekly_costs[date] = self.get_daily_cost(date)

        return weekly_costs

    def get_monthly_total(self) -> float:
        """Náklady za aktuální měsíc"""
        current_month = datetime.now().strftime("%Y-%m")
        total = 0.0

        for date, data in self.costs_data["daily_costs"].items():
            if date.startswith(current_month):
                total += data.get("total_cost", 0)

        return total

    def check_budget_alert(self, daily_limit: float = 5.0, alert_threshold: float = 0.8) -> Dict[str, Any]:
        """Kontrola překročení budgetu"""
        today_cost = self.get_daily_cost()
        alert_limit = daily_limit * alert_threshold

        return {
            "today_cost": today_cost,
            "daily_limit": daily_limit,
            "alert_threshold": alert_limit,
            "alert_triggered": today_cost >= alert_limit,
            "budget_exceeded": today_cost >= daily_limit,
            "remaining_budget": max(0, daily_limit - today_cost),
            "usage_percentage": min(100, (today_cost / daily_limit) * 100)
        }

    def get_cost_analytics(self) -> Dict[str, Any]:
        """Analýza nákladů a usage"""
        weekly_costs = self.get_weekly_costs()
        monthly_total = self.get_monthly_total()

        # Průměrné denní náklady
        non_zero_days = [cost for cost in weekly_costs.values() if cost > 0]
        avg_daily = sum(non_zero_days) / len(non_zero_days) if non_zero_days else 0

        # Nejvíce nákladný den
        max_day = max(weekly_costs.items(), key=lambda x: x[1]) if weekly_costs else ("N/A", 0)

        # Token efficiency
        total_tokens = sum(
            data.get("tokens_input", 0) + data.get("tokens_output", 0)
            for data in self.costs_data["daily_costs"].values()
        )

        total_cost = sum(
            data.get("total_cost", 0)
            for data in self.costs_data["daily_costs"].values()
        )

        cost_per_token = total_cost / total_tokens if total_tokens > 0 else 0

        return {
            "weekly_costs": weekly_costs,
            "monthly_total": monthly_total,
            "average_daily": avg_daily,
            "most_expensive_day": {"date": max_day[0], "cost": max_day[1]},
            "total_tokens_used": total_tokens,
            "total_cost_all_time": total_cost,
            "cost_per_token": cost_per_token,
            "api_breakdown": self.costs_data["api_usage"]
        }

    def optimize_usage_recommendations(self) -> List[str]:
        """Doporučení pro optimalizaci nákladů"""
        analytics = self.get_cost_analytics()
        recommendations = []

        # Vysoké náklady
        if analytics["average_daily"] > 2.0:
            recommendations.append("🔥 Průměrné denní náklady jsou vysoké - zvažte shallow research místo deep")

        # Vysoké náklady na token
        if analytics["cost_per_token"] > 0.0001:
            recommendations.append("📊 Vysoké náklady na token - použijte více batch zpracování")

        # Neefektivní usage
        total_calls = sum(
            api["total_calls"] for api in analytics["api_breakdown"].values()
        )

        if total_calls > 100 and analytics["monthly_total"] > 50:
            recommendations.append("⚡ Vysoký počet API calls - implementujte agresivnější caching")

        # Pozitivní trendy
        weekly_costs = list(analytics["weekly_costs"].values())
        if len(weekly_costs) >= 3:
            recent_trend = sum(weekly_costs[:3]) / 3
            older_trend = sum(weekly_costs[3:6]) / 3 if len(weekly_costs) >= 6 else recent_trend

            if recent_trend < older_trend * 0.8:
                recommendations.append("✅ Náklady klesají - dobré optimalizace!")

        if not recommendations:
            recommendations.append("✅ Náklady jsou v rozumných mezích")

        return recommendations

    def predict_monthly_costs(self) -> Dict[str, float]:
        """
        Předpověď měsíčních nákladů na základě historických dat.
        Vrací odhad nákladů pro aktuální měsíc a následující měsíc.
        """
        current_month = datetime.now().strftime("%Y-%m")
        next_month = (datetime.now().replace(day=1) + timedelta(days=32)).strftime("%Y-%m")

        # Součet nákladů za aktuální měsíc
        current_month_cost = sum(
            data.get("total_cost", 0)
            for date, data in self.costs_data["daily_costs"].items()
            if date.startswith(current_month)
        )

        # Odhad na základě průměrných denních nákladů
        days_in_month = (datetime.now().replace(day=1) + timedelta(days=32)).replace(day=1) - datetime.now().replace(day=1)
        avg_daily_cost = current_month_cost / max(1, datetime.now().day)
        next_month_cost = avg_daily_cost * days_in_month.days

        return {
            "current_month": current_month_cost,
            "next_month": next_month_cost
        }

    def auto_optimize(self):
        """
        Automatická optimalizace parametrů API volání pro snížení nákladů.
        Dynamicky upravuje batch velikost a typ analýzy na základě aktuálních nákladů.
        """
        analytics = self.get_cost_analytics()
        avg_daily_cost = analytics.get("average_daily", 0)

        # Dynamická úprava batch velikosti
        if avg_daily_cost > 2.0:
            self.logger.warning("🔥 Průměrné denní náklady jsou vysoké - snižuji batch velikost.")
            return {"batch_size": 3, "analysis_type": "quick"}
        elif avg_daily_cost < 1.0:
            self.logger.info("✅ Náklady jsou nízké - zvyšuji batch velikost.")
            return {"batch_size": 8, "analysis_type": "comprehensive"}
        else:
            self.logger.info("📊 Náklady jsou stabilní - zachovávám aktuální nastavení.")
            return {"batch_size": 5, "analysis_type": "summary"}

    def generate_cost_report(self) -> Dict[str, Any]:
        """
        Vytvoření strukturovaného přehledu nákladů ve formátu JSON/dict.
        Obsahuje denní, týdenní a měsíční náklady, efektivitu tokenů a doporučení.
        """
        analytics = self.get_cost_analytics()
        weekly_costs = analytics.get("weekly_costs", {})
        monthly_total = analytics.get("monthly_total", 0)
        cost_per_token = analytics.get("cost_per_token", 0)
        recommendations = self.optimize_usage_recommendations()

        return {
            "daily_costs": self.get_daily_cost(),
            "weekly_costs": weekly_costs,
            "monthly_total": monthly_total,
            "cost_per_token": cost_per_token,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }

    def analyze_token_efficiency(self) -> Dict[str, Any]:
        """
        Analýza efektivity tokenů na základě historických dat.
        Vrací přehled o poměru vstupních a výstupních tokenů a identifikuje neefektivní dotazy.
        """
        total_input_tokens = sum(
            data.get("tokens_input", 0)
            for data in self.costs_data["daily_costs"].values()
        )
        total_output_tokens = sum(
            data.get("tokens_output", 0)
            for data in self.costs_data["daily_costs"].values()
        )

        efficiency_ratio = total_output_tokens / total_input_tokens if total_input_tokens > 0 else 0

        inefficient_queries = [
            query["query"] for date, data in self.costs_data["daily_costs"].items()
            for query in data.get("queries", [])
            if query["tokens"]["output"] < query["tokens"]["input"] * 0.3
        ]

        return {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "efficiency_ratio": efficiency_ratio,
            "inefficient_queries": inefficient_queries
        }

    def optimize_query(self, query: str) -> str:
        """
        Optimalizace dotazu na základě historických dat a token usage.
        Zkracuje nebo rozšiřuje dotaz podle efektivity tokenů.
        """
        token_analysis = self.analyze_token_efficiency()
        efficiency_ratio = token_analysis.get("efficiency_ratio", 1)

        if efficiency_ratio < 0.5:
            self.logger.warning("⚠️ Nízká efektivita tokenů - zkracuji dotaz.")
            return " ".join(query.split()[:5])  # Zkrácení na prvních 5 slov
        elif efficiency_ratio > 0.8:
            self.logger.info("✅ Vysoká efektivita tokenů - rozšiřuji dotaz.")
            return query + " detailed analysis trends"
        else:
            self.logger.info("📊 Efektivita tokenů je stabilní - dotaz zachován.")
            return query

    def preprocess_text_for_tokens(self, text: str, max_tokens: int = 3000) -> str:
        """
        Zkrácení textu na základě klíčových slov a relevance pro snížení tokenů.
        """
        sentences = text.split(".")
        important_keywords = ["research", "analysis", "trend", "key", "finding"]

        scored_sentences = [
            (sentence, sum(1 for word in important_keywords if word in sentence.lower()))
            for sentence in sentences
        ]

        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        optimized_text = " ".join(
            sentence for sentence, score in scored_sentences[:max_tokens // 50]  # Přibližně 50 tokenů na větu
        )

        return optimized_text

    def ab_test_optimization_strategies(self, strategy_a: Callable, strategy_b: Callable, *args, **kwargs) -> Dict[str, Any]:
        """A/B testování dvou optimalizačních strategií"""
        self.logger.info("Spouštím A/B testování optimalizačních strategií...")
        return self.benchmarking_tools.ab_test(strategy_a, strategy_b, *args, **kwargs)

class SmartBatchProcessor:
    """Inteligentní batch processor pro minimalizaci API calls"""

    def __init__(self, cost_tracker: CostTracker):
        self.cost_tracker = cost_tracker
        self.logger = logging.getLogger(__name__)
        self.batch_cache = {}
        self.cache_expiry = timedelta(hours=6)  # Expirační čas cache

    async def process_with_smart_batching(self,
                                        texts: List[str],
                                        analysis_type: str,
                                        gemini_manager) -> List[Dict[str, Any]]:
        """Inteligentní batch zpracování s cache a optimalizací"""

        # 1. Kontrola cache
        cached_results = self._check_cache(texts, analysis_type)
        uncached_texts = []
        cache_indices = {}

        for i, text in enumerate(texts):
            cache_key = self._generate_cache_key(text, analysis_type)
            if cache_key in cached_results:
                cache_indices[i] = cached_results[cache_key]
            else:
                uncached_texts.append((i, text))

        # 2. Zpracování pouze uncached textů
        if not uncached_texts:
            self.logger.info("🎯 Všechny výsledky načteny z cache - 0 API calls!")
            return [cached_results[self._generate_cache_key(text, analysis_type)] for text in texts]

        # 3. Optimální batch velikost
        optimal_batch_size = self._calculate_optimal_batch_size(uncached_texts, analysis_type)

        # 4. Batch zpracování
        results = [None] * len(texts)

        # Vložení cached výsledků
        for i, cached_result in cache_indices.items():
            results[i] = cached_result

        # Zpracování uncached v batches
        uncached_indices, uncached_text_list = zip(*uncached_texts) if uncached_texts else ([], [])

        for i in range(0, len(uncached_text_list), optimal_batch_size):
            batch_texts = list(uncached_text_list[i:i + optimal_batch_size])
            batch_indices = list(uncached_indices[i:i + optimal_batch_size])

            try:
                batch_results = await gemini_manager.analyze_batch_efficiently(batch_texts, analysis_type)

                # Uložení do cache a výsledků
                for idx, (original_idx, text) in enumerate(zip(batch_indices, batch_texts)):
                    if idx < len(batch_results):
                        result = batch_results[idx]
                        results[original_idx] = result

                        # Cache
                        cache_key = self._generate_cache_key(text, analysis_type)
                        self.batch_cache[cache_key] = result

                # Krátká pauza mezi batches
                await asyncio.sleep(0.5)

            except Exception as e:
                self.logger.error(f"Chyba při batch zpracování: {e}")
                # Fallback jednotlivé zpracování
                for original_idx, text in zip(batch_indices, batch_texts):
                    results[original_idx] = {"error": str(e), "text": text[:100]}

        self.logger.info(f"📊 Zpracováno {len(uncached_texts)} nových textů, {len(cache_indices)} z cache")

        return results

    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Kontrola platnosti cache na základě expiračního času.
        """
        if cache_key not in self.batch_cache:
            return False

        cache_entry = self.batch_cache[cache_key]
        return datetime.now() - cache_entry["timestamp"] <= self.cache_expiry

    def _generate_cache_key(self, text: str, analysis_type: str) -> str:
        """
        Generování cache klíče na základě textu a typu analýzy.
        """
        import hashlib

        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        return f"{analysis_type}_{text_hash}"

    def _check_cache(self, texts: List[str], analysis_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Kontrola cache pro texty s expiračním časem.
        """
        cached = {}

        for text in texts:
            cache_key = self._generate_cache_key(text, analysis_type)
            if cache_key in self.batch_cache and self._is_cache_valid(cache_key):
                cached[cache_key] = self.batch_cache[cache_key]["data"]

        return cached

    def clear_cache(self):
        """
        Vyčištění cache.
        """
        self.batch_cache.clear()
        self.logger.info("🧹 Cache vyčištěna")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Statistiky cache.
        """
        valid_cache = {
            key: entry for key, entry in self.batch_cache.items()
            if self._is_cache_valid(key)
        }

        return {
            "cache_size": len(valid_cache),
            "memory_usage_mb": len(str(valid_cache).encode()) / (1024 * 1024),
            "cache_keys": list(valid_cache.keys())[:10]  # První 10 klíčů
        }

class AdvancedOptimizationManager:
    """Pokročilé optimalizační techniky pro snížení nákladů na API volání"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.ttl_cache = {}
        self.lru_cache = []
        self.semantic_cache = {}

    def semantic_fingerprinting(self, content: str) -> str:
        """Vytvoření unikátního fingerprintu pro obsah"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def hybrid_caching_strategy(self, key: str, value: Any, ttl: int = 3600):
        """Kombinace TTL, LRU a semantic caching"""
        self.ttl_cache[key] = (value, time.time() + ttl)
        self.lru_cache.append(key)
        self.semantic_cache[key] = value

        # Udržení LRU cache na max. velikosti
        if len(self.lru_cache) > 100:
            oldest_key = self.lru_cache.pop(0)
            self.ttl_cache.pop(oldest_key, None)
            self.semantic_cache.pop(oldest_key, None)

    def get_cached_value(self, key: str) -> Any:
        """Získání hodnoty z cache"""
        if key in self.ttl_cache:
            value, expiry = self.ttl_cache[key]
            if time.time() < expiry:
                return value
            else:
                self.ttl_cache.pop(key, None)

        return self.semantic_cache.get(key)

    def log_cache_statistics(self):
        """Logování statistik cache"""
        self.logger.info(f"TTL Cache Size: {len(self.ttl_cache)}")
        self.logger.info(f"LRU Cache Size: {len(self.lru_cache)}")
        self.logger.info(f"Semantic Cache Size: {len(self.semantic_cache)}")
