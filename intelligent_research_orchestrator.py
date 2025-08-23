#!/usr/bin/env python3
"""
Intelligent Research Orchestrator - Pokročilý orchestrátor pro akademický výzkum
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from academic_scraper import AcademicScraper
from cache.unified_cache_system import get_cache_manager, cached
import logging

logger = logging.getLogger(__name__)

class IntelligentResearchOrchestrator:
    """Inteligentní orchestrátor pro koordinaci výzkumných aktivit"""

    def __init__(self):
        self.scraper = None
        self.cache_manager = get_cache_manager()
        self.research_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_response_time": 0.0,
            "cache_hit_rate": 0.0
        }

    async def __aenter__(self):
        self.scraper = AcademicScraper()
        await self.scraper.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.scraper:
            await self.scraper.__aexit__(exc_type, exc_val, exc_tb)

    @cached(ttl=3600, key_prefix="research")
    async def comprehensive_research(
        self,
        query: str,
        research_mode: str = "comprehensive",
        evidence_threshold: str = "high",
        include_safety_assessment: bool = True
    ) -> Dict[str, Any]:
        """
        Provede komplexní výzkum s inteligentní analýzou
        """
        start_time = time.time()
        self.research_stats["total_queries"] += 1

        try:
            # Základní vyhledávání
            search_results = await self.scraper.search_comprehensive(query)

            # Analýza a hodnocení výsledků
            analyzed_results = await self._analyze_results(search_results, evidence_threshold)

            # Safety assessment pokud je požadován
            safety_info = None
            if include_safety_assessment:
                safety_info = await self._assess_safety(query, analyzed_results)

            # Generování doporučení
            recommendations = await self._generate_recommendations(
                query, analyzed_results, research_mode
            )

            response_time = time.time() - start_time
            self._update_stats(response_time, True)

            return {
                "query": query,
                "research_mode": research_mode,
                "evidence_threshold": evidence_threshold,
                "results": analyzed_results,
                "safety_assessment": safety_info,
                "recommendations": recommendations,
                "metadata": {
                    "response_time": round(response_time, 2),
                    "total_sources": len(analyzed_results.get("all_results", [])),
                    "high_confidence_sources": len([
                        r for r in analyzed_results.get("all_results", [])
                        if r.get("confidence", 0) > 0.8
                    ]),
                    "timestamp": time.time()
                },
                "success": True
            }

        except Exception as e:
            self._update_stats(time.time() - start_time, False)
            logger.error(f"Research failed for query '{query}': {e}")

            return {
                "query": query,
                "error": str(e),
                "success": False,
                "timestamp": time.time()
            }

    async def _analyze_results(self, search_results: Dict[str, Any], evidence_threshold: str) -> Dict[str, Any]:
        """Analyzuje a hodnotí kvalitu výsledků"""
        all_results = []

        # Kombinuj výsledky ze všech zdrojů
        for source_name, results in search_results.items():
            if source_name in ['wikipedia', 'pubmed'] and isinstance(results, list):
                for result in results:
                    result['source_type'] = source_name
                    all_results.append(result)

        # Filtrování podle úrovně důkazů
        confidence_threshold = {
            "high": 0.8,
            "medium": 0.6,
            "all": 0.0
        }.get(evidence_threshold, 0.6)

        filtered_results = [
            r for r in all_results
            if r.get("confidence", 0) >= confidence_threshold
        ]

        # Ranking podle relevance a kvality
        ranked_results = sorted(
            filtered_results,
            key=lambda x: (x.get("confidence", 0), x.get("source", "") == "pubmed"),
            reverse=True
        )

        return {
            "all_results": ranked_results,
            "filtered_count": len(filtered_results),
            "total_count": len(all_results),
            "evidence_threshold": evidence_threshold,
            "top_sources": ranked_results[:5] if ranked_results else []
        }

    async def _assess_safety(self, query: str, analyzed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Provede základní bezpečnostní hodnocení"""
        # Jednoduchá heuristika pro detekci bezpečnostních klíčových slov
        safety_keywords = [
            "side effects", "adverse", "toxicity", "contraindication",
            "warning", "caution", "dose", "dosage", "safety"
        ]

        safety_mentions = 0
        safety_sources = []

        for result in analyzed_results.get("all_results", []):
            content = f"{result.get('title', '')} {result.get('snippet', '')}"
            content_lower = content.lower()

            for keyword in safety_keywords:
                if keyword in content_lower:
                    safety_mentions += 1
                    safety_sources.append(result)
                    break

        risk_level = "unknown"
        if safety_mentions >= 3:
            risk_level = "high_safety_data"
        elif safety_mentions >= 1:
            risk_level = "some_safety_data"
        else:
            risk_level = "limited_safety_data"

        return {
            "risk_level": risk_level,
            "safety_mentions": safety_mentions,
            "safety_sources": safety_sources[:3],  # Top 3 safety-related sources
            "recommendation": self._get_safety_recommendation(risk_level),
            "disclaimer": "This is not medical advice. Consult healthcare professionals."
        }

    def _get_safety_recommendation(self, risk_level: str) -> str:
        """Generuje bezpečnostní doporučení"""
        recommendations = {
            "high_safety_data": "Extensive safety data available. Review all sources carefully.",
            "some_safety_data": "Limited safety data found. Exercise caution and consult experts.",
            "limited_safety_data": "Minimal safety information available. Proceed with extreme caution."
        }
        return recommendations.get(risk_level, "Safety data not assessed.")

    async def _generate_recommendations(
        self,
        query: str,
        analyzed_results: Dict[str, Any],
        research_mode: str
    ) -> Dict[str, Any]:
        """Generuje inteligentní doporučení na základě výsledků"""
        top_sources = analyzed_results.get("top_sources", [])

        recommendations = {
            "further_research": [],
            "key_papers": [],
            "expert_consultation": False,
            "confidence_level": "medium"
        }

        # Doporučení pro další výzkum
        if len(top_sources) < 3:
            recommendations["further_research"].append(
                "Consider broader search terms or alternative databases"
            )

        # Identifikace klíčových článků
        pubmed_sources = [s for s in top_sources if s.get("source") == "pubmed"]
        if pubmed_sources:
            recommendations["key_papers"] = pubmed_sources[:3]
            recommendations["confidence_level"] = "high"

        # Doporučení konzultace s expertem
        if research_mode in ["safety", "dosage"] or "limited_safety_data" in str(analyzed_results):
            recommendations["expert_consultation"] = True

        return recommendations

    def _update_stats(self, response_time: float, success: bool):
        """Aktualizuje statistiky výkonu"""
        if success:
            self.research_stats["successful_queries"] += 1
        else:
            self.research_stats["failed_queries"] += 1

        # Aktualizuj průměrný čas odpovědi
        total_queries = self.research_stats["total_queries"]
        current_avg = self.research_stats["avg_response_time"]
        self.research_stats["avg_response_time"] = (
            (current_avg * (total_queries - 1) + response_time) / total_queries
        )

    def get_stats(self) -> Dict[str, Any]:
        """Vrací statistiky výkonu"""
        return self.research_stats.copy()
