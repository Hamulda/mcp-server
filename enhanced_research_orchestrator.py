"""
Enhanced Intelligent Research Orchestrator - Pokročilý AI-řízený orchestrátor
Implementuje kreativní optimalizace: AI-powered výběr zdrojů, režimy výzkumu, prediktivní cache
"""

import asyncio
import json
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

try:
    from unified_cache_system import get_unified_cache
    from local_ai_adapter import M1OptimizedOllamaClient, quick_ai_query
    from advanced_source_aggregator import AdvancedSourceAggregator
    from quality_assessment_system import QualityAssessmentSystem
    from biohacking_research_engine import BiohackingResearchEngine, BiohackingResearchRequest
    from peptide_prompts import PEPTIDE_RESEARCH_PROMPTS, BIOHACKING_PROMPTS
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ResearchMode:
    """Definice režimů výzkumu s různými strategiemi"""
    name: str
    description: str
    max_sources: int
    depth_level: int  # 1-5
    evidence_threshold: float  # 0.0-1.0
    time_budget_seconds: int
    ai_analysis_depth: str  # "basic", "detailed", "expert"
    include_community: bool
    predictive_preload: bool

@dataclass
class UserResearchProfile:
    """Rozšířený uživatelský profil s AI learning"""
    user_id: str
    research_interests: List[str] = field(default_factory=list)
    preferred_sources: List[str] = field(default_factory=list)
    evidence_preference: str = "high"  # high, medium, mixed
    complexity_level: str = "intermediate"  # beginner, intermediate, expert
    research_patterns: Dict[str, Any] = field(default_factory=dict)
    success_metrics: Dict[str, float] = field(default_factory=dict)
    query_embeddings: List[List[float]] = field(default_factory=list)
    learning_weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class PredictiveInsight:
    """AI-generovaný insight pro prediktivní analýzu"""
    insight_type: str
    confidence: float
    prediction: str
    suggested_queries: List[str]
    related_compounds: List[str]
    risk_factors: List[str]

class EnhancedIntelligentOrchestrator:
    """
    Pokročilý AI orchestrátor s kreativními optimalizacemi
    """

    def __init__(self):
        # Core components
        self.ai_client = None
        self.source_aggregator = None
        self.quality_system = None
        self.research_engine = None
        self.cache = get_unified_cache()

        # Research modes
        self.research_modes = self._initialize_research_modes()

        # User profiles and learning
        self.user_profiles = {}
        self.query_patterns = {}
        self.source_performance = {}

        # Predictive systems
        self.prediction_cache = {}
        self.preload_queue = asyncio.Queue()

        # Performance optimization
        self.adaptive_weights = {
            "source_reliability": 0.4,
            "response_time": 0.2,
            "content_quality": 0.3,
            "user_preference": 0.1
        }

        # Data paths
        self.data_dir = Path("data/enhanced_orchestrator")
        self.data_dir.mkdir(exist_ok=True)

    def _initialize_research_modes(self) -> Dict[str, ResearchMode]:
        """Inicializace přednastavených režimů výzkumu"""
        return {
            "quick_overview": ResearchMode(
                name="Quick Overview",
                description="Rychlý přehled s nejdůležitějšími informacemi",
                max_sources=3,
                depth_level=2,
                evidence_threshold=0.7,
                time_budget_seconds=30,
                ai_analysis_depth="basic",
                include_community=False,
                predictive_preload=True
            ),
            "balanced_research": ResearchMode(
                name="Balanced Research",
                description="Vyvážený výzkum s dobrým poměrem rychlosti a kvality",
                max_sources=5,
                depth_level=3,
                evidence_threshold=0.6,
                time_budget_seconds=60,
                ai_analysis_depth="detailed",
                include_community=True,
                predictive_preload=True
            ),
            "deep_analysis": ResearchMode(
                name="Deep Analysis",
                description="Hloubková analýza s maximální kvalitou výsledků",
                max_sources=8,
                depth_level=5,
                evidence_threshold=0.5,
                time_budget_seconds=120,
                ai_analysis_depth="expert",
                include_community=True,
                predictive_preload=False
            ),
            "fact_verification": ResearchMode(
                name="Fact Verification",
                description="Ověření faktů s vysokými nároky na důkazy",
                max_sources=6,
                depth_level=4,
                evidence_threshold=0.8,
                time_budget_seconds=90,
                ai_analysis_depth="expert",
                include_community=False,
                predictive_preload=False
            ),
            "safety_focused": ResearchMode(
                name="Safety Focused",
                description="Zaměření na bezpečnost a vedlejší účinky",
                max_sources=7,
                depth_level=4,
                evidence_threshold=0.7,
                time_budget_seconds=75,
                ai_analysis_depth="detailed",
                include_community=True,
                predictive_preload=True
            )
        }

    async def __aenter__(self):
        """Inicializace všech komponent"""
        try:
            if DEPS_AVAILABLE:
                self.ai_client = M1OptimizedOllamaClient()
                await self.ai_client.__aenter__()

                self.source_aggregator = AdvancedSourceAggregator()
                await self.source_aggregator.__aenter__()

                self.quality_system = QualityAssessmentSystem()
                await self.quality_system.__aenter__()

                self.research_engine = BiohackingResearchEngine()
                await self.research_engine.__aenter__()

            await self._load_user_profiles()
            await self._load_learning_data()

            # Start background tasks
            asyncio.create_task(self._predictive_preloader())

            logger.info("✅ Enhanced Intelligent Orchestrator initialized")

        except Exception as e:
            logger.error(f"❌ Orchestrator initialization failed: {e}")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup"""
        components = [self.ai_client, self.source_aggregator, self.quality_system, self.research_engine]

        for component in components:
            if component:
                await component.__aexit__(exc_type, exc_val, exc_tb)

        await self._save_user_profiles()
        await self._save_learning_data()

    async def intelligent_research(
        self,
        query: str,
        user_id: str = "default",
        mode: str = "balanced_research",
        custom_sources: Optional[List[str]] = None,
        generate_insights: bool = True
    ) -> Dict[str, Any]:
        """
        Hlavní metoda pro inteligentní výzkum s AI optimalizacemi
        """
        start_time = time.time()

        # Get or create user profile
        user_profile = await self._get_user_profile(user_id)
        research_mode = self.research_modes.get(mode, self.research_modes["balanced_research"])

        logger.info(f"🧠 Starting intelligent research in {mode} mode: '{query}'")

        try:
            # 1. AI-powered query analysis and enhancement
            enhanced_query, query_insights = await self._ai_analyze_query(query, user_profile)

            # 2. Intelligent source selection
            optimal_sources = await self._ai_select_sources(
                enhanced_query, research_mode, user_profile, custom_sources
            )

            # 3. Predictive cache check
            cached_result = await self._check_predictive_cache(enhanced_query, user_profile)
            if cached_result:
                logger.info("🎯 Predictive cache hit")
                return cached_result

            # 4. Parallel research execution with monitoring
            research_results = await self._execute_parallel_research(
                enhanced_query, optimal_sources, research_mode, user_profile
            )

            # 5. AI-powered result synthesis
            synthesized_results = await self._ai_synthesize_results(
                research_results, query_insights, research_mode, user_profile
            )

            # 6. Quality assessment and scoring
            quality_assessment = await self._comprehensive_quality_assessment(
                synthesized_results, enhanced_query
            )

            # 7. Generate predictive insights
            insights = []
            if generate_insights and self.ai_client:
                insights = await self._generate_predictive_insights(
                    synthesized_results, user_profile
                )

            # 8. Update learning systems
            await self._update_learning_systems(
                user_profile, query, research_results, quality_assessment
            )

            # 9. Schedule predictive preloading
            await self._schedule_predictive_preload(insights, user_profile)

            # Compile final result
            final_result = {
                "original_query": query,
                "enhanced_query": enhanced_query,
                "research_mode": mode,
                "user_id": user_id,
                "execution_time": time.time() - start_time,
                "sources_used": optimal_sources,
                "research_results": synthesized_results,
                "quality_assessment": quality_assessment,
                "predictive_insights": insights,
                "query_insights": query_insights,
                "personalization_applied": True,
                "ai_enhanced": True,
                "success": True
            }

            # Cache result for future predictive use
            await self._cache_result_for_prediction(final_result, user_profile)

            logger.info(f"✅ Intelligent research completed in {final_result['execution_time']:.2f}s")
            return final_result

        except Exception as e:
            logger.error(f"❌ Intelligent research failed: {e}")
            return {
                "original_query": query,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "success": False
            }

    async def _ai_analyze_query(
        self,
        query: str,
        user_profile: UserResearchProfile
    ) -> Tuple[str, Dict[str, Any]]:
        """AI analýza dotazu s kontextovým vylepšením"""

        if not self.ai_client:
            return query, {"analysis": "AI not available"}

        # Prepare context from user profile
        user_context = {
            "interests": user_profile.research_interests[:5],
            "complexity_level": user_profile.complexity_level,
            "evidence_preference": user_profile.evidence_preference
        }

        analysis_prompt = f"""Analyze this research query and enhance it for optimal results:

Query: "{query}"
User Context: {json.dumps(user_context)}

Provide:
1. Query intent classification (safety, dosage, mechanisms, benefits, interactions)
2. Compound/substance identification 
3. Enhanced query with relevant synonyms and terms
4. Research complexity assessment (1-5)
5. Suggested research areas
6. Potential safety considerations

Format as JSON."""

        try:
            analysis_response = await self.ai_client.generate_response(analysis_prompt)
            query_insights = json.loads(analysis_response)

            # Extract enhanced query
            enhanced_query = query_insights.get("enhanced_query", query)

            return enhanced_query, query_insights

        except Exception as e:
            logger.warning(f"AI query analysis failed: {e}")
            return query, {"analysis_error": str(e)}

    async def _ai_select_sources(
        self,
        query: str,
        mode: ResearchMode,
        user_profile: UserResearchProfile,
        custom_sources: Optional[List[str]] = None
    ) -> List[str]:
        """AI-powered inteligentní výběr zdrojů"""

        if custom_sources:
            return custom_sources[:mode.max_sources]

        # Available sources with metadata
        available_sources = {
            "pubmed": {"reliability": 0.95, "academic": True, "speed": 0.6, "peptide_focus": 0.8},
            "clinical_trials": {"reliability": 0.9, "academic": True, "speed": 0.7, "peptide_focus": 0.9},
            "examine": {"reliability": 0.85, "academic": False, "speed": 0.9, "peptide_focus": 0.7},
            "google_scholar": {"reliability": 0.8, "academic": True, "speed": 0.5, "peptide_focus": 0.6},
            "reddit_peptides": {"reliability": 0.4, "academic": False, "speed": 0.95, "peptide_focus": 0.95},
            "reddit_nootropics": {"reliability": 0.4, "academic": False, "speed": 0.95, "peptide_focus": 0.3},
            "selfhacked": {"reliability": 0.6, "academic": False, "speed": 0.8, "peptide_focus": 0.6},
            "longecity": {"reliability": 0.5, "academic": False, "speed": 0.7, "peptide_focus": 0.4}
        }

        # Score sources based on multiple factors
        source_scores = {}

        for source, metadata in available_sources.items():
            score = 0.0

            # Base reliability weight
            score += metadata["reliability"] * self.adaptive_weights["source_reliability"]

            # User preference weight
            if source in user_profile.preferred_sources:
                score += 0.3 * self.adaptive_weights["user_preference"]

            # Evidence preference matching
            if user_profile.evidence_preference == "high" and metadata["academic"]:
                score += 0.2
            elif user_profile.evidence_preference == "mixed":
                score += 0.1

            # Mode-specific adjustments
            if mode.include_community or metadata["academic"]:
                score += 0.1

            if mode.time_budget_seconds < 60:  # Quick mode prefers fast sources
                score += metadata["speed"] * 0.2

            # Query-specific relevance (simple keyword matching for now)
            if any(keyword in query.lower() for keyword in ["peptide", "bpc", "tb-", "ghrp"]):
                score += metadata["peptide_focus"] * 0.3

            # Historical performance
            if source in self.source_performance:
                perf = self.source_performance[source]
                score += (perf.get("success_rate", 0.5) - 0.5) * 0.2

            source_scores[source] = score

        # Select top sources up to mode limit
        selected_sources = sorted(source_scores.items(), key=lambda x: x[1], reverse=True)
        final_sources = [source for source, score in selected_sources[:mode.max_sources]]

        logger.info(f"🎯 AI selected sources: {final_sources}")
        return final_sources

    async def _execute_parallel_research(
        self,
        query: str,
        sources: List[str],
        mode: ResearchMode,
        user_profile: UserResearchProfile
    ) -> Dict[str, Any]:
        """Paralelní výzkum s monitorováním výkonu"""

        start_time = time.time()

        # Execute research tasks in parallel
        tasks = []

        # Source aggregation
        if self.source_aggregator:
            task = self.source_aggregator.multi_source_search(
                query=query,
                sources=sources,
                max_results_per_source=mode.depth_level,
                evidence_level=user_profile.evidence_preference
            )
            tasks.append(("source_aggregation", task))

        # Specialized research engine
        if self.research_engine:
            request = BiohackingResearchRequest(
                compound=query,
                research_type="comprehensive" if mode.depth_level >= 4 else "balanced",
                evidence_level=user_profile.evidence_preference
            )
            task = self.research_engine.research_compound(request)
            tasks.append(("specialized_research", task))

        # Execute with timeout
        results = {}

        for task_name, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=mode.time_budget_seconds)
                results[task_name] = result

                # Update source performance metrics
                if task_name == "source_aggregation" and isinstance(result, dict):
                    await self._update_source_performance(result, time.time() - start_time)

            except asyncio.TimeoutError:
                logger.warning(f"Task {task_name} timed out after {mode.time_budget_seconds}s")
                results[task_name] = {"error": "timeout"}
            except Exception as e:
                logger.warning(f"Task {task_name} failed: {e}")
                results[task_name] = {"error": str(e)}

        return results

    async def _ai_synthesize_results(
        self,
        research_results: Dict[str, Any],
        query_insights: Dict[str, Any],
        mode: ResearchMode,
        user_profile: UserResearchProfile
    ) -> Dict[str, Any]:
        """AI syntéza výsledků s personalizací"""

        if not self.ai_client:
            return research_results

        # Prepare synthesis context
        synthesis_context = {
            "user_complexity": user_profile.complexity_level,
            "research_mode": mode.name,
            "query_intent": query_insights.get("intent", "general"),
            "evidence_preference": user_profile.evidence_preference
        }

        synthesis_prompt = f"""Synthesize research results for a {user_profile.complexity_level} user:

Research Results: {json.dumps(research_results, indent=2)[:3000]}
Query Insights: {json.dumps(query_insights)}
Context: {json.dumps(synthesis_context)}

Create a comprehensive synthesis including:
1. Executive summary
2. Key findings with evidence levels
3. Safety considerations
4. Dosing information (if applicable)
5. Quality assessment of sources
6. Practical recommendations
7. Areas needing more research

Adapt depth and language to {user_profile.complexity_level} level."""

        try:
            synthesis = await self.ai_client.generate_response(synthesis_prompt)

            return {
                "ai_synthesis": synthesis,
                "raw_results": research_results,
                "synthesis_context": synthesis_context,
                "personalized": True
            }

        except Exception as e:
            logger.warning(f"AI synthesis failed: {e}")
            return {
                "raw_results": research_results,
                "synthesis_error": str(e),
                "personalized": False
            }

    async def _comprehensive_quality_assessment(
        self,
        results: Dict[str, Any],
        query: str
    ) -> Dict[str, Any]:
        """Komplexní hodnocení kvality s AI analýzou"""

        if not self.quality_system:
            return {"quality_score": 5.0, "assessment": "Quality system not available"}

        try:
            assessment = await self.quality_system.assess_research_quality(
                results.get("raw_results", {}), query
            )
            return assessment

        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return {"error": str(e), "quality_score": 0.0}

    async def _generate_predictive_insights(
        self,
        results: Dict[str, Any],
        user_profile: UserResearchProfile
    ) -> List[PredictiveInsight]:
        """Generování prediktivních insights pomocí AI"""

        if not self.ai_client:
            return []

        insight_prompt = f"""Generate predictive insights based on this research:

Results: {json.dumps(results, indent=2)[:2000]}
User Interests: {user_profile.research_interests}

Generate 3-5 insights predicting:
1. Related compounds user might research
2. Potential safety concerns to investigate
3. Synergistic combinations worth exploring
4. Follow-up research questions
5. Emerging trends in this area

Format as JSON array with: type, confidence, prediction, suggested_queries, related_compounds, risk_factors"""

        try:
            insights_response = await self.ai_client.generate_response(insight_prompt)
            insights_data = json.loads(insights_response)

            insights = []
            for insight_data in insights_data:
                insight = PredictiveInsight(
                    insight_type=insight_data.get("type", "general"),
                    confidence=insight_data.get("confidence", 0.5),
                    prediction=insight_data.get("prediction", ""),
                    suggested_queries=insight_data.get("suggested_queries", []),
                    related_compounds=insight_data.get("related_compounds", []),
                    risk_factors=insight_data.get("risk_factors", [])
                )
                insights.append(insight)

            return insights

        except Exception as e:
            logger.warning(f"Predictive insights generation failed: {e}")
            return []

    async def _predictive_preloader(self):
        """Background task pro prediktivní preloading"""

        while True:
            try:
                # Process preload queue
                if not self.preload_queue.empty():
                    preload_item = await self.preload_queue.get()
                    await self._execute_preload(preload_item)

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.warning(f"Predictive preloader error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _get_user_profile(self, user_id: str) -> UserResearchProfile:
        """Získání nebo vytvoření uživatelského profilu"""

        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserResearchProfile(user_id=user_id)

        return self.user_profiles[user_id]

    async def _update_learning_systems(
        self,
        user_profile: UserResearchProfile,
        query: str,
        results: Dict[str, Any],
        quality: Dict[str, Any]
    ):
        """Aktualizace learning systémů"""

        # Update user interests based on successful queries
        quality_score = quality.get("overall_quality", 0.0)

        if quality_score > 7.0:  # High quality result
            # Extract keywords from successful query
            keywords = query.lower().split()
            for keyword in keywords:
                if len(keyword) > 3:  # Ignore short words
                    if keyword not in user_profile.research_interests:
                        user_profile.research_interests.append(keyword)

        # Update success metrics
        user_profile.success_metrics[query[:50]] = quality_score

        # Keep only recent interests (last 50)
        user_profile.research_interests = user_profile.research_interests[-50:]

    async def _load_user_profiles(self):
        """Načtení uživatelských profilů"""
        profiles_file = self.data_dir / "user_profiles.json"

        if profiles_file.exists():
            try:
                with open(profiles_file) as f:
                    data = json.load(f)
                    for user_id, profile_data in data.items():
                        self.user_profiles[user_id] = UserResearchProfile(**profile_data)
            except Exception as e:
                logger.warning(f"Failed to load user profiles: {e}")

    async def _save_user_profiles(self):
        """Uložení uživatelských profilů"""
        profiles_file = self.data_dir / "user_profiles.json"

        try:
            serializable_profiles = {}
            for user_id, profile in self.user_profiles.items():
                serializable_profiles[user_id] = {
                    "user_id": profile.user_id,
                    "research_interests": profile.research_interests,
                    "preferred_sources": profile.preferred_sources,
                    "evidence_preference": profile.evidence_preference,
                    "complexity_level": profile.complexity_level,
                    "success_metrics": profile.success_metrics
                }

            with open(profiles_file, 'w') as f:
                json.dump(serializable_profiles, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save user profiles: {e}")

    async def _load_learning_data(self):
        """Načtení learning dat"""
        pass  # Placeholder for additional learning data

    async def _save_learning_data(self):
        """Uložení learning dat"""
        pass  # Placeholder for additional learning data

    async def _check_predictive_cache(self, query: str, user_profile: UserResearchProfile) -> Optional[Dict]:
        """Kontrola prediktivní cache"""
        cache_key = f"predictive_{user_profile.user_id}_{hashlib.md5(query.encode()).hexdigest()}"
        return await self.cache.get(cache_key)

    async def _cache_result_for_prediction(self, result: Dict[str, Any], user_profile: UserResearchProfile):
        """Cache výsledku pro prediktivní použití"""
        cache_key = f"predictive_{user_profile.user_id}_{hashlib.md5(result['enhanced_query'].encode()).hexdigest()}"
        await self.cache.set(cache_key, result, ttl=3600)  # 1 hour

    async def _schedule_predictive_preload(self, insights: List[PredictiveInsight], user_profile: UserResearchProfile):
        """Naplánování prediktivního preloadingu"""
        for insight in insights:
            for query in insight.suggested_queries:
                preload_item = {
                    "user_id": user_profile.user_id,
                    "query": query,
                    "priority": insight.confidence,
                    "scheduled_at": time.time()
                }
                await self.preload_queue.put(preload_item)

    async def _execute_preload(self, preload_item: Dict):
        """Provedení prediktivního preloadu"""
        try:
            # Execute lightweight research for preloading
            result = await self.intelligent_research(
                preload_item["query"],
                preload_item["user_id"],
                mode="quick_overview",
                generate_insights=False
            )
            logger.info(f"📦 Preloaded: {preload_item['query'][:50]}")
        except Exception as e:
            logger.warning(f"Preload failed for {preload_item['query']}: {e}")

    async def _update_source_performance(self, results: Dict, execution_time: float):
        """Aktualizace výkonnostních metrik zdrojů"""
        for source, source_results in results.items():
            if source not in self.source_performance:
                self.source_performance[source] = {
                    "success_rate": 0.0,
                    "avg_response_time": 0.0,
                    "total_requests": 0
                }

            perf = self.source_performance[source]
            perf["total_requests"] += 1

            # Update success rate
            success = bool(source_results and not isinstance(source_results, dict) or "error" not in source_results)
            perf["success_rate"] = (perf["success_rate"] * (perf["total_requests"] - 1) + (1.0 if success else 0.0)) / perf["total_requests"]

            # Update average response time
            perf["avg_response_time"] = (perf["avg_response_time"] * (perf["total_requests"] - 1) + execution_time) / perf["total_requests"]

# Export
__all__ = [
    'EnhancedIntelligentOrchestrator',
    'ResearchMode',
    'UserResearchProfile',
    'PredictiveInsight'
]
