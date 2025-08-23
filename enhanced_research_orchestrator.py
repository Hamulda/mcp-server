"""
Enhanced Intelligent Research Orchestrator - Pokročilý AI-řízený orchestrátor
Implementuje kreativní optimalizace: AI-powered výběr zdrojů, režimy výzkumu, prediktivní cache
CONSOLIDATED - sloučeno s intelligent_research_orchestrator.py
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
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
    # Define fallback classes to prevent import errors
    def get_unified_cache():
        return None

    class M1OptimizedOllamaClient:
        pass

    class AdvancedSourceAggregator:
        pass

    class QualityAssessmentSystem:
        pass

    class BiohackingResearchEngine:
        pass

    class BiohackingResearchRequest:
        pass

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """Uživatelský profil pro personalizaci (sloučeno z intelligent_research_orchestrator)"""
    user_id: str
    research_interests: List[str] = field(default_factory=list)
    experience_level: str = "beginner"  # beginner, intermediate, advanced, expert
    safety_preference: str = "conservative"  # conservative, moderate, aggressive
    preferred_evidence_level: str = "high"  # high, medium, all
    target_goals: List[str] = field(default_factory=list)
    compound_history: List[str] = field(default_factory=list)
    successful_protocols: List[Dict] = field(default_factory=list)
    adverse_reactions: List[Dict] = field(default_factory=list)
    preferred_sources: List[str] = field(default_factory=list)
    learning_preferences: Dict[str, Any] = field(default_factory=dict)
    # Enhanced features
    research_patterns: Dict[str, Any] = field(default_factory=dict)
    success_metrics: Dict[str, float] = field(default_factory=dict)
    query_embeddings: List[List[float]] = field(default_factory=list)
    learning_weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class ResearchSession:
    """Session pro sledování výzkumné aktivity (sloučeno z intelligent_research_orchestrator)"""
    session_id: str
    user_id: str
    start_time: datetime
    queries: List[str] = field(default_factory=list)
    compounds_researched: List[str] = field(default_factory=list)
    time_spent: float = 0.0
    results_quality: float = 0.0
    user_satisfaction: Optional[float] = None
    learned_insights: List[str] = field(default_factory=list)

@dataclass
class PerformanceMetrics:
    """Metriky výkonu systému (sloučeno z intelligent_research_orchestrator)"""
    query_response_time: float
    source_success_rate: float
    ai_processing_time: float
    cache_hit_rate: float
    memory_usage_mb: float
    user_engagement_score: float
    research_depth_score: float
    safety_compliance_score: float

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
class PredictiveInsight:
    """AI-generovaný insight pro prediktivní analýzu"""
    insight_type: str
    confidence: float
    prediction: str
    suggested_queries: List[str]
    related_compounds: List[str]
    risk_factors: List[str]

class ConsolidatedResearchOrchestrator:
    """
    Pokročilý AI orchestrátor s kreativními optimalizacemi
    CONSOLIDATED - sloučeno enhanced + intelligent research orchestrátory
    """

    def __init__(self):
        # Core components
        self.ai_client = None
        self.source_aggregator = None
        self.quality_system = None
        self.research_engine = None
        self.cache = get_unified_cache()

        # User management (z intelligent_research_orchestrator)
        self.user_profiles = {}
        self.research_sessions = {}
        self.performance_history = []

        # Research modes (z enhanced_research_orchestrator)
        self.research_modes = self._initialize_research_modes()

        # Learning and optimization (sloučeno z obou)
        self.query_patterns = {}
        self.source_performance = {}
        self.query_embeddings = {}
        self.success_patterns = {}
        self.optimization_rules = {}

        # Predictive systems (z enhanced_research_orchestrator)
        self.prediction_cache = {}
        self.preload_queue = asyncio.Queue()

        # Performance optimization (sloučeno z obou)
        self.adaptive_weights = {
            "source_reliability": 0.4,
            "response_time": 0.2,
            "content_quality": 0.3,
            "user_preference": 0.1
        }
        self.metrics_buffer = []
        self.optimization_threshold = 10  # sessions before optimization

        # Data paths
        self.data_dir = Path("data/enhanced_orchestrator")
        self.data_dir.mkdir(exist_ok=True)

        # Phase 1 optimization components
        self.async_optimizer = None
        self.token_optimizer = None
        self.distributed_cache = None
        self.connection_pool_manager = None
        self._initialize_phase1_optimizations()

    def _initialize_research_modes(self) -> Dict[str, ResearchMode]:
        """Inicializace přednastavených režimů výzkumu"""
        return {
            "quick_overview": ResearchMode(
                name="Quick Overview",
                description="Rychlý přehled s nejdůle��itějšími informacemi",
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
        """Inicializace všech komponent s Phase 1 optimalizacemi"""
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

            # Phase 1 optimalizace - inicializace async optimizeru
            try:
                from async_performance_optimizer import get_async_optimizer
                from token_optimization_system import get_token_optimizer

                self.async_optimizer = get_async_optimizer()
                await self.async_optimizer.__aenter__()

                self.token_optimizer = get_token_optimizer()

                logger.info("✅ Phase 1 optimalizace aktivovány")

            except ImportError:
                logger.warning("Phase 1 optimizers not available, using fallback")
                self.async_optimizer = None
                self.token_optimizer = None

            await self._load_user_profiles()
            await self._load_learning_data()

            # Start background tasks
            asyncio.create_task(self._predictive_preloader())

            logger.info("✅ Enhanced Intelligent Orchestrator initialized with Phase 1 optimizations")

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
        research_type: str = "auto"
    ) -> Dict[str, Any]:
        """Inteligentní výzkum s personalizací a optimalizací (sloučeno z intelligent_research_orchestrator)"""

        session_start = time.time()
        session_id = self._generate_session_id()

        # Get or create user profile
        user_profile = await self._get_user_profile(user_id)

        # Create research session
        session = ResearchSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            queries=[query]
        )
        self.research_sessions[session_id] = session

        logger.info(f"🎯 Starting intelligent research for user {user_id}: {query}")

        try:
            # 1. Query analysis and enhancement
            enhanced_query, research_strategy = await self._analyze_and_enhance_query(
                query, user_profile
            )

            # 2. Select optimal research mode
            mode = await self._select_research_mode(enhanced_query, user_profile, research_type)

            # 3. Execute research with selected mode
            results = await self._execute_research_with_mode(enhanced_query, mode, user_profile)

            # 4. Post-process and learn from results
            final_results = await self._post_process_and_learn(results, session, user_profile)

            session.time_spent = time.time() - session_start
            session.results_quality = final_results.get('quality_score', 0.0)

            return final_results

        except Exception as e:
            logger.error(f"❌ Research failed: {e}")
            return {"error": str(e), "session_id": session_id}

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

    async def _get_user_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        return self.user_profiles[user_id]

    async def _analyze_and_enhance_query(
        self,
        query: str,
        user_profile: UserProfile
    ) -> Tuple[str, Any]:
        """Analyze and enhance query for intelligent research"""

        if not self.ai_client:
            return query, {"analysis": "AI not available"}

        # Prepare context from user profile
        user_context = {
            "interests": user_profile.research_interests[:5],
            "complexity_level": user_profile.experience_level,
            "safety_preference": user_profile.safety_preference,
            "preferred_evidence_level": user_profile.preferred_evidence_level
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

    async def _select_research_mode(
        self,
        query: str,
        user_profile: UserProfile,
        research_type: str = "auto"
    ) -> ResearchMode:
        """Select optimal research mode based on query and user profile"""

        if research_type == "auto":
            # Heuristic: longer queries with complex terms may need deep analysis
            if len(query.split()) > 5 or any(term in query.lower() for term in ["mechanism", "pathway", "interaction"]):
                return self.research_modes["deep_analysis"]
            else:
                return self.research_modes["balanced_research"]
        else:
            return self.research_modes.get(research_type, self.research_modes["balanced_research"])

    async def _execute_research_with_mode(
        self,
        query: str,
        mode: ResearchMode,
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Execute research using the selected mode"""

        if mode.name == "Quick Overview":
            return await self._quick_overview_research(query, user_profile)
        elif mode.name == "Balanced Research":
            return await self._balanced_research(query, user_profile)
        elif mode.name == "Deep Analysis":
            return await self._deep_analysis_research(query, user_profile)
        elif mode.name == "Fact Verification":
            return await self._fact_verification_research(query, user_profile)
        elif mode.name == "Safety Focused":
            return await self._safety_focused_research(query, user_profile)
        else:
            logger.warning(f"Unknown research mode: {mode.name}")
            return {"error": "Unknown research mode"}

    async def _quick_overview_research(
        self,
        query: str,
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Quick overview research strategy"""

        # Limit sources and depth
        mode = self.research_modes["quick_overview"]

        # AI-powered query enhancement
        enhanced_query, _ = await self._analyze_and_enhance_query(query, user_profile)

        # Intelligent source selection
        sources = await self._ai_select_sources(enhanced_query, mode, user_profile)

        # Execute research
        results = await self._execute_parallel_research(enhanced_query, sources, mode, user_profile)

        # AI-powered result synthesis
        synthesized_results = await self._ai_synthesize_results(results, query_insights={}, mode=mode, user_profile=user_profile)

        return synthesized_results

    async def _balanced_research(
        self,
        query: str,
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Balanced research strategy"""

        mode = self.research_modes["balanced_research"]
        start_time = time.time()

        # AI-powered query enhancement
        enhanced_query, query_insights = await self._analyze_and_enhance_query(query, user_profile)

        # Intelligent source selection
        sources = await self._ai_select_sources(enhanced_query, mode, user_profile)

        # Predictive cache check
        cached_result = await self._check_predictive_cache(enhanced_query, user_profile)
        if cached_result:
            logger.info("🎯 Predictive cache hit")
            return cached_result

        # Execute research
        results = await self._execute_parallel_research(enhanced_query, sources, mode, user_profile)

        # AI-powered result synthesis
        synthesized_results = await self._ai_synthesize_results(results, query_insights, mode=mode, user_profile=user_profile)

        # Quality assessment
        quality_assessment = await self._comprehensive_quality_assessment(synthesized_results, enhanced_query)

        # Update user profile with new insights
        await self._update_user_profile_with_insights(user_profile, synthesized_results)

        return {
            "original_query": query,
            "enhanced_query": enhanced_query,
            "research_mode": mode.name,
            "user_id": user_profile.user_id,
            "execution_time": time.time() - start_time,
            "sources_used": sources,
            "research_results": synthesized_results,
            "quality_assessment": quality_assessment,
            "query_insights": query_insights,
            "personalization_applied": True,
            "ai_enhanced": True,
            "success": True
        }

    async def _deep_analysis_research(
        self,
        query: str,
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Deep analysis research strategy"""

        mode = self.research_modes["deep_analysis"]
        start_time = time.time()

        # AI-powered query enhancement
        enhanced_query, query_insights = await self._analyze_and_enhance_query(query, user_profile)

        # Intelligent source selection
        sources = await self._ai_select_sources(enhanced_query, mode, user_profile)

        # Predictive cache check
        cached_result = await self._check_predictive_cache(enhanced_query, user_profile)
        if cached_result:
            logger.info("🎯 Predictive cache hit")
            return cached_result

        # Execute research
        results = await self._execute_parallel_research(enhanced_query, sources, mode, user_profile)

        # AI-powered result synthesis
        synthesized_results = await self._ai_synthesize_results(results, query_insights, mode=mode, user_profile=user_profile)

        # Quality assessment
        quality_assessment = await self._comprehensive_quality_assessment(synthesized_results, enhanced_query)

        # Update user profile with new insights
        await self._update_user_profile_with_insights(user_profile, synthesized_results)

        return {
            "original_query": query,
            "enhanced_query": enhanced_query,
            "research_mode": mode.name,
            "user_id": user_profile.user_id,
            "execution_time": time.time() - start_time,
            "sources_used": sources,
            "research_results": synthesized_results,
            "quality_assessment": quality_assessment,
            "query_insights": query_insights,
            "personalization_applied": True,
            "ai_enhanced": True,
            "success": True
        }

    async def _fact_verification_research(
        self,
        query: str,
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Fact verification research strategy"""

        mode = self.research_modes["fact_verification"]
        start_time = time.time()

        # AI-powered query enhancement
        enhanced_query, query_insights = await self._analyze_and_enhance_query(query, user_profile)

        # Intelligent source selection
        sources = await self._ai_select_sources(enhanced_query, mode, user_profile)

        # Predictive cache check
        cached_result = await self._check_predictive_cache(enhanced_query, user_profile)
        if cached_result:
            logger.info("🎯 Predictive cache hit")
            return cached_result

        # Execute research
        results = await self._execute_parallel_research(enhanced_query, sources, mode, user_profile)

        # AI-powered result synthesis
        synthesized_results = await self._ai_synthesize_results(results, query_insights, mode=mode, user_profile=user_profile)

        # Quality assessment
        quality_assessment = await self._comprehensive_quality_assessment(synthesized_results, enhanced_query)

        # Update user profile with new insights
        await self._update_user_profile_with_insights(user_profile, synthesized_results)

        return {
            "original_query": query,
            "enhanced_query": enhanced_query,
            "research_mode": mode.name,
            "user_id": user_profile.user_id,
            "execution_time": time.time() - start_time,
            "sources_used": sources,
            "research_results": synthesized_results,
            "quality_assessment": quality_assessment,
            "query_insights": query_insights,
            "personalization_applied": True,
            "ai_enhanced": True,
            "success": True
        }

    async def _safety_focused_research(
        self,
        query: str,
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Safety focused research strategy"""

        mode = self.research_modes["safety_focused"]
        start_time = time.time()

        # AI-powered query enhancement
        enhanced_query, query_insights = await self._analyze_and_enhance_query(query, user_profile)

        # Intelligent source selection
        sources = await self._ai_select_sources(enhanced_query, mode, user_profile)

        # Predictive cache check
        cached_result = await self._check_predictive_cache(enhanced_query, user_profile)
        if cached_result:
            logger.info("🎯 Predictive cache hit")
            return cached_result

        # Execute research
        results = await self._execute_parallel_research(enhanced_query, sources, mode, user_profile)

        # AI-powered result synthesis
        synthesized_results = await self._ai_synthesize_results(results, query_insights, mode=mode, user_profile=user_profile)

        # Quality assessment
        quality_assessment = await self._comprehensive_quality_assessment(synthesized_results, enhanced_query)

        # Update user profile with new insights
        await self._update_user_profile_with_insights(user_profile, synthesized_results)

        return {
            "original_query": query,
            "enhanced_query": enhanced_query,
            "research_mode": mode.name,
            "user_id": user_profile.user_id,
            "execution_time": time.time() - start_time,
            "sources_used": sources,
            "research_results": synthesized_results,
            "quality_assessment": quality_assessment,
            "query_insights": query_insights,
            "personalization_applied": True,
            "ai_enhanced": True,
            "success": True
        }

    async def _ai_select_sources(
        self,
        query: str,
        mode: ResearchMode,
        user_profile: UserProfile,
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
            if user_profile.preferred_evidence_level == "high" and metadata["academic"]:
                score += 0.2
            elif user_profile.preferred_evidence_level == "mixed":
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
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Paralelní výzkum s Phase 1 optimalizacemi"""

        start_time = time.time()

        # Phase 1 optimalizace - použití async optimizeru pokud je dostupný
        if self.async_optimizer:
            # Vytvoř batch requests pro optimalizované zpracování
            from async_performance_optimizer import BatchRequest

            batch_requests = []
            for i, source in enumerate(sources):
                req = BatchRequest(
                    id=f"research_{source}_{int(time.time())}_{i}",
                    query=query,
                    source=source,
                    priority=1 if mode.name == "Quick Overview" else 2,
                    timeout=mode.time_budget_seconds
                )
                batch_requests.append(req)

            # Využij optimalizované batch processing
            batch_results = await self.async_optimizer.batch_process_requests(batch_requests)

            # Konvertuj batch results zpět na standardní formát
            results = {}
            for req_id, result in batch_results.items():
                if isinstance(result, dict) and "source" in result:
                    source = result["source"]
                    results[f"source_{source}"] = result

            logger.info(f"✅ Phase 1 async optimization použito pro {len(sources)} zdrojů")

        else:
            # Fallback na standardní paralelní zpracování
            tasks = []

            # Source aggregation
            if self.source_aggregator:
                task = self.source_aggregator.multi_source_search(
                    query=query,
                    sources=sources,
                    max_results_per_source=mode.depth_level,
                    evidence_level=user_profile.preferred_evidence_level
                )
                tasks.append(("source_aggregation", task))

            # Specialized research engine
            if self.research_engine:
                request = BiohackingResearchRequest(
                    compound=query,
                    research_type="comprehensive" if mode.depth_level >= 4 else "balanced",
                    evidence_level=user_profile.preferred_evidence_level
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
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """AI syntéza výsledků s personalizací"""

        if not self.ai_client:
            return research_results

        # Prepare synthesis context
        synthesis_context = {
            "user_complexity": user_profile.experience_level,
            "research_mode": mode.name,
            "query_intent": query_insights.get("intent", "general"),
            "evidence_preference": user_profile.preferred_evidence_level
        }

        synthesis_prompt = f"""Synthesize research results for a {user_profile.experience_level} user:

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

Adapt depth and language to {user_profile.experience_level} level."""

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
        user_profile: UserProfile
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

    async def _update_user_profile_with_insights(
        self,
        user_profile: UserProfile,
        results: Dict[str, Any]
    ):
        """Aktualizace uživatelského profilu na základě nových insights"""

        # Extract keywords from successful results
        new_interests = set()
        for result in results.get("research_results", []):
            if isinstance(result, dict) and "keywords" in result:
                new_interests.update(result["keywords"])

        # Update user interests, keeping the most recent 50
        user_profile.research_interests = list(
            dict.fromkeys(list(new_interests) + user_profile.research_interests)
        )[-50:]

        # Update success metrics
        user_profile.success_metrics[results["original_query"][:50]] = results.get("quality_score", 0.0)

    async def _get_user_profile(self, user_id: str) -> UserProfile:
        """Získání nebo vytvoření uživatelského profilu"""

        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)

        return self.user_profiles[user_id]

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

    async def _post_process_and_learn(
        self,
        results: Dict[str, Any],
        session: ResearchSession,
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Post-process results and learn from session"""

        start_time = time.time()

        # Generate predictive insights
        insights = await self._generate_predictive_insights(results, user_profile)

        # Update learning patterns
        await self._update_learning_patterns(session, results, user_profile)

        # Schedule predictive preloading
        await self._schedule_predictive_preloading(insights, user_profile)

        # Prepare final results
        final_results = {
            "original_query": session.queries[0],
            "research_mode": "auto",
            "user_id": user_profile.user_id,
            "session_id": session.session_id,
            "execution_time": time.time() - start_time,
            "research_results": results,
            "predictive_insights": [
                {
                    "type": insight.insight_type,
                    "confidence": insight.confidence,
                    "prediction": insight.prediction,
                    "suggested_queries": insight.suggested_queries,
                    "related_compounds": insight.related_compounds,
                    "risk_factors": insight.risk_factors
                } for insight in insights
            ],
            "personalization_applied": True,
            "ai_enhanced": True,
            "success": True
        }

        return final_results

    async def _update_learning_patterns(
        self,
        session: ResearchSession,
        results: Dict[str, Any],
        user_profile: UserProfile
    ):
        """Update learning patterns from session data"""

        # Extract patterns from this session
        query_pattern = self._extract_query_pattern(session.queries[0])

        # Update pattern frequency
        pattern_key = query_pattern.get("pattern_type", "general")
        if pattern_key not in self.query_patterns:
            self.query_patterns[pattern_key] = {"frequency": 0, "success_rate": 0.0}

        self.query_patterns[pattern_key]["frequency"] += 1

        # Update success metrics
        quality_score = results.get("quality_score", 0.0)
        current_success = self.query_patterns[pattern_key]["success_rate"]
        frequency = self.query_patterns[pattern_key]["frequency"]

        # Weighted average of success rate
        new_success_rate = (current_success * (frequency - 1) + quality_score) / frequency
        self.query_patterns[pattern_key]["success_rate"] = new_success_rate

    def _extract_query_pattern(self, query: str) -> Dict[str, Any]:
        """Extract pattern from query for learning"""

        query_lower = query.lower()

        # Determine pattern type
        if any(word in query_lower for word in ["dosage", "dose", "mcg", "mg"]):
            pattern_type = "dosage_inquiry"
        elif any(word in query_lower for word in ["safety", "side effect", "risk"]):
            pattern_type = "safety_inquiry"
        elif any(word in query_lower for word in ["mechanism", "how", "pathway"]):
            pattern_type = "mechanism_inquiry"
        elif any(word in query_lower for word in ["benefit", "effect", "improve"]):
            pattern_type = "benefits_inquiry"
        elif any(word in query_lower for word in ["stack", "combination", "synergy"]):
            pattern_type = "stacking_inquiry"
        else:
            pattern_type = "general_inquiry"

        return {
            "pattern_type": pattern_type,
            "length": len(query.split()),
            "complexity": self._assess_query_complexity(query)
        }

    def _assess_query_complexity(self, query: str) -> int:
        """Assess query complexity (1-5 scale)"""

        technical_terms = ["mechanism", "pathway", "receptor", "bioavailability", "pharmacokinetics"]
        compound_names = ["bpc-157", "tb-500", "ghrp", "mod-grf", "ipamorelin"]

        complexity = 1

        # Length factor
        if len(query.split()) > 10:
            complexity += 1

        # Technical terms
        if any(term in query.lower() for term in technical_terms):
            complexity += 1

        # Specific compounds
        if any(compound in query.lower() for compound in compound_names):
            complexity += 1

        # Multiple questions or concepts
        if "and" in query.lower() or "vs" in query.lower() or "?" in query:
            complexity += 1

        return min(complexity, 5)

    async def _schedule_predictive_preloading(
        self,
        insights: List[PredictiveInsight],
        user_profile: UserProfile
    ):
        """Schedule predictive preloading based on insights"""

        for insight in insights[:3]:  # Limit to top 3 insights
            for query in insight.suggested_queries[:2]:  # Top 2 queries per insight
                preload_item = {
                    "query": query,
                    "user_id": user_profile.user_id,
                    "confidence": insight.confidence,
                    "scheduled_at": time.time()
                }

                await self.preload_queue.put(preload_item)

    async def _check_predictive_cache(
        self,
        query: str,
        user_profile: UserProfile
    ) -> Optional[Dict[str, Any]]:
        """Check if we have predictively cached results"""

        if not self.cache:
            return None

        # Generate cache key
        cache_key = f"predictive_{user_profile.user_id}_{hashlib.md5(query.encode()).hexdigest()}"

        try:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                logger.info(f"🎯 Predictive cache hit for: {query[:50]}")
                return cached_result
        except Exception as e:
            logger.warning(f"Predictive cache check failed: {e}")

        return None

    async def _load_user_profiles(self):
        """Load user profiles from disk"""
        profiles_file = self.data_dir / "user_profiles.json"

        if profiles_file.exists():
            try:
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)

                for user_id, profile_data in profiles_data.items():
                    self.user_profiles[user_id] = UserProfile(**profile_data)

                logger.info(f"✅ Loaded {len(self.user_profiles)} user profiles")
            except Exception as e:
                logger.warning(f"Failed to load user profiles: {e}")

    async def _save_user_profiles(self):
        """Save user profiles to disk"""
        profiles_file = self.data_dir / "user_profiles.json"

        try:
            profiles_data = {}
            for user_id, profile in self.user_profiles.items():
                profiles_data[user_id] = {
                    "user_id": profile.user_id,
                    "research_interests": profile.research_interests,
                    "experience_level": profile.experience_level,
                    "safety_preference": profile.safety_preference,
                    "preferred_evidence_level": profile.preferred_evidence_level,
                    "target_goals": profile.target_goals,
                    "compound_history": profile.compound_history,
                    "successful_protocols": profile.successful_protocols,
                    "adverse_reactions": profile.adverse_reactions,
                    "preferred_sources": profile.preferred_sources,
                    "learning_preferences": profile.learning_preferences,
                    "research_patterns": profile.research_patterns,
                    "success_metrics": profile.success_metrics,
                    "learning_weights": profile.learning_weights
                }

            with open(profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)

            logger.info(f"✅ Saved {len(self.user_profiles)} user profiles")
        except Exception as e:
            logger.warning(f"Failed to save user profiles: {e}")

    async def _load_learning_data(self):
        """Load learning patterns and optimization rules"""
        learning_file = self.data_dir / "learning_data.json"

        if learning_file.exists():
            try:
                with open(learning_file, 'r') as f:
                    learning_data = json.load(f)

                self.query_patterns = learning_data.get("query_patterns", {})
                self.source_performance = learning_data.get("source_performance", {})
                self.optimization_rules = learning_data.get("optimization_rules", {})

                logger.info("✅ Loaded learning data")
            except Exception as e:
                logger.warning(f"Failed to load learning data: {e}")

    async def _save_learning_data(self):
        """Save learning patterns and optimization rules"""
        learning_file = self.data_dir / "learning_data.json"

        try:
            learning_data = {
                "query_patterns": self.query_patterns,
                "source_performance": self.source_performance,
                "optimization_rules": self.optimization_rules,
                "last_updated": datetime.now().isoformat()
            }

            with open(learning_file, 'w') as f:
                json.dump(learning_data, f, indent=2)

            logger.info("✅ Saved learning data")
        except Exception as e:
            logger.warning(f"Failed to save learning data: {e}")

    def _initialize_phase1_optimizations(self):
        """Inicializuje Phase 1 optimalizační komponenty"""
        try:
            from async_performance_optimizer import M1AsyncOptimizer
            from token_optimization_system import TokenOptimizedMCPResponse
            from distributed_cache_optimizer import create_distributed_cache
            from connection_pool_optimizer import ConnectionPoolManager

            # Initialize components
            self.async_optimizer = M1AsyncOptimizer(
                max_concurrent_requests=20,
                batch_size=10,
                enable_m1_optimizations=True
            )

            self.token_optimizer = TokenOptimizedMCPResponse()
            self.connection_pool_manager = ConnectionPoolManager()

            logger.info("Phase 1 optimizations initialized successfully")

        except ImportError as e:
            logger.warning(f"Phase 1 optimization components not available: {e}")

    async def research_with_optimizations(
        self,
        query: str,
        enable_async_optimization: bool = True,
        enable_token_optimization: bool = True,
        enable_cache_optimization: bool = True,
        enable_connection_pooling: bool = True,
        priority_level: str = "STANDARD"
    ) -> Dict[str, Any]:
        """
        Research s kompletními Phase 1 optimalizacemi
        - Async processing optimization
        - Token-optimized responses
        - Distributed caching
        - Advanced connection pooling
        """
        start_time = time.time()

        # Initialize distributed cache if needed
        if enable_cache_optimization and self.distributed_cache is None:
            try:
                from distributed_cache_optimizer import create_distributed_cache
                # Default Redis URLs (můžeš upravit podle své konfigurace)
                redis_urls = [
                    "redis://localhost:6379/0",
                    "redis://localhost:6380/0"  # Backup instance
                ]
                self.distributed_cache = await create_distributed_cache(redis_urls)
            except Exception as e:
                logger.warning(f"Distributed cache initialization failed: {e}")
                enable_cache_optimization = False

        # Check cache first
        cache_key = f"research_optimized:{hashlib.md5(query.encode()).hexdigest()}"
        if enable_cache_optimization and self.distributed_cache:
            cached_result = await self.distributed_cache.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_result

        # Prepare async batch requests
        if enable_async_optimization and self.async_optimizer:
            # Create batch requests for different sources
            from async_performance_optimizer import BatchRequest

            batch_requests = [
                BatchRequest(
                    id=f"source_{i}",
                    query=query,
                    source=source,
                    priority=1
                ) for i, source in enumerate([
                    "pubmed", "arxiv", "google_scholar",
                    "semantic_scholar", "crossref"
                ])
            ]

            # Process with async optimization
            logger.info(f"Processing {len(batch_requests)} sources with async optimization")
            raw_results = await self.async_optimizer.process_batch_requests(batch_requests)

        else:
            # Fallback to sequential processing
            raw_results = await self._process_sources_sequential(query)

        # Aggregate and process results
        aggregated_results = await self._aggregate_research_results(raw_results, query)

        # Apply token optimization
        if enable_token_optimization and self.token_optimizer:
            from token_optimization_system import ResponsePriority

            priority_map = {
                "CRITICAL": ResponsePriority.CRITICAL,
                "STANDARD": ResponsePriority.STANDARD,
                "DETAILED": ResponsePriority.DETAILED,
                "VERBOSE": ResponsePriority.VERBOSE
            }

            priority = priority_map.get(priority_level, ResponsePriority.STANDARD)
            optimized_response = await self.token_optimizer.optimize_response(
                aggregated_results, priority
            )

            final_results = optimized_response['response']
            token_metrics = optimized_response['metrics']

            logger.info(f"Token optimization: {token_metrics.compression_ratio:.1%} size reduction")

        else:
            final_results = aggregated_results
            token_metrics = None

        # Add optimization metadata
        processing_time = time.time() - start_time
        final_results['optimization_metadata'] = {
            "processing_time": processing_time,
            "async_optimization_enabled": enable_async_optimization,
            "token_optimization_enabled": enable_token_optimization,
            "cache_optimization_enabled": enable_cache_optimization,
            "connection_pooling_enabled": enable_connection_pooling,
            "token_metrics": {
                "compression_ratio": token_metrics.compression_ratio if token_metrics else 0,
                "tokens_saved": token_metrics.tokens_saved if token_metrics else 0
            } if token_metrics else None,
            "async_metrics": {
                "batch_size": len(batch_requests) if 'batch_requests' in locals() else 0,
                "avg_response_time": self.async_optimizer.metrics.avg_response_time if self.async_optimizer else 0
            } if enable_async_optimization else None
        }

        # Cache optimized results
        if enable_cache_optimization and self.distributed_cache:
            await self.distributed_cache.set(cache_key, final_results, ttl=3600)

        logger.info(f"Research completed with optimizations in {processing_time:.2f}s")
        return final_results

    async def _process_sources_sequential(self, query: str) -> List[Dict[str, Any]]:
        """Fallback sequential processing when async optimizer is not available"""
        sources = ["pubmed", "arxiv", "google_scholar", "semantic_scholar", "crossref"]
        results = []

        for source in sources:
            try:
                # Simulate source processing
                await asyncio.sleep(0.1)  # Simulate API call
                results.append({
                    "source": source,
                    "query": query,
                    "results": [f"Mock result from {source}"],
                    "success": True
                })
            except Exception as e:
                results.append({
                    "source": source,
                    "query": query,
                    "error": str(e),
                    "success": False
                })

        return results

    async def _aggregate_research_results(self, raw_results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Agreguje výsledky z různých zdrojů"""
        successful_results = [r for r in raw_results if r.get('success', False)]
        failed_results = [r for r in raw_results if not r.get('success', False)]

        # Create aggregated response
        aggregated = {
            "query": query,
            "sources": successful_results,
            "failed_sources": failed_results,
            "summary": {
                "total_sources": len(raw_results),
                "successful_sources": len(successful_results),
                "failed_sources": len(failed_results)
            },
            "quality_metrics": {
                "overall_score": len(successful_results) / len(raw_results) if raw_results else 0,
                "source_diversity": len(set(r.get('source', '') for r in successful_results)),
                "reliability": 0.8  # Mock reliability score
            },
            "generated_at": datetime.now().isoformat(),
            "processing_metadata": {
                "version": "enhanced_with_phase1_optimizations",
                "optimization_level": "phase1"
            }
        }

        return aggregated

    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Získá statistiky všech Phase 1 optimalizací"""
        stats = {
            "async_optimization": None,
            "token_optimization": None,
            "distributed_cache": None,
            "connection_pooling": None
        }

        # Async optimizer stats
        if self.async_optimizer:
            stats["async_optimization"] = {
                "total_requests": self.async_optimizer.metrics.total_requests,
                "successful_requests": self.async_optimizer.metrics.successful_requests,
                "avg_response_time": self.async_optimizer.metrics.avg_response_time,
                "m1_optimizations_used": self.async_optimizer.metrics.m1_optimizations_used
            }

        # Distributed cache stats
        if self.distributed_cache:
            stats["distributed_cache"] = await self.distributed_cache.get_cache_stats()

        # Connection pool stats
        if self.connection_pool_manager:
            stats["connection_pooling"] = await self.connection_pool_manager.get_aggregate_stats()

        return stats

    async def close(self):
        """Zavře všechny optimalizační komponenty"""
        if self.distributed_cache:
            await self.distributed_cache.close()

        if self.connection_pool_manager:
            await self.connection_pool_manager.close_all()

        logger.info("Enhanced Research Orchestrator optimization components closed")

# For backward compatibility
EnhancedIntelligentOrchestrator = ConsolidatedResearchOrchestrator
IntelligentResearchOrchestrator = ConsolidatedResearchOrchestrator

# Factory function for easy instantiation
def get_research_orchestrator():
    """Factory function to get the consolidated research orchestrator"""
    return ConsolidatedResearchOrchestrator()

# Export
__all__ = [
    'ConsolidatedResearchOrchestrator',
    'EnhancedIntelligentOrchestrator',  # Backward compatibility
    'IntelligentResearchOrchestrator',  # Backward compatibility
    'get_research_orchestrator',
    'ResearchMode',
    'UserProfile',
    'PredictiveInsight',
    'ResearchSession',
    'PerformanceMetrics'
]
