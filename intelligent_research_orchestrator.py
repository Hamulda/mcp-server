Intelligent Research Orchestrator - AI-řízený orchestrátor pro optimalizaci výzkumu
Personalizované doporučení, adaptive learning, a performance monitoring
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import hashlib

try:
    from biohacking_research_engine import BiohackingResearchEngine, BiohackingResearchRequest, CompoundProfile
    from advanced_source_aggregator import AdvancedSourceAggregator
    from local_ai_adapter import M1OptimizedOllamaClient
    from peptide_prompts import PEPTIDE_RESEARCH_PROMPTS, BIOHACKING_PROMPTS, QUALITY_CONTROL_PROMPTS
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """Uživatelský profil pro personalizaci"""
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

@dataclass
class ResearchSession:
    """Session pro sledování výzkumné aktivity"""
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
    """Metriky výkonu systému"""
    query_response_time: float
    source_success_rate: float
    ai_processing_time: float
    cache_hit_rate: float
    memory_usage_mb: float
    user_engagement_score: float
    research_depth_score: float
    safety_compliance_score: float

class IntelligentResearchOrchestrator:
    """AI-řízený orchestrátor výzkumu s adaptivním učením"""

    def __init__(self):
        self.user_profiles = {}
        self.research_sessions = {}
        self.performance_history = []

        # AI components
        self.ai_client = None
        self.research_engine = None
        self.source_aggregator = None

        # Learning and optimization
        self.query_embeddings = {}
        self.success_patterns = {}
        self.optimization_rules = {}

        # Paths
        self.data_dir = Path("data/orchestrator")
        self.data_dir.mkdir(exist_ok=True)

        # Performance monitoring
        self.metrics_buffer = []
        self.optimization_threshold = 10  # sessions before optimization

    async def __aenter__(self):
        """Initialize all components"""
        if DEPS_AVAILABLE:
            self.ai_client = M1OptimizedOllamaClient()
            await self.ai_client.__aenter__()

            self.research_engine = BiohackingResearchEngine()
            await self.research_engine.__aenter__()

            self.source_aggregator = AdvancedSourceAggregator()
            await self.source_aggregator.__aenter__()

        await self._load_user_profiles()
        await self._load_optimization_rules()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup"""
        if self.ai_client:
            await self.ai_client.__aexit__(exc_type, exc_val, exc_tb)
        if self.research_engine:
            await self.research_engine.__aexit__(exc_type, exc_val, exc_tb)
        if self.source_aggregator:
            await self.source_aggregator.__aexit__(exc_type, exc_val, exc_tb)

        await self._save_user_profiles()
        await self._save_optimization_rules()

    async def intelligent_research(
        self,
        query: str,
        user_id: str = "default",
        research_type: str = "auto"
    ) -> Dict[str, Any]:
        """Inteligentní výzkum s personalizací a optimalizací"""

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

            # 2. Personalized research execution
            research_request = BiohackingResearchRequest(
                compound=enhanced_query,
                research_type=research_strategy,
                target_areas=user_profile.target_goals,
                evidence_level=user_profile.preferred_evidence_level,
                include_safety=user_profile.safety_preference != "aggressive"
            )

            # 3. Execute research with monitoring
            research_results = await self._execute_monitored_research(
                research_request, user_profile, session
            )

            # 4. AI-powered result synthesis
            synthesized_results = await self._synthesize_results(
                research_results, user_profile, query
            )

            # 5. Generate personalized recommendations
            recommendations = await self._generate_recommendations(
                synthesized_results, user_profile
            )

            # 6. Safety assessment
            safety_assessment = await self._assess_safety(
                synthesized_results, user_profile
            )

            # 7. Update user profile and learning
            await self._update_learning(session, synthesized_results, user_profile)

            # 8. Performance tracking
            session_time = time.time() - session_start
            metrics = PerformanceMetrics(
                query_response_time=session_time,
                source_success_rate=self._calculate_source_success_rate(research_results),
                ai_processing_time=0.0,  # Would be tracked separately
                cache_hit_rate=0.0,      # Would be tracked separately
                memory_usage_mb=self._get_memory_usage(),
                user_engagement_score=0.0,  # Would be updated based on user feedback
                research_depth_score=self._calculate_research_depth(synthesized_results),
                safety_compliance_score=safety_assessment.get("score", 0.0)
            )

            self.performance_history.append(metrics)

            # 9. Prepare final response
            response = {
                "session_id": session_id,
                "query": query,
                "enhanced_query": enhanced_query,
                "research_strategy": research_strategy,
                "results": synthesized_results,
                "recommendations": recommendations,
                "safety_assessment": safety_assessment,
                "performance_metrics": metrics.__dict__,
                "personalization_applied": True,
                "sources_used": list(research_results.keys()),
                "research_quality_score": self._calculate_quality_score(synthesized_results),
                "next_suggested_queries": await self._suggest_follow_up_queries(
                    query, synthesized_results, user_profile
                )
            }

            logger.info(f"✅ Research completed in {session_time:.2f}s with quality score {response['research_quality_score']:.2f}")

            return response

        except Exception as e:
            logger.error(f"❌ Research failed: {e}")
            return {
                "session_id": session_id,
                "error": str(e),
                "query": query,
                "performance_metrics": {"query_response_time": time.time() - session_start}
            }

    async def _analyze_and_enhance_query(
        self,
        query: str,
        user_profile: UserProfile
    ) -> Tuple[str, str]:
        """Analýza a vylepšení dotazu na základě AI a user profilu"""

        # Detect query intent and compound type
        intent_analysis = await self._analyze_query_intent(query)

        # Enhance based on user profile
        enhanced_query = query
        research_strategy = "balanced"

        # Apply user preferences
        if user_profile.experience_level == "beginner":
            enhanced_query += " safety basics introduction"
            research_strategy = "safety"
        elif user_profile.experience_level == "expert":
            enhanced_query += " advanced mechanisms pharmacology"
            research_strategy = "comprehensive"

        # Add target goals context
        if user_profile.target_goals:
            enhanced_query += f" {' '.join(user_profile.target_goals)}"

        # Apply learning from previous successful queries
        if query.lower() in self.success_patterns:
            pattern = self.success_patterns[query.lower()]
            enhanced_query = pattern.get("enhanced_query", enhanced_query)
            research_strategy = pattern.get("strategy", research_strategy)

        return enhanced_query, research_strategy

    async def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """AI analýza záměru dotazu"""
        if not self.ai_client:
            return {"intent": "general", "compound_type": "unknown"}

        prompt = f"""Analyze this biohacking/research query and identify:
1. Primary intent (safety, dosing, mechanisms, stacking, etc.)
2. Compound type (peptide, nootropic, supplement, etc.)
3. Experience level implied (beginner, intermediate, advanced)
4. Specific areas of interest
5. Safety concerns mentioned

Query: {query}

Respond with JSON format."""

        try:
            response = await self.ai_client.generate_response(prompt)
            return json.loads(response)
        except:
            # Fallback analysis
            return self._fallback_query_analysis(query)

    def _fallback_query_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback analýza bez AI"""
        intent = "general"
        compound_type = "unknown"

        # Simple keyword matching
        if any(word in query.lower() for word in ["dose", "dosing", "how much"]):
            intent = "dosing"
        elif any(word in query.lower() for word in ["safe", "safety", "side effect"]):
            intent = "safety"
        elif any(word in query.lower() for word in ["mechanism", "how does", "works"]):
            intent = "mechanisms"
        elif any(word in query.lower() for word in ["stack", "combine", "together"]):
            intent = "stacking"

        if "peptide" in query.lower():
            compound_type = "peptide"
        elif any(word in query.lower() for word in ["nootropic", "smart drug", "cognitive"]):
            compound_type = "nootropic"

        return {"intent": intent, "compound_type": compound_type}

    async def _execute_monitored_research(
        self,
        request: BiohackingResearchRequest,
        user_profile: UserProfile,
        session: ResearchSession
    ) -> Dict[str, Any]:
        """Sledované provádění výzkumu"""

        start_time = time.time()

        # Select sources based on user preferences
        preferred_sources = user_profile.preferred_sources or None

        # Execute research
        if self.research_engine:
            compound_profile = await self.research_engine.research_compound(request)

            # Also get multi-source data
            if self.source_aggregator:
                source_results = await self.source_aggregator.multi_source_search(
                    request.compound,
                    sources=preferred_sources,
                    max_results_per_source=3,
                    evidence_level=user_profile.preferred_evidence_level
                )
            else:
                source_results = {}

            research_results = {
                "compound_profile": compound_profile.__dict__ if compound_profile else {},
                "source_results": source_results,
                "execution_time": time.time() - start_time
            }
        else:
            research_results = {"error": "Research engine not available"}

        # Update session
        session.time_spent = time.time() - start_time
        session.compounds_researched.append(request.compound)

        return research_results

    async def _synthesize_results(
        self,
        research_results: Dict[str, Any],
        user_profile: UserProfile,
        original_query: str
    ) -> Dict[str, Any]:
        """AI syntéza výsledků podle uživatelských preferencí"""

        if not self.ai_client:
            return research_results

        # Prepare synthesis prompt based on user level
        if user_profile.experience_level == "beginner":
            synthesis_prompt = BIOHACKING_PROMPTS["protocol_optimization"]
        elif user_profile.experience_level == "expert":
            synthesis_prompt = "expert_synthesis"
        else:
            synthesis_prompt = "comprehensive_analysis"

        prompt = f"""Synthesize this research data for a {user_profile.experience_level} user interested in {user_profile.target_goals}:

Original Query: {original_query}
Research Data: {json.dumps(research_results, indent=2)}

Focus on:
1. Most relevant findings for user goals
2. Practical recommendations
3. Safety considerations for {user_profile.safety_preference} approach
4. Evidence quality assessment
5. Next steps for research

Provide clear, actionable insights."""

        try:
            synthesis = await self.ai_client.generate_response(prompt)
            return {
                "ai_synthesis": synthesis,
                "raw_data": research_results,
                "personalization_applied": user_profile.experience_level
            }
        except Exception as e:
            logger.warning(f"AI synthesis failed: {e}")
            return research_results

    async def _generate_recommendations(
        self,
        synthesized_results: Dict[str, Any],
        user_profile: UserProfile
    ) -> List[Dict[str, Any]]:
        """Generování personalizovaných doporučení"""

        recommendations = []

        # Safety-based recommendations
        if user_profile.safety_preference == "conservative":
            recommendations.append({
                "type": "safety",
                "priority": "high",
                "title": "Start with minimal effective dose",
                "description": "Begin with the lowest recommended dose and monitor response",
                "reasoning": "Conservative safety approach"
            })

        # Experience-based recommendations
        if user_profile.experience_level == "beginner":
            recommendations.append({
                "type": "education",
                "priority": "high",
                "title": "Research basics first",
                "description": "Study fundamental mechanisms before starting",
                "reasoning": "Beginner safety and education"
            })

        # Goal-based recommendations
        for goal in user_profile.target_goals:
            if goal == "longevity":
                recommendations.append({
                    "type": "protocol",
                    "priority": "medium",
                    "title": f"Consider longevity biomarkers",
                    "description": "Monitor telomeres, NAD+, inflammatory markers",
                    "reasoning": f"Aligned with {goal} goal"
                })

        return recommendations

    async def _assess_safety(
        self,
        results: Dict[str, Any],
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Bezpečnostní hodnocení"""

        safety_score = 5.0  # Default medium safety
        warnings = []
        recommendations = []

        # Check against user's adverse reaction history
        for adverse_reaction in user_profile.adverse_reactions:
            if adverse_reaction.get("compound") in str(results):
                warnings.append(f"Previous adverse reaction to similar compound")
                safety_score -= 2.0

        # Safety level adjustments
        if user_profile.safety_preference == "conservative":
            safety_score = min(safety_score, 7.0)
            recommendations.append("Extra monitoring recommended")
        elif user_profile.safety_preference == "aggressive":
            safety_score = max(safety_score, 3.0)

        return {
            "score": max(0.0, min(10.0, safety_score)),
            "warnings": warnings,
            "recommendations": recommendations,
            "level": user_profile.safety_preference
        }

    async def _update_learning(
        self,
        session: ResearchSession,
        results: Dict[str, Any],
        user_profile: UserProfile
    ):
        """Aktualizace učení a optimalizace"""

        # Update success patterns
        for query in session.queries:
            if query.lower() not in self.success_patterns:
                self.success_patterns[query.lower()] = {
                    "count": 1,
                    "avg_quality": results.get("quality_score", 0.0),
                    "successful_strategies": []
                }
            else:
                pattern = self.success_patterns[query.lower()]
                pattern["count"] += 1

        # Update user profile interests
        for compound in session.compounds_researched:
            if compound not in user_profile.research_interests:
                user_profile.research_interests.append(compound)

    def _calculate_source_success_rate(self, results: Dict[str, Any]) -> float:
        """Výpočet úspěšnosti zdrojů"""
        if not results or "source_results" not in results:
            return 0.0

        source_results = results["source_results"]
        total_sources = len(source_results)
        successful_sources = sum(1 for results in source_results.values() if results)

        return successful_sources / total_sources if total_sources > 0 else 0.0

    def _calculate_research_depth(self, results: Dict[str, Any]) -> float:
        """Výpočet hloubky výzkumu"""
        # Simple heuristic based on content length and source variety
        depth_score = 0.0

        if "ai_synthesis" in results:
            depth_score += min(len(results["ai_synthesis"]) / 1000, 3.0)

        if "source_results" in results.get("raw_data", {}):
            source_count = len(results["raw_data"]["source_results"])
            depth_score += min(source_count / 5.0, 2.0)

        return min(depth_score, 5.0)

    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Výpočet celkové kvality výsledků"""
        quality_factors = []

        # AI synthesis quality
        if "ai_synthesis" in results:
            quality_factors.append(min(len(results["ai_synthesis"]) / 500, 1.0))

        # Source diversity
        if "source_results" in results.get("raw_data", {}):
            source_diversity = len(results["raw_data"]["source_results"]) / 8.0
            quality_factors.append(min(source_diversity, 1.0))

        # Default quality if no factors
        if not quality_factors:
            return 0.5

        return sum(quality_factors) / len(quality_factors)

    async def _suggest_follow_up_queries(
        self,
        original_query: str,
        results: Dict[str, Any],
        user_profile: UserProfile
    ) -> List[str]:
        """Návrhy následujících dotazů"""

        suggestions = []

        # Based on original query type
        if "dosing" not in original_query.lower():
            suggestions.append(f"{original_query} dosing protocol")

        if "safety" not in original_query.lower():
            suggestions.append(f"{original_query} safety side effects")

        if "stack" not in original_query.lower():
            suggestions.append(f"{original_query} stacking combinations")

        # Based on user goals
        for goal in user_profile.target_goals[:2]:  # Limit to 2
            suggestions.append(f"{original_query} {goal} optimization")

        return suggestions[:5]  # Limit suggestions

    async def _get_user_profile(self, user_id: str) -> UserProfile:
        """Získání nebo vytvoření user profilu"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        return self.user_profiles[user_id]

    def _generate_session_id(self) -> str:
        """Generování session ID"""
        return hashlib.md5(f"{time.time()}".encode()).hexdigest()[:12]

    def _get_memory_usage(self) -> float:
        """Získání využití paměti (zjednodušeno)"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0

    async def _load_user_profiles(self):
        """Načtení uživatelských profilů"""
        profiles_file = self.data_dir / "user_profiles.json"
        if profiles_file.exists():
            try:
                with open(profiles_file) as f:
                    data = json.load(f)
                    for user_id, profile_data in data.items():
                        self.user_profiles[user_id] = UserProfile(**profile_data)
            except Exception as e:
                logger.warning(f"Failed to load user profiles: {e}")

    async def _save_user_profiles(self):
        """Uložení uživatelských profilů"""
        profiles_file = self.data_dir / "user_profiles.json"
        try:
            serializable_profiles = {}
            for user_id, profile in self.user_profiles.items():
                serializable_profiles[user_id] = profile.__dict__

            with open(profiles_file, 'w') as f:
                json.dump(serializable_profiles, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save user profiles: {e}")

    async def _load_optimization_rules(self):
        """Načtení optimalizačních pravidel"""
        rules_file = self.data_dir / "optimization_rules.json"
        if rules_file.exists():
            try:
                with open(rules_file) as f:
                    self.optimization_rules = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load optimization rules: {e}")

    async def _save_optimization_rules(self):
        """Uložení optimalizačních pravidel"""
        rules_file = self.data_dir / "optimization_rules.json"
        try:
            with open(rules_file, 'w') as f:
                json.dump(self.optimization_rules, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save optimization rules: {e}")

# Export
__all__ = ['IntelligentResearchOrchestrator', 'UserProfile', 'ResearchSession', 'PerformanceMetrics']
