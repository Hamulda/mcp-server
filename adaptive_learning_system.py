"""
Enhanced Adaptive Learning System - Pokročilý systém učení s AI personalizací
Implementuje učení z uživatelských vzorů, personalizované AI prompty a generování insights
"""

import asyncio
import json
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
from collections import defaultdict, Counter

try:
    from unified_cache_system import get_unified_cache
    from local_ai_adapter import M1OptimizedOllamaClient
    from peptide_prompts import PEPTIDE_RESEARCH_PROMPTS, BIOHACKING_PROMPTS
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class UserLearningProfile:
    """Rozšířený learning profil uživatele"""
    user_id: str
    learning_style: str = "balanced"  # visual, analytical, practical, balanced
    expertise_level: str = "intermediate"
    preferred_detail_level: str = "medium"  # brief, medium, detailed, expert
    successful_query_patterns: List[str] = field(default_factory=list)
    failed_query_patterns: List[str] = field(default_factory=list)
    topic_expertise: Dict[str, float] = field(default_factory=dict)  # topic -> expertise score 0-1
    response_preferences: Dict[str, float] = field(default_factory=dict)
    learning_velocity: float = 0.5  # How fast user learns (0-1)
    curiosity_score: float = 0.7  # How exploratory vs focused (0-1)

@dataclass
class PersonalizedPrompt:
    """Personalizovaný AI prompt"""
    base_prompt: str
    user_adaptations: List[str]
    complexity_level: str
    focus_areas: List[str]
    generated_prompt: str

class EnhancedAdaptiveLearningSystem:
    """
    Pokročilý adaptivní learning systém s AI personalizací
    """

    def __init__(self):
        self.user_profiles = {}
        self.learning_patterns = {}
        self.prompt_templates = {}
        self.ai_client = None
        self.cache = get_unified_cache()

        # Learning analytics
        self.pattern_analyzer = PatternAnalyzer()
        self.insight_generator = InsightGenerator()
        self.prompt_personalizer = PromptPersonalizer()

        # Data storage
        self.data_dir = Path("data/adaptive_learning")
        self.data_dir.mkdir(exist_ok=True)

        # Initialize base prompt templates
        self._initialize_prompt_templates()

    def _initialize_prompt_templates(self):
        """Inicializace základních prompt šablon"""
        self.prompt_templates = {
            "beginner": {
                "research": """Explain {topic} in simple terms suitable for a beginner:
1. What is it and what does it do?
2. Is it safe for beginners?
3. Basic dosing information
4. Key things to know before starting
5. Common mistakes to avoid

Use simple language and focus on safety.""",

                "safety": """Provide beginner-friendly safety information about {topic}:
1. Is this safe for beginners?
2. What are the most common side effects?
3. What should I watch out for?
4. When should I stop using it?
5. Should I consult a doctor?

Keep explanations simple and emphasize safety."""
            },

            "intermediate": {
                "research": """Provide comprehensive information about {topic}:
1. Mechanism of action and pharmacology
2. Evidence-based benefits and effects
3. Optimal dosing protocols and timing
4. Potential side effects and contraindications
5. Interactions with other substances
6. Quality of research evidence

Balance detail with practicality.""",

                "safety": """Analyze the safety profile of {topic}:
1. Side effect profile and frequency
2. Contraindications and warnings
3. Drug and supplement interactions
4. Special populations considerations
5. Long-term safety data
6. Risk mitigation strategies

Provide evidence-based assessment."""
            },

            "expert": {
                "research": """Provide expert-level analysis of {topic}:
1. Detailed pharmacokinetics and pharmacodynamics
2. Molecular mechanisms and receptor interactions
3. Clinical trial data and meta-analyses
4. Dosing optimization and cycling protocols
5. Synergistic combinations and stacking
6. Research gaps and future directions
7. Regulatory status and legal considerations

Include technical details and citations.""",

                "safety": """Conduct expert safety analysis of {topic}:
1. Comprehensive adverse event profile
2. Mechanism-based toxicity predictions
3. Population pharmacokinetics variations
4. Complex drug interaction analysis
5. Long-term safety extrapolations
6. Risk-benefit optimization strategies
7. Monitoring parameters and biomarkers

Provide clinical-grade assessment."""
            }
        }

    async def __aenter__(self):
        """Initialize learning system"""
        if DEPS_AVAILABLE:
            self.ai_client = M1OptimizedOllamaClient()
            await self.ai_client.__aenter__()

        await self._load_learning_data()
        logger.info("✅ Enhanced Adaptive Learning System initialized")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup"""
        if self.ai_client:
            await self.ai_client.__aexit__(exc_type, exc_val, exc_tb)
        await self._save_learning_data()

    async def learn_from_interaction(
        self,
        user_id: str,
        query: str,
        response: Dict[str, Any],
        user_feedback: Optional[Dict] = None,
        success_metrics: Optional[Dict] = None
    ):
        """
        Učení z uživatelské interakce
        """
        profile = await self._get_learning_profile(user_id)

        # Analyze query patterns
        await self._analyze_query_pattern(profile, query, response, success_metrics)

        # Process user feedback if provided
        if user_feedback:
            await self._process_user_feedback(profile, query, response, user_feedback)

        # Update expertise levels
        await self._update_topic_expertise(profile, query, success_metrics)

        # Generate learning insights
        insights = await self._generate_learning_insights(profile, query, response)

        # Update learning velocity and curiosity
        await self._update_learning_characteristics(profile, query, response, user_feedback)

        logger.info(f"📚 Learning updated for user {user_id}")
        return insights

    async def generate_personalized_prompt(
        self,
        user_id: str,
        prompt_type: str,
        topic: str,
        context: Optional[Dict] = None
    ) -> PersonalizedPrompt:
        """
        Generování personalizovaného AI promptu
        """
        profile = await self._get_learning_profile(user_id)

        # Select base template
        base_template = self._select_base_template(profile, prompt_type)

        # Personalize prompt based on user profile
        personalized_prompt = await self._personalize_prompt(
            base_template, profile, topic, context
        )

        return personalized_prompt

    async def _get_learning_profile(self, user_id: str) -> UserLearningProfile:
        """Získání nebo vytvoření learning profilu"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserLearningProfile(user_id=user_id)
        return self.user_profiles[user_id]

    async def _analyze_query_pattern(
        self,
        profile: UserLearningProfile,
        query: str,
        response: Dict[str, Any],
        success_metrics: Optional[Dict]
    ):
        """Analýza vzorů dotazů"""

        # Extract query characteristics
        query_features = {
            "length": len(query.split()),
            "complexity": await self._assess_query_complexity(query),
            "topic": await self._extract_topic(query),
            "intent": await self._classify_intent(query)
        }

        # Determine if query was successful
        success = self._determine_success(response, success_metrics)

        if success:
            profile.successful_query_patterns.append(json.dumps(query_features))
        else:
            profile.failed_query_patterns.append(json.dumps(query_features))

        # Keep only recent patterns (last 100)
        profile.successful_query_patterns = profile.successful_query_patterns[-100:]
        profile.failed_query_patterns = profile.failed_query_patterns[-100:]

    async def _personalize_prompt(
        self,
        base_template: str,
        profile: UserLearningProfile,
        topic: str,
        context: Optional[Dict]
    ) -> PersonalizedPrompt:
        """Personalizace promptu podle profilu uživatele"""

        # Start with base template
        prompt = base_template.format(topic=topic)

        adaptations = []

        # Adjust for learning style
        if profile.learning_style == "visual":
            adaptations.append("Include visual descriptions and analogies")
            prompt += "\n\nUse visual analogies and descriptive language."

        elif profile.learning_style == "analytical":
            adaptations.append("Focus on data, studies, and logical analysis")
            prompt += "\n\nEmphasize scientific data, studies, and logical reasoning."

        elif profile.learning_style == "practical":
            adaptations.append("Focus on practical applications and protocols")
            prompt += "\n\nFocus on practical applications and step-by-step protocols."

        # Adjust for curiosity level
        if profile.curiosity_score > 0.7:
            adaptations.append("Include related topics and future directions")
            prompt += "\n\nAlso mention related topics and emerging research directions."

        # Adjust for expertise in this topic
        topic_expertise = profile.topic_expertise.get(topic, 0.5)
        if topic_expertise > 0.8:
            adaptations.append("Provide advanced technical details")
            prompt += "\n\nInclude advanced technical details and nuances."
        elif topic_expertise < 0.3:
            adaptations.append("Keep explanations simple and basic")
            prompt += "\n\nKeep explanations simple and avoid jargon."

        # Add user-specific preferences
        if "safety_focused" in profile.response_preferences:
            if profile.response_preferences["safety_focused"] > 0.7:
                adaptations.append("Emphasize safety considerations")
                prompt += "\n\nPay special attention to safety considerations and warnings."

        return PersonalizedPrompt(
            base_prompt=base_template,
            user_adaptations=adaptations,
            complexity_level=profile.expertise_level,
            focus_areas=list(profile.topic_expertise.keys())[:5],
            generated_prompt=prompt
        )

    def _select_base_template(self, profile: UserLearningProfile, prompt_type: str) -> str:
        """Výběr základní šablony podle profilu"""

        # Determine effective expertise level
        if profile.expertise_level == "beginner":
            level = "beginner"
        elif profile.expertise_level == "expert":
            level = "expert"
        else:
            level = "intermediate"

        return self.prompt_templates[level].get(prompt_type,
                                                self.prompt_templates[level]["research"])

class PatternAnalyzer:
    """Analyzér vzorů v uživatelském chování"""

    def __init__(self):
        self.pattern_cache = {}

    async def analyze_success_patterns(self, successful_queries: List[str]) -> Dict[str, Any]:
        """Analýza úspěšných vzorů"""
        if not successful_queries:
            return {}

        patterns = []
        for query_json in successful_queries:
            try:
                pattern = json.loads(query_json)
                patterns.append(pattern)
            except:
                continue

        if not patterns:
            return {}

        # Analyze common characteristics
        common_features = {
            "avg_length": np.mean([p["length"] for p in patterns]),
            "common_topics": Counter(p["topic"] for p in patterns).most_common(5),
            "common_intents": Counter(p["intent"] for p in patterns).most_common(3),
            "complexity_range": (
                min(p["complexity"] for p in patterns),
                max(p["complexity"] for p in patterns)
            )
        }

        return common_features

class InsightGenerator:
    """Generátor insights pro learning systém"""

    async def generate_insights(
        self,
        profile: UserLearningProfile,
        recent_interactions: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generování personalizovaných insights"""

        insights = []

        # Learning velocity insight
        if profile.learning_velocity > 0.8:
            insights.append({
                "type": "learning_acceleration",
                "message": "You're learning quickly! Consider exploring advanced topics.",
                "suggestions": ["Try expert-level queries", "Explore related compounds"]
            })
        elif profile.learning_velocity < 0.3:
            insights.append({
                "type": "learning_support",
                "message": "Take your time to understand basics before advancing.",
                "suggestions": ["Review fundamental concepts", "Use simpler queries"]
            })

        # Topic expertise insights
        top_topics = sorted(profile.topic_expertise.items(),
                          key=lambda x: x[1], reverse=True)[:3]

        if top_topics:
            insights.append({
                "type": "expertise_areas",
                "message": f"Your strongest areas: {', '.join([t[0] for t in top_topics])}",
                "suggestions": [f"Explore advanced aspects of {top_topics[0][0]}"]
            })

        return insights

class PromptPersonalizer:
    """Personalizace AI promptů"""

    def __init__(self):
        self.personalization_rules = {}

    async def create_personalized_research_prompt(
        self,
        base_prompt: str,
        user_profile: UserLearningProfile,
        topic: str
    ) -> str:
        """Vytvoření personalizovaného research promptu"""

        # Start with base
        personalized = base_prompt

        # Add user-specific modifiers
        modifiers = []

        if user_profile.learning_style == "practical":
            modifiers.append("Focus on practical applications and protocols")

        if user_profile.curiosity_score > 0.7:
            modifiers.append("Include related research directions")

        # Apply expertise adjustments
        topic_expertise = user_profile.topic_expertise.get(topic, 0.5)
        if topic_expertise > 0.7:
            modifiers.append("Provide technical details and mechanisms")
        elif topic_expertise < 0.4:
            modifiers.append("Use simple language and explain basics")

        # Combine modifiers
        if modifiers:
            personalized += f"\n\nPersonalization: {'. '.join(modifiers)}."

        return personalized

# Export
__all__ = [
    'EnhancedAdaptiveLearningSystem',
    'UserLearningProfile',
    'PersonalizedPrompt',
    'PatternAnalyzer',
    'InsightGenerator',
    'PromptPersonalizer'
]
