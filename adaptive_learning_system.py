Adaptive Learning & Personalization System
Učí se z vašich vzorů výzkumu a personalizuje výsledky pro vaše specifické potřeby
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import sqlite3

try:
    from unified_config import get_config
    from local_ai_adapter import M1OptimizedOllamaClient
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class UserPreference:
    """Uživatelská preference pro personalizaci"""
    category: str  # 'peptide_type', 'research_focus', 'detail_level'
    value: str
    confidence: float  # 0.0 - 1.0
    frequency: int
    last_updated: float

@dataclass
class ResearchSession:
    """Záznam research session pro learning"""
    session_id: str
    timestamp: float
    queries: List[str]
    sources_used: List[str]
    time_spent_seconds: float
    user_satisfaction: Optional[float] = None
    follow_up_queries: List[str] = field(default_factory=list)

class PersonalizationEngine:
    """Engine pro personalizaci výsledků na základě učení"""

    def __init__(self, user_profile_dir: Optional[Path] = None):
        self.profile_dir = user_profile_dir or Path("cache/user_profile")
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.profile_dir / "user_profile.db"
        self._init_database()

        # In-memory caches
        self.preferences: Dict[str, UserPreference] = {}
        self.research_patterns: Dict[str, int] = defaultdict(int)
        self.peptide_interests: Counter = Counter()
        self.preferred_sources: Counter = Counter()

        # Load existing preferences
        asyncio.create_task(self._load_preferences())

    def _init_database(self):
        """Initialize SQLite database for user profile"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    category TEXT,
                    value TEXT,
                    confidence REAL,
                    frequency INTEGER,
                    last_updated REAL,
                    PRIMARY KEY (category, value)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_sessions (
                    session_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    queries TEXT,
                    sources_used TEXT,
                    time_spent_seconds REAL,
                    user_satisfaction REAL,
                    follow_up_queries TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_insights (
                    insight_type TEXT,
                    insight_data TEXT,
                    confidence REAL,
                    created_at REAL
                )
            """)

    async def _load_preferences(self):
        """Load existing preferences from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM preferences")
            for row in cursor.fetchall():
                category, value, confidence, frequency, last_updated = row
                key = f"{category}:{value}"
                self.preferences[key] = UserPreference(
                    category=category,
                    value=value,
                    confidence=confidence,
                    frequency=frequency,
                    last_updated=last_updated
                )

    async def learn_from_query(self, query: str, sources_used: List[str],
                             time_spent: float, follow_ups: List[str] = None):
        """Učí se z uživatelského dotazu a chování"""

        # Analyze query patterns
        await self._analyze_query_patterns(query)

        # Learn source preferences
        for source in sources_used:
            self.preferred_sources[source] += 1
            await self._update_preference('preferred_source', source, 0.1)

        # Learn from follow-up queries (indicates interest depth)
        if follow_ups:
            for follow_up in follow_ups:
                await self._learn_interest_depth(query, follow_up)

        # Store research session
        session = ResearchSession(
            session_id=f"session_{int(time.time())}",
            timestamp=time.time(),
            queries=[query] + (follow_ups or []),
            sources_used=sources_used,
            time_spent_seconds=time_spent,
            follow_up_queries=follow_ups or []
        )

        await self._store_research_session(session)

    async def _analyze_query_patterns(self, query: str):
        """Analyzuje vzory v dotazech pro učení preferencí"""
        query_lower = query.lower()

        # Peptide preferences
        peptide_keywords = {
            'bpc': 'bpc-157', 'tb500': 'tb-500', 'gh': 'growth_hormone',
            'ghrp': 'ghrp', 'cjc': 'cjc-1295', 'ipamorelin': 'ipamorelin',
            'thymosin': 'thymosin', 'melanotan': 'melanotan'
        }

        for keyword, peptide in peptide_keywords.items():
            if keyword in query_lower:
                self.peptide_interests[peptide] += 1
                await self._update_preference('peptide_interest', peptide, 0.2)

        # Research focus preferences
        focus_keywords = {
            'dosage': 'dosing_protocols',
            'side effects': 'safety_focus',
            'mechanism': 'scientific_depth',
            'benefits': 'outcomes_focus',
            'stack': 'combination_research',
            'clinical': 'clinical_evidence_preference'
        }

        for keyword, focus in focus_keywords.items():
            if keyword in query_lower:
                await self._update_preference('research_focus', focus, 0.15)

        # Detail level preferences
        if len(query.split()) > 20:
            await self._update_preference('detail_level', 'comprehensive', 0.1)
        elif len(query.split()) < 5:
            await self._update_preference('detail_level', 'concise', 0.1)

    async def _update_preference(self, category: str, value: str, confidence_delta: float):
        """Update user preference with learning"""
        key = f"{category}:{value}"

        if key in self.preferences:
            pref = self.preferences[key]
            pref.confidence = min(1.0, pref.confidence + confidence_delta)
            pref.frequency += 1
            pref.last_updated = time.time()
        else:
            self.preferences[key] = UserPreference(
                category=category,
                value=value,
                confidence=confidence_delta,
                frequency=1,
                last_updated=time.time()
            )

        # Persist to database
        with sqlite3.connect(self.db_path) as conn:
            pref = self.preferences[key]
            conn.execute("""
                INSERT OR REPLACE INTO preferences 
                (category, value, confidence, frequency, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """, (pref.category, pref.value, pref.confidence, pref.frequency, pref.last_updated))

    async def _learn_interest_depth(self, original_query: str, follow_up: str):
        """Learn from follow-up queries to understand interest depth"""
        # If user asks follow-up questions, they're deeply interested in the topic
        original_lower = original_query.lower()
        follow_up_lower = follow_up.lower()

        # Extract common elements
        original_words = set(original_lower.split())
        follow_up_words = set(follow_up_lower.split())
        common_words = original_words & follow_up_words

        # If there's significant overlap, user is diving deeper
        if len(common_words) >= 2:
            await self._update_preference('research_depth', 'thorough', 0.2)

    async def _store_research_session(self, session: ResearchSession):
        """Store research session for analysis"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO research_sessions
                (session_id, timestamp, queries, sources_used, time_spent_seconds, 
                 user_satisfaction, follow_up_queries)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.timestamp,
                json.dumps(session.queries),
                json.dumps(session.sources_used),
                session.time_spent_seconds,
                session.user_satisfaction,
                json.dumps(session.follow_up_queries)
            ))

    async def get_personalized_sources(self, query: str, max_sources: int = 5) -> List[str]:
        """Získá personalizované zdroje na základě učení"""

        # Start with user's preferred sources
        preferred = []
        for source, count in self.preferred_sources.most_common():
            if len(preferred) < max_sources // 2:  # Half from preferences
                preferred.append(source)

        # Add query-specific sources
        query_lower = query.lower()

        # Peptide-specific source preferences
        if any(p in query_lower for p in ['peptide', 'bpc', 'tb500', 'gh']):
            peptide_sources = ['peptide_guide', 'pubmed', 'clinicaltrials']
            for source in peptide_sources:
                if source not in preferred and len(preferred) < max_sources:
                    preferred.append(source)

        # Fill remaining slots with high-quality defaults
        default_sources = ['examine_com', 'pubmed', 'selfhacked', 'mayo_clinic']
        for source in default_sources:
            if source not in preferred and len(preferred) < max_sources:
                preferred.append(source)

        return preferred[:max_sources]

    async def personalize_ai_prompt(self, base_prompt: str, query: str) -> str:
        """Personalizuje AI prompt na základě uživatelských preferencí"""

        personalization_additions = []

        # Detail level preference
        detail_pref = self._get_preference('detail_level')
        if detail_pref and detail_pref.confidence > 0.3:
            if detail_pref.value == 'comprehensive':
                personalization_additions.append("Provide comprehensive, detailed analysis with mechanisms and citations.")
            elif detail_pref.value == 'concise':
                personalization_additions.append("Keep response concise and focus on key practical points.")

        # Research focus preferences
        focus_prefs = [p for p in self.preferences.values()
                      if p.category == 'research_focus' and p.confidence > 0.2]

        if focus_prefs:
            focus_areas = [p.value.replace('_', ' ') for p in focus_prefs]
            personalization_additions.append(f"Pay special attention to: {', '.join(focus_areas)}.")

        # Safety focus if learned
        safety_pref = self._get_preference('research_focus', 'safety_focus')
        if safety_pref and safety_pref.confidence > 0.3:
            personalization_additions.append("Include detailed safety analysis and contraindications.")

        # Add personalization to prompt
        if personalization_additions:
            personalized_prompt = base_prompt + "\n\nPersonalization notes:\n" + "\n".join(personalization_additions)
            return personalized_prompt

        return base_prompt

    def _get_preference(self, category: str, value: str = None) -> Optional[UserPreference]:
        """Get specific preference"""
        if value:
            key = f"{category}:{value}"
            return self.preferences.get(key)
        else:
            # Get strongest preference in category
            category_prefs = [p for p in self.preferences.values() if p.category == category]
            if category_prefs:
                return max(category_prefs, key=lambda p: p.confidence)
        return None

    async def generate_insights(self) -> Dict[str, Any]:
        """Generuje insights o uživatelských preferencích pomocí AI"""

        if not CONFIG_AVAILABLE or not self.preferences:
            return {"insights": "Insufficient data for insights"}

        # Prepare data for AI analysis
        preference_summary = {}
        for pref in self.preferences.values():
            if pref.category not in preference_summary:
                preference_summary[pref.category] = []
            preference_summary[pref.category].append({
                'value': pref.value,
                'confidence': pref.confidence,
                'frequency': pref.frequency
            })

        insights_prompt = f"""
        Analyze this user's research patterns and generate personalized insights:
        
        Preferences: {json.dumps(preference_summary, indent=2)}
        Top peptide interests: {dict(self.peptide_interests.most_common(5))}
        Preferred sources: {dict(self.preferred_sources.most_common(5))}
        
        Generate insights about:
        1. Primary research interests
        2. Preferred information depth
        3. Safety consciousness level
        4. Recommended research areas to explore
        5. Potential knowledge gaps to address
        
        Keep insights practical and actionable.
        """

        try:
            async with M1OptimizedOllamaClient() as client:
                insights = await client.generate_optimized(
                    insights_prompt,
                    priority="balanced",
                    use_specialized_prompt=False
                )

                return {
                    "insights": insights,
                    "preference_summary": preference_summary,
                    "top_interests": dict(self.peptide_interests.most_common(5)),
                    "generated_at": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return {"error": f"Insight generation failed: {str(e)}"}

    def get_learning_stats(self) -> Dict[str, Any]:
        """Získá statistiky učení"""
        return {
            'total_preferences': len(self.preferences),
            'high_confidence_preferences': len([p for p in self.preferences.values() if p.confidence > 0.5]),
            'peptide_interests_count': len(self.peptide_interests),
            'preferred_sources_count': len(self.preferred_sources),
            'most_researched_peptide': self.peptide_interests.most_common(1)[0] if self.peptide_interests else None,
            'most_used_source': self.preferred_sources.most_common(1)[0] if self.preferred_sources else None,
            'learning_categories': list(set(p.category for p in self.preferences.values()))
        }

class AdaptiveResponseOptimizer:
    """Optimalizuje odpovědi na základě naučených preferencí"""

    def __init__(self, personalization_engine: PersonalizationEngine):
        self.personalization = personalization_engine

    async def optimize_response_format(self, response: str, query: str) -> str:
        """Optimalizuje formát odpovědi na základě preferencí"""

        # Get user preferences
        detail_pref = self.personalization._get_preference('detail_level')
        safety_pref = self.personalization._get_preference('research_focus', 'safety_focus')

        optimizations = []

        # Add safety warnings if user is safety-focused
        if safety_pref and safety_pref.confidence > 0.3:
            if 'warning' not in response.lower() and 'side effect' not in response.lower():
                optimizations.append("⚠️ SAFETY NOTE: Always consult healthcare professionals before using any peptides or supplements.")

        # Add structure for comprehensive preference
        if detail_pref and detail_pref.value == 'comprehensive' and detail_pref.confidence > 0.3:
            if not any(marker in response for marker in ['##', '**', '1.', '2.']):
                # Add structure markers
                optimizations.append("📋 [Response has been structured for comprehensive analysis]")

        # Apply optimizations
        if optimizations:
            optimized_response = "\n".join(optimizations) + "\n\n" + response
            return optimized_response

        return response

# Global personalization instance
_personalization_instance = None

def get_personalization_engine() -> PersonalizationEngine:
    """Get global personalization engine instance"""
    global _personalization_instance
    if _personalization_instance is None:
        _personalization_instance = PersonalizationEngine()
    return _personalization_instance

# Convenience functions
async def learn_from_research_session(query: str, sources: List[str], time_spent: float, follow_ups: List[str] = None):
    """Learn from a research session"""
    engine = get_personalization_engine()
    await engine.learn_from_query(query, sources, time_spent, follow_ups)

async def get_personalized_research_setup(query: str) -> Dict[str, Any]:
    """Get personalized research setup"""
    engine = get_personalization_engine()

    sources = await engine.get_personalized_sources(query)
    base_prompt = f"Research this topic: {query}"
    personalized_prompt = await engine.personalize_ai_prompt(base_prompt, query)

    return {
        'personalized_sources': sources,
        'personalized_prompt': personalized_prompt,
        'user_insights': await engine.generate_insights()
    }
