Advanced Biohacking Research Engine - Specialized for Peptides & Longevity Research
Optimalizováno pro lokální AI výzkum s maximální soukromostí
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import hashlib
import re
from pathlib import Path

try:
    from local_ai_adapter import M1OptimizedOllamaClient, quick_ai_query
    from unified_research_engine import M1OptimizedResearchEngine
    from peptide_prompts import PEPTIDE_RESEARCH_PROMPTS, BIOHACKING_PROMPTS
    from academic_scraper import create_scraping_orchestrator
    LOCAL_DEPS_AVAILABLE = True
except ImportError:
    LOCAL_DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class BiohackingResearchRequest:
    """Specializovaný request pro biohacking výzkum"""
    compound: str
    research_type: str = "comprehensive"  # comprehensive, safety, dosage, interactions
    target_areas: List[str] = field(default_factory=lambda: ["longevity", "performance", "recovery"])
    include_safety: bool = True
    include_stacking: bool = True
    evidence_level: str = "high"  # high, medium, all
    max_depth: int = 3

@dataclass
class CompoundProfile:
    """Profil bioaktivní látky"""
    name: str
    category: str
    mechanism: str
    half_life: Optional[str] = None
    dosage_range: Optional[str] = None
    administration: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    interactions: Dict[str, str] = field(default_factory=dict)
    research_status: str = "unknown"
    safety_rating: str = "unknown"
    legal_status: str = "unknown"
    sources: List[str] = field(default_factory=list)

class BiohackingResearchEngine:
    """Pokročilý engine pro biohacking výzkum"""

    def __init__(self):
        self.ai_client = None
        self.research_engine = M1OptimizedResearchEngine()
        self.compound_database = {}
        self.interaction_matrix = {}

        # Specialized knowledge bases
        self.peptide_db_path = Path("data/peptides.json")
        self.nootropics_db_path = Path("data/nootropics.json")
        self.interactions_db_path = Path("data/interactions.json")

        # Research sources prioritization
        self.trusted_sources = {
            "pubmed": 10,
            "clinicaltrials.gov": 9,
            "examine.com": 8,
            "selfhacked.com": 6,
            "reddit.com/r/peptides": 4,
            "reddit.com/r/nootropics": 4
        }

    async def __aenter__(self):
        if LOCAL_DEPS_AVAILABLE:
            self.ai_client = M1OptimizedOllamaClient()
            await self.ai_client.__aenter__()
        await self.research_engine.__aenter__()
        await self._load_knowledge_bases()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.ai_client:
            await self.ai_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.research_engine.__aexit__(exc_type, exc_val, exc_tb)

    async def _load_knowledge_bases(self):
        """Načte lokální databáze znalostí"""
        for db_path in [self.peptide_db_path, self.nootropics_db_path, self.interactions_db_path]:
            if db_path.exists():
                try:
                    with open(db_path) as f:
                        data = json.load(f)
                        if "peptides" in db_path.name:
                            self.compound_database.update(data)
                        elif "interactions" in db_path.name:
                            self.interaction_matrix = data
                except Exception as e:
                    logger.warning(f"Failed to load {db_path}: {e}")

    async def research_compound(self, request: BiohackingResearchRequest) -> CompoundProfile:
        """Comprehensive compound research"""
        logger.info(f"🔬 Researching compound: {request.compound}")

        # Check local database first
        profile = self._get_cached_profile(request.compound)
        if not profile:
            profile = CompoundProfile(name=request.compound, category="unknown", mechanism="unknown")

        # Multi-phase research approach
        tasks = []

        if request.research_type in ["comprehensive", "safety"]:
            tasks.append(self._research_safety_profile(request.compound))

        if request.research_type in ["comprehensive", "dosage"]:
            tasks.append(self._research_dosage_protocols(request.compound))

        if request.research_type in ["comprehensive", "interactions"]:
            tasks.append(self._research_interactions(request.compound))

        if request.include_stacking:
            tasks.append(self._research_stacking_protocols(request.compound, request.target_areas))

        # Execute research tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results into profile
        for result in results:
            if isinstance(result, dict) and not isinstance(result, Exception):
                await self._merge_research_data(profile, result)

        # AI analysis for synthesis
        if self.ai_client:
            profile = await self._ai_synthesis(profile, request)

        # Save to cache
        await self._cache_profile(profile)

        return profile

    async def _research_safety_profile(self, compound: str) -> Dict:
        """Research safety and side effects"""
        query = f"{compound} safety profile side effects contraindications clinical studies"

        search_results = await self.research_engine.search_academic_sources(
            query, max_sources=5, domains=["pubmed", "clinicaltrials"]
        )

        if self.ai_client:
            prompt = f"""Analyze the safety profile of {compound} based on this research data:
            
{json.dumps(search_results, indent=2)}

Extract:
1. Common side effects (frequency if available)
2. Serious adverse events
3. Contraindications
4. Special populations warnings
5. Long-term safety data
6. Dosage-dependent effects

Format as structured JSON."""

            safety_analysis = await self.ai_client.generate_response(prompt)
            try:
                return json.loads(safety_analysis)
            except:
                return {"safety_text": safety_analysis, "raw_data": search_results}

        return {"raw_data": search_results}

    async def _research_dosage_protocols(self, compound: str) -> Dict:
        """Research optimal dosing protocols"""
        query = f"{compound} dosage protocol administration frequency cycling"

        # Search multiple source types
        academic_results = await self.research_engine.search_academic_sources(query, max_sources=3)
        community_results = await self.research_engine.search_community_sources(query, max_sources=2)

        if self.ai_client:
            prompt = f"""Analyze dosing protocols for {compound}:

Academic data: {json.dumps(academic_results, indent=2)}
Community data: {json.dumps(community_results, indent=2)}

Extract dosing information:
1. Standard dose ranges (mg/kg or absolute)
2. Administration frequency
3. Timing recommendations
4. Cycling protocols
5. Dose escalation strategies
6. Administration routes
7. Storage requirements

Prioritize peer-reviewed data. Format as JSON."""

            dosage_analysis = await self.ai_client.generate_response(prompt)
            try:
                return json.loads(dosage_analysis)
            except:
                return {"dosage_text": dosage_analysis, "academic": academic_results, "community": community_results}

        return {"academic": academic_results, "community": community_results}

    async def _research_interactions(self, compound: str) -> Dict:
        """Research drug and supplement interactions"""
        # Check local interaction matrix first
        known_interactions = self.interaction_matrix.get(compound.lower(), {})

        query = f"{compound} drug interactions contraindications CYP450 metabolism"
        results = await self.research_engine.search_academic_sources(query, max_sources=4)

        if self.ai_client:
            prompt = f"""Analyze potential interactions for {compound}:

Known interactions: {json.dumps(known_interactions)}
Research data: {json.dumps(results, indent=2)}

Identify:
1. Major drug interactions
2. Supplement interactions
3. Food interactions
4. Metabolic pathway conflicts
5. Synergistic combinations
6. Contraindicated combinations

Rate interaction severity (major/moderate/minor). Format as JSON."""

            interaction_analysis = await self.ai_client.generate_response(prompt)
            try:
                parsed = json.loads(interaction_analysis)
                parsed["cached_interactions"] = known_interactions
                return parsed
            except:
                return {"interaction_text": interaction_analysis, "known": known_interactions, "research": results}

        return {"known": known_interactions, "research": results}

    async def _research_stacking_protocols(self, compound: str, target_areas: List[str]) -> Dict:
        """Research effective stacking combinations"""
        stacks = {}

        for area in target_areas:
            query = f"{compound} stack combination {area} synergy protocol"
            results = await self.research_engine.search_community_sources(query, max_sources=3)
            stacks[area] = results

        if self.ai_client:
            prompt = f"""Analyze stacking protocols for {compound} targeting {target_areas}:

{json.dumps(stacks, indent=2)}

For each target area, identify:
1. Synergistic compounds
2. Recommended stack ratios
3. Timing protocols
4. Expected benefits
5. Potential risks
6. Experience reports quality

Focus on evidence-based combinations. Format as JSON."""

            stack_analysis = await self.ai_client.generate_response(prompt)
            try:
                return json.loads(stack_analysis)
            except:
                return {"stack_text": stack_analysis, "raw_stacks": stacks}

        return {"raw_stacks": stacks}

    async def _ai_synthesis(self, profile: CompoundProfile, request: BiohackingResearchRequest) -> CompoundProfile:
        """AI synthesis of all research data"""
        prompt = f"""Synthesize comprehensive profile for {profile.name}:

Current profile: {json.dumps(profile.__dict__, indent=2)}

Create final assessment:
1. Overall mechanism summary
2. Evidence-based benefits
3. Risk assessment
4. Practical recommendations
5. Research quality rating
6. Legal/availability status

Be critical of evidence quality. Highlight gaps in research."""

        synthesis = await self.ai_client.generate_response(prompt)

        # Parse and update profile
        try:
            updates = json.loads(synthesis)
            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
        except:
            profile.ai_synthesis = synthesis

        return profile

    def _get_cached_profile(self, compound: str) -> Optional[CompoundProfile]:
        """Get cached compound profile"""
        return self.compound_database.get(compound.lower())

    async def _cache_profile(self, profile: CompoundProfile):
        """Cache compound profile"""
        self.compound_database[profile.name.lower()] = profile

        # Save to disk
        cache_file = Path("data/compound_cache.json")
        cache_file.parent.mkdir(exist_ok=True)

        try:
            with open(cache_file, 'w') as f:
                # Convert dataclass to dict for JSON serialization
                serializable_db = {}
                for key, value in self.compound_database.items():
                    if isinstance(value, CompoundProfile):
                        serializable_db[key] = value.__dict__
                    else:
                        serializable_db[key] = value
                json.dump(serializable_db, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache profile: {e}")

    async def _merge_research_data(self, profile: CompoundProfile, data: Dict):
        """Merge research data into profile"""
        if "safety" in data:
            safety = data["safety"]
            if isinstance(safety, dict):
                profile.side_effects.extend(safety.get("side_effects", []))
                profile.contraindications.extend(safety.get("contraindications", []))

        if "dosage" in data:
            dosage = data["dosage"]
            if isinstance(dosage, dict):
                profile.dosage_range = dosage.get("range", profile.dosage_range)
                profile.administration.extend(dosage.get("routes", []))

        if "interactions" in data:
            interactions = data["interactions"]
            if isinstance(interactions, dict):
                profile.interactions.update(interactions.get("major", {}))

        if "benefits" in data:
            benefits = data["benefits"]
            if isinstance(benefits, list):
                profile.benefits.extend(benefits)

# Specialized research functions for common biohacking categories
class PeptideResearchSpecialist:
    """Specialized research for peptides"""

    def __init__(self, engine: BiohackingResearchEngine):
        self.engine = engine

    async def research_growth_hormone_peptides(self, peptide: str) -> Dict:
        """Research GH-related peptides (GHRP, GHRH, etc.)"""
        specialized_query = f"{peptide} growth hormone release IGF-1 muscle growth fat loss"

        research_areas = [
            "growth_hormone_effects",
            "body_composition_changes",
            "aging_benefits",
            "dosing_frequency",
            "injection_timing"
        ]

        results = {}
        for area in research_areas:
            query = f"{peptide} {area.replace('_', ' ')}"
            results[area] = await self.engine.research_engine.search_academic_sources(query, max_sources=2)

        return results

    async def research_nootropic_peptides(self, peptide: str) -> Dict:
        """Research cognitive enhancement peptides"""
        specialized_query = f"{peptide} cognitive enhancement memory focus neuroprotection"

        cognitive_areas = [
            "memory_enhancement",
            "focus_attention",
            "neuroprotection",
            "neuroplasticity",
            "mood_effects"
        ]

        results = {}
        for area in cognitive_areas:
            query = f"{peptide} {area.replace('_', ' ')}"
            results[area] = await self.engine.research_engine.search_academic_sources(query, max_sources=2)

        return results

# Export hlavních tříd
__all__ = [
    'BiohackingResearchEngine',
    'BiohackingResearchRequest',
    'CompoundProfile',
    'PeptideResearchSpecialist'
]
