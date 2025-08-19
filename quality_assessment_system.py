Quality Assessment System - Systém pro hodnocení kvality výzkumu a optimalizaci
Automatické vyhodnocování spolehlivosti zdrojů a kvality informací
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import statistics

try:
    from local_ai_adapter import M1OptimizedOllamaClient
    from peptide_prompts import QUALITY_CONTROL_PROMPTS
    LOCAL_AI_AVAILABLE = True
except ImportError:
    LOCAL_AI_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SourceQualityMetrics:
    """Metriky kvality zdroje"""
    source_name: str
    reliability_score: float = 0.0
    response_time: float = 0.0
    success_rate: float = 0.0
    content_quality: float = 0.0
    citation_frequency: int = 0
    peer_review_status: bool = False
    recency_score: float = 0.0
    bias_score: float = 0.0  # Lower is better
    overall_score: float = 0.0

@dataclass
class ContentQualityAssessment:
    """Hodnocení kvality obsahu"""
    content_id: str
    evidence_grade: str = "C"  # A, B, C, D
    study_quality: float = 0.0
    sample_size_adequacy: float = 0.0
    methodology_score: float = 0.0
    statistical_rigor: float = 0.0
    replication_status: str = "unknown"
    bias_indicators: List[str] = field(default_factory=list)
    confidence_level: float = 0.0
    clinical_relevance: float = 0.0

@dataclass
class SafetyAssessment:
    """Bezpečnostní hodnocení"""
    compound: str
    safety_grade: str = "Unknown"
    risk_level: str = "medium"  # low, medium, high, critical
    contraindications: List[str] = field(default_factory=list)
    interaction_warnings: List[str] = field(default_factory=list)
    dosage_safety_margin: float = 0.0
    long_term_safety_data: bool = False
    regulatory_status: str = "unregulated"
    monitoring_requirements: List[str] = field(default_factory=list)

class QualityAssessmentSystem:
    """Komplexní systém pro hodnocení kvality výzkumu"""

    def __init__(self):
        self.ai_client = None
        self.quality_database = {}
        self.source_metrics = {}
        self.assessment_cache = {}

        # Quality thresholds
        self.quality_thresholds = {
            "excellent": 8.5,
            "good": 7.0,
            "acceptable": 5.0,
            "poor": 3.0
        }

        # Evidence hierarchy
        self.evidence_hierarchy = {
            "systematic_review": 10,
            "meta_analysis": 9,
            "randomized_controlled_trial": 8,
            "cohort_study": 6,
            "case_control": 5,
            "case_series": 3,
            "case_report": 2,
            "expert_opinion": 1,
            "anecdotal": 0.5
        }

        # Bias detection patterns
        self.bias_patterns = {
            "selection_bias": [
                "cherry.picked", "selected.studies", "favorable.results",
                "excluded.negative", "only.positive"
            ],
            "funding_bias": [
                "sponsored.by", "funded.by.industry", "conflict.of.interest",
                "pharmaceutical.company", "supplement.company"
            ],
            "publication_bias": [
                "unpublished.negative", "file.drawer", "grey.literature",
                "positive.results.only"
            ],
            "confirmation_bias": [
                "supports.hypothesis", "confirms.theory", "as.expected",
                "predicted.outcome"
            ]
        }

        self.data_dir = Path("data/quality_assessment")
        self.data_dir.mkdir(exist_ok=True)

    async def __aenter__(self):
        if LOCAL_AI_AVAILABLE:
            self.ai_client = M1OptimizedOllamaClient()
            await self.ai_client.__aenter__()
        await self._load_quality_database()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.ai_client:
            await self.ai_client.__aexit__(exc_type, exc_val, exc_tb)
        await self._save_quality_database()

    async def assess_research_quality(
        self,
        research_data: Dict[str, Any],
        compound: str
    ) -> Dict[str, Any]:
        """Komplexní hodnocení kvality výzkumu"""

        logger.info(f"🔍 Assessing research quality for: {compound}")

        assessments = {
            "overall_quality": 0.0,
            "source_assessments": {},
            "content_assessments": {},
            "safety_assessment": None,
            "evidence_summary": {},
            "reliability_score": 0.0,
            "recommendations": []
        }

        # 1. Assess individual sources
        if "source_results" in research_data:
            for source_name, results in research_data["source_results"].items():
                source_assessment = await self._assess_source_quality(
                    source_name, results
                )
                assessments["source_assessments"][source_name] = source_assessment

        # 2. Assess content quality
        if "compound_profile" in research_data:
            content_assessment = await self._assess_content_quality(
                research_data["compound_profile"], compound
            )
            assessments["content_assessments"]["compound_profile"] = content_assessment

        # 3. Safety assessment
        safety_assessment = await self._assess_safety_profile(
            research_data, compound
        )
        assessments["safety_assessment"] = safety_assessment

        # 4. Evidence synthesis
        evidence_summary = await self._synthesize_evidence_quality(
            research_data, assessments
        )
        assessments["evidence_summary"] = evidence_summary

        # 5. Calculate overall scores
        overall_quality = self._calculate_overall_quality(assessments)
        assessments["overall_quality"] = overall_quality
        assessments["reliability_score"] = self._calculate_reliability_score(assessments)

        # 6. Generate recommendations
        recommendations = await self._generate_quality_recommendations(assessments)
        assessments["recommendations"] = recommendations

        # 7. Cache assessment
        await self._cache_assessment(compound, assessments)

        return assessments

    async def _assess_source_quality(
        self,
        source_name: str,
        results: List[Dict]
    ) -> SourceQualityMetrics:
        """Hodnocení kvality zdroje"""

        if source_name in self.source_metrics:
            base_metrics = self.source_metrics[source_name]
        else:
            base_metrics = SourceQualityMetrics(source_name=source_name)

        # Calculate metrics based on results
        if results:
            # Content quality based on length, structure
            content_scores = []
            for result in results:
                content_score = self._evaluate_content_structure(result)
                content_scores.append(content_score)

            base_metrics.content_quality = statistics.mean(content_scores) if content_scores else 0.0
            base_metrics.success_rate = len([r for r in results if r.get("title")]) / len(results)

        # Source-specific scoring
        source_reliability = {
            "pubmed": 9.5,
            "clinical_trials": 9.0,
            "google_scholar": 7.5,
            "examine": 8.0,
            "selfhacked": 6.0,
            "reddit_peptides": 4.0,
            "reddit_nootropics": 4.0,
            "longecity": 5.0
        }

        base_metrics.reliability_score = source_reliability.get(
            source_name.lower().replace(" ", "_"), 5.0
        )

        # Calculate overall score
        base_metrics.overall_score = (
            base_metrics.reliability_score * 0.4 +
            base_metrics.content_quality * 0.3 +
            base_metrics.success_rate * 10 * 0.3
        )

        self.source_metrics[source_name] = base_metrics
        return base_metrics

    def _evaluate_content_structure(self, content: Dict) -> float:
        """Hodnocení struktury obsahu"""
        score = 0.0

        # Title quality
        title = content.get("title", "")
        if title:
            score += 1.0
            if len(title) > 10:
                score += 0.5
            if any(word in title.lower() for word in ["study", "trial", "analysis", "review"]):
                score += 0.5

        # Snippet/abstract quality
        snippet = content.get("snippet", "")
        if snippet:
            score += 1.0
            if len(snippet) > 100:
                score += 0.5
            if len(snippet.split()) > 20:
                score += 0.5

        # URL quality (academic domains)
        url = content.get("url", "")
        academic_domains = [".edu", ".gov", "pubmed", "clinical", "scholar"]
        if any(domain in url for domain in academic_domains):
            score += 1.0

        return min(score, 5.0)

    async def _assess_content_quality(
        self,
        content: Dict,
        compound: str
    ) -> ContentQualityAssessment:
        """Hodnocení kvality obsahu pomocí AI"""

        assessment = ContentQualityAssessment(content_id=f"{compound}_profile")

        if not self.ai_client:
            return assessment

        # Prepare content for analysis
        content_text = json.dumps(content, indent=2)

        prompt = QUALITY_CONTROL_PROMPTS["evidence_grading"].format(query=content_text)

        try:
            ai_assessment = await self.ai_client.generate_response(prompt)

            # Parse AI response
            assessment_data = self._parse_ai_assessment(ai_assessment)

            # Update assessment with AI insights
            assessment.evidence_grade = assessment_data.get("grade", "C")
            assessment.study_quality = assessment_data.get("study_quality", 5.0)
            assessment.confidence_level = assessment_data.get("confidence", 5.0)
            assessment.bias_indicators = assessment_data.get("bias_indicators", [])

        except Exception as e:
            logger.warning(f"AI content assessment failed: {e}")

        # Manual quality indicators
        assessment = self._apply_manual_quality_checks(assessment, content)

        return assessment

    def _parse_ai_assessment(self, ai_response: str) -> Dict[str, Any]:
        """Parsování AI hodnocení"""
        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        # Fallback text parsing
        assessment = {}

        # Extract grade
        grade_match = re.search(r'grade[:\s]*([A-D])', ai_response, re.IGNORECASE)
        if grade_match:
            assessment["grade"] = grade_match.group(1).upper()

        # Extract numeric scores
        score_matches = re.findall(r'(\d+(?:\.\d+)?)', ai_response)
        if score_matches:
            assessment["study_quality"] = float(score_matches[0])
            if len(score_matches) > 1:
                assessment["confidence"] = float(score_matches[1])

        return assessment

    def _apply_manual_quality_checks(
        self,
        assessment: ContentQualityAssessment,
        content: Dict
    ) -> ContentQualityAssessment:
        """Manuální kontroly kvality"""

        # Check for bias indicators
        content_text = str(content).lower()

        for bias_type, patterns in self.bias_patterns.items():
            for pattern in patterns:
                if re.search(pattern.replace(".", r"\."), content_text):
                    if bias_type not in assessment.bias_indicators:
                        assessment.bias_indicators.append(bias_type)

        # Adjust confidence based on bias
        bias_penalty = len(assessment.bias_indicators) * 0.5
        assessment.confidence_level = max(0.0, assessment.confidence_level - bias_penalty)

        # Sample size assessment
        sample_size_matches = re.findall(r'n\s*=\s*(\d+)', content_text)
        if sample_size_matches:
            max_n = max(int(n) for n in sample_size_matches)
            if max_n > 1000:
                assessment.sample_size_adequacy = 10.0
            elif max_n > 100:
                assessment.sample_size_adequacy = 7.0
            elif max_n > 50:
                assessment.sample_size_adequacy = 5.0
            else:
                assessment.sample_size_adequacy = 3.0

        return assessment

    async def _assess_safety_profile(
        self,
        research_data: Dict[str, Any],
        compound: str
    ) -> SafetyAssessment:
        """Bezpečnostní hodnocení"""

        assessment = SafetyAssessment(compound=compound)

        # Extract safety information from research data
        compound_profile = research_data.get("compound_profile", {})

        # Side effects analysis
        side_effects = compound_profile.get("side_effects", [])
        if side_effects:
            serious_effects = [
                effect for effect in side_effects
                if any(word in effect.lower() for word in [
                    "severe", "serious", "death", "hospital", "emergency",
                    "cardiac", "liver", "kidney", "brain"
                ])
            ]

            if serious_effects:
                assessment.risk_level = "high"
                assessment.safety_grade = "D"
            elif len(side_effects) > 5:
                assessment.risk_level = "medium"
                assessment.safety_grade = "C"
            else:
                assessment.risk_level = "low"
                assessment.safety_grade = "B"

        # Contraindications
        contraindications = compound_profile.get("contraindications", [])
        assessment.contraindications = contraindications

        # Interactions
        interactions = compound_profile.get("interactions", {})
        if interactions:
            major_interactions = [
                interaction for interaction in interactions.values()
                if "major" in str(interaction).lower() or "severe" in str(interaction).lower()
            ]
            assessment.interaction_warnings = major_interactions

        # Regulatory status
        legal_status = compound_profile.get("legal_status", "unknown")
        if "approved" in legal_status.lower():
            assessment.regulatory_status = "approved"
        elif "prescription" in legal_status.lower():
            assessment.regulatory_status = "prescription"
        elif "banned" in legal_status.lower():
            assessment.regulatory_status = "banned"
            assessment.risk_level = "critical"

        return assessment

    async def _synthesize_evidence_quality(
        self,
        research_data: Dict[str, Any],
        assessments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Syntéza kvality důkazů"""

        evidence_levels = []
        source_scores = []

        # Collect evidence quality scores
        for source_name, source_assessment in assessments["source_assessments"].items():
            source_scores.append(source_assessment.overall_score)

        for content_name, content_assessment in assessments["content_assessments"].items():
            evidence_levels.append(content_assessment.confidence_level)

        # Calculate summary metrics
        avg_source_quality = statistics.mean(source_scores) if source_scores else 0.0
        avg_evidence_level = statistics.mean(evidence_levels) if evidence_levels else 0.0

        # Evidence consistency check
        consistency_score = self._calculate_evidence_consistency(research_data)

        return {
            "average_source_quality": avg_source_quality,
            "average_evidence_level": avg_evidence_level,
            "evidence_consistency": consistency_score,
            "number_of_sources": len(source_scores),
            "overall_evidence_grade": self._assign_evidence_grade(
                avg_source_quality, avg_evidence_level, consistency_score
            )
        }

    def _calculate_evidence_consistency(self, research_data: Dict[str, Any]) -> float:
        """Výpočet konzistence důkazů"""
        # Simplified consistency check
        # In real implementation, would compare findings across sources
        return 7.0  # Default moderate consistency

    def _assign_evidence_grade(
        self,
        source_quality: float,
        evidence_level: float,
        consistency: float
    ) -> str:
        """Přiřazení celkového grade důkazů"""

        combined_score = (source_quality + evidence_level + consistency) / 3

        if combined_score >= 8.5:
            return "A"
        elif combined_score >= 7.0:
            return "B"
        elif combined_score >= 5.0:
            return "C"
        else:
            return "D"

    def _calculate_overall_quality(self, assessments: Dict[str, Any]) -> float:
        """Výpočet celkové kvality"""

        quality_components = []

        # Source quality component
        source_assessments = assessments.get("source_assessments", {})
        if source_assessments:
            source_scores = [sa.overall_score for sa in source_assessments.values()]
            quality_components.append(statistics.mean(source_scores))

        # Content quality component
        content_assessments = assessments.get("content_assessments", {})
        if content_assessments:
            content_scores = [ca.confidence_level for ca in content_assessments.values()]
            quality_components.append(statistics.mean(content_scores))

        # Safety assessment component
        safety_assessment = assessments.get("safety_assessment")
        if safety_assessment:
            safety_grades = {"A": 10, "B": 7.5, "C": 5.0, "D": 2.5}
            safety_score = safety_grades.get(safety_assessment.safety_grade, 5.0)
            quality_components.append(safety_score)

        return statistics.mean(quality_components) if quality_components else 5.0

    def _calculate_reliability_score(self, assessments: Dict[str, Any]) -> float:
        """Výpočet skóre spolehlivosti"""

        reliability_factors = []

        # Source reliability
        source_assessments = assessments.get("source_assessments", {})
        if source_assessments:
            reliability_scores = [sa.reliability_score for sa in source_assessments.values()]
            reliability_factors.append(statistics.mean(reliability_scores))

        # Evidence grade factor
        evidence_summary = assessments.get("evidence_summary", {})
        grade = evidence_summary.get("overall_evidence_grade", "C")
        grade_scores = {"A": 10, "B": 8, "C": 6, "D": 4}
        reliability_factors.append(grade_scores[grade])

        # Bias penalty
        content_assessments = assessments.get("content_assessments", {})
        total_bias_indicators = 0
        for ca in content_assessments.values():
            total_bias_indicators += len(ca.bias_indicators)

        bias_penalty = min(total_bias_indicators * 0.5, 3.0)

        base_reliability = statistics.mean(reliability_factors) if reliability_factors else 5.0
        return max(0.0, base_reliability - bias_penalty)

    async def _generate_quality_recommendations(
        self,
        assessments: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generování doporučení na základě kvality"""

        recommendations = []

        overall_quality = assessments.get("overall_quality", 0.0)
        reliability_score = assessments.get("reliability_score", 0.0)

        # Quality-based recommendations
        if overall_quality < self.quality_thresholds["acceptable"]:
            recommendations.append({
                "type": "quality_warning",
                "priority": "high",
                "message": "Low overall research quality detected",
                "suggestion": "Seek additional high-quality sources before making decisions"
            })

        if reliability_score < 6.0:
            recommendations.append({
                "type": "reliability_warning",
                "priority": "high",
                "message": "Low reliability score",
                "suggestion": "Cross-reference with peer-reviewed sources"
            })

        # Safety-based recommendations
        safety_assessment = assessments.get("safety_assessment")
        if safety_assessment and safety_assessment.risk_level == "high":
            recommendations.append({
                "type": "safety_warning",
                "priority": "critical",
                "message": "High risk profile detected",
                "suggestion": "Consult healthcare professional before use"
            })

        # Bias-based recommendations
        content_assessments = assessments.get("content_assessments", {})
        bias_indicators = []
        for ca in content_assessments.values():
            bias_indicators.extend(ca.bias_indicators)

        if "funding_bias" in bias_indicators:
            recommendations.append({
                "type": "bias_warning",
                "priority": "medium",
                "message": "Potential funding bias detected",
                "suggestion": "Look for independent research studies"
            })

        return recommendations

    async def _cache_assessment(self, compound: str, assessment: Dict[str, Any]):
        """Cache hodnocení"""
        cache_key = f"{compound.lower().replace(' ', '_')}_assessment"
        self.assessment_cache[cache_key] = {
            "timestamp": datetime.now().isoformat(),
            "assessment": assessment
        }

    async def _load_quality_database(self):
        """Načtení databáze kvality"""
        db_file = self.data_dir / "quality_database.json"
        if db_file.exists():
            try:
                with open(db_file) as f:
                    data = json.load(f)
                    self.quality_database = data.get("quality_database", {})

                    # Load source metrics
                    source_metrics_data = data.get("source_metrics", {})
                    for source_name, metrics_dict in source_metrics_data.items():
                        self.source_metrics[source_name] = SourceQualityMetrics(**metrics_dict)

            except Exception as e:
                logger.warning(f"Failed to load quality database: {e}")

    async def _save_quality_database(self):
        """Uložení databáze kvality"""
        db_file = self.data_dir / "quality_database.json"

        try:
            # Prepare serializable data
            serializable_source_metrics = {}
            for source_name, metrics in self.source_metrics.items():
                serializable_source_metrics[source_name] = metrics.__dict__

            data = {
                "quality_database": self.quality_database,
                "source_metrics": serializable_source_metrics,
                "assessment_cache": self.assessment_cache
            }

            with open(db_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save quality database: {e}")

# Export
__all__ = [
    'QualityAssessmentSystem',
    'SourceQualityMetrics',
    'ContentQualityAssessment',
    'SafetyAssessment'
]
