Peptide & Biohacking Research Prompts - Specializované prompty pro maximální kvalitu
Optimalizované pro Llama 3.1 8B model na MacBook Air M1
"""

# Peptide research prompts
PEPTIDE_RESEARCH_PROMPTS = {
    "basic_info": """Analyze this peptide compound with focus on:
1. Mechanism of action
2. Typical dosing protocols
3. Half-life and administration
4. Known benefits and research evidence
5. Potential side effects
6. Drug interactions
7. Legal status and availability

Provide evidence-based information with citations where possible. Be precise about dosages and contraindications.

Peptide: {query}""",

    "dosage_analysis": """Provide detailed dosing information for this peptide:
1. Standard dosing range (mcg/mg per kg bodyweight)
2. Injection frequency and timing
3. Cycling protocols (on/off periods)
4. Storage requirements
5. Reconstitution instructions
6. Administration methods (SubQ, IM, etc.)
7. Dose escalation protocols

Focus on safety and cite clinical studies.

Peptide: {query}""",

    "safety_profile": """Analyze the safety profile of this peptide:
1. Common side effects and frequency
2. Serious adverse events
3. Contraindications
4. Drug interactions
5. Special populations (pregnancy, elderly, etc.)
6. Long-term usage considerations
7. Monitoring requirements

Provide risk-benefit analysis based on available data.

Peptide: {query}""",

    "research_summary": """Summarize current research on this peptide:
1. Clinical trial status and results
2. Primary research areas and applications
3. Key studies and their findings
4. Research gaps and future directions
5. Quality of evidence assessment
6. Regulatory status updates

Peptide: {query}""",

    "stacking_analysis": """Analyze stacking potential for this peptide:
1. Synergistic combinations
2. Contraindicated combinations
3. Timing and sequencing
4. Dose adjustments when stacked
5. Enhanced effects and risks
6. Monitoring requirements

Focus on evidence-based combinations and safety.

Peptide: {query}""",

    "longevity_focus": """Analyze this peptide's longevity and anti-aging potential:
1. Cellular mechanisms (autophagy, senescence, etc.)
2. Biomarker improvements
3. Healthspan vs lifespan effects
4. Age-related decline reversal
5. Long-term safety in healthy aging
6. Comparison to other longevity interventions

Peptide: {query}""",

    "performance_enhancement": """Analyze this peptide for performance enhancement:
1. Athletic performance benefits
2. Recovery acceleration
3. Injury prevention/healing
4. Muscle growth and fat loss
5. Endurance improvements
6. Cognitive performance effects
7. Detection in drug testing

Peptide: {query}"""
}

# Biohacking research prompts
BIOHACKING_PROMPTS = {
    "nootropic_analysis": """Analyze this nootropic compound comprehensively:
1. Mechanism of action (neurotransmitter systems)
2. Cognitive benefits (memory, focus, creativity)
3. Optimal dosing and timing
4. Stacking potential and synergies
5. Side effects and tolerance
6. Long-term safety profile
7. Individual variation factors

Compound: {query}""",

    "supplement_interaction": """Analyze potential interactions for this supplement:
1. Major drug interactions
2. Supplement-supplement interactions
3. Food and nutrient interactions
4. Timing considerations
5. Absorption factors
6. Metabolic pathway competition
7. Synergistic combinations

Focus on safety and efficacy optimization.

Supplement: {query}""",

    "biomarker_optimization": """Analyze how this intervention affects key biomarkers:
1. Cardiovascular markers (lipids, CRP, homocysteine)
2. Metabolic markers (glucose, insulin, HbA1c)
3. Hormonal markers (testosterone, cortisol, thyroid)
4. Inflammatory markers (IL-6, TNF-α)
5. Longevity markers (telomeres, NAD+)
6. Cognitive markers (BDNF, acetylcholine)
7. Timeline for measurable changes

Intervention: {query}""",

    "protocol_optimization": """Design an optimal protocol for this biohacking intervention:
1. Dosing schedule (daily/cycling)
2. Timing optimization (circadian)
3. Combination strategies
4. Monitoring parameters
5. Adjustment criteria
6. Safety checkpoints
7. Expected timeline and outcomes

Intervention: {query}""",

    "risk_assessment": """Perform comprehensive risk assessment:
1. Short-term risks and mitigation
2. Long-term safety concerns
3. Individual risk factors
4. Contraindications
5. Monitoring requirements
6. Red flag symptoms
7. Risk-benefit analysis

Intervention: {query}""",

    "evidence_evaluation": """Evaluate the quality of evidence for this intervention:
1. Clinical trial quality and quantity
2. Study limitations and biases
3. Population studied vs target population
4. Outcome measures relevance
5. Statistical significance vs clinical significance
6. Replication and consistency
7. Overall evidence grade

Intervention: {query}""",

    "personalization_factors": """Analyze personalization factors for this intervention:
1. Genetic factors (polymorphisms)
2. Age and sex considerations
3. Health status modifiers
4. Lifestyle interactions
5. Individual response predictors
6. Adjustment strategies
7. Monitoring for personalization

Intervention: {query}"""
}

# Specialized analysis prompts for different research depths
ANALYSIS_DEPTH_PROMPTS = {
    "quick_overview": """Provide a concise overview in 3-4 bullet points:
• Primary mechanism and effects
• Standard dosing and administration
• Key safety considerations
• Evidence quality summary

Topic: {query}""",

    "comprehensive_analysis": """Provide detailed analysis covering:
1. Detailed mechanism of action
2. Complete dosing protocols
3. Full safety profile
4. Interaction analysis
5. Clinical evidence review
6. Practical implementation
7. Risk-benefit assessment
8. Personalization factors

Topic: {query}""",

    "expert_synthesis": """Provide expert-level synthesis suitable for healthcare professionals:
1. Advanced mechanistic understanding
2. Clinical pharmacology details
3. Evidence-based recommendations
4. Complex interaction analysis
5. Special population considerations
6. Monitoring and adjustment protocols
7. Research gaps and limitations
8. Future research directions

Topic: {query}"""
}

# Source-specific prompt modifications
SOURCE_PROMPTS = {
    "pubmed_focus": "Focus on peer-reviewed clinical studies and systematic reviews.",
    "clinical_trials": "Emphasize clinical trial data and regulatory submissions.",
    "community_wisdom": "Include experienced user reports and practical implementation insights.",
    "safety_database": "Prioritize adverse event reports and safety monitoring data.",
    "regulatory_focus": "Include regulatory approvals, warnings, and legal status information."
}

# Compound category prompts
CATEGORY_PROMPTS = {
    "growth_hormone": """Focus on growth hormone pathway analysis:
1. GH/IGF-1 axis effects
2. Body composition changes
3. Metabolic improvements
4. Sleep quality effects
5. Recovery benefits
6. Age-related considerations

Compound: {query}""",

    "cognitive_enhancement": """Focus on cognitive enhancement analysis:
1. Neurotransmitter effects
2. Memory and learning
3. Focus and attention
4. Mood and motivation
5. Neuroprotection
6. Long-term brain health

Compound: {query}""",

    "metabolic_optimization": """Focus on metabolic optimization:
1. Insulin sensitivity
2. Fat oxidation
3. Mitochondrial function
4. Energy production
5. Metabolic flexibility
6. Weight management

Compound: {query}""",

    "recovery_enhancement": """Focus on recovery and repair:
1. Tissue repair mechanisms
2. Inflammation modulation
3. Sleep quality improvement
4. Stress adaptation
5. Immune function
6. Cellular regeneration

Compound: {query}""",

    "longevity_intervention": """Focus on longevity mechanisms:
1. Cellular aging pathways
2. DNA repair and protection
3. Senescent cell clearance
4. Autophagy enhancement
5. Oxidative stress reduction
6. Healthspan extension

Compound: {query}"""
}

# Quality control prompts
QUALITY_CONTROL_PROMPTS = {
    "source_verification": """Verify information quality:
1. Check for peer-reviewed sources
2. Identify potential conflicts of interest
3. Assess study methodology quality
4. Note sample sizes and populations
5. Evaluate statistical methods
6. Check for replication studies

Information to verify: {query}""",

    "bias_detection": """Identify potential biases in this information:
1. Selection bias in studies
2. Publication bias
3. Industry funding influence
4. Cherry-picking of data
5. Correlation vs causation issues
6. Overgeneralization concerns

Information: {query}""",

    "evidence_grading": """Grade the quality of evidence:
1. Study design quality (RCT > observational > case reports)
2. Sample size adequacy
3. Duration of follow-up
4. Outcome measure relevance
5. Statistical power
6. Replication status
7. Overall evidence grade (A/B/C/D)

Evidence to grade: {query}"""
}

# Export all prompt collections
__all__ = [
    'PEPTIDE_RESEARCH_PROMPTS',
    'BIOHACKING_PROMPTS',
    'ANALYSIS_DEPTH_PROMPTS',
    'SOURCE_PROMPTS',
    'CATEGORY_PROMPTS',
    'QUALITY_CONTROL_PROMPTS'
]
