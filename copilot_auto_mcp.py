#!/usr/bin/env python3
"""
Copilot Auto-MCP Integration - Inteligentní auto-aktivace MCP nástrojů pro GitHub Copilot
Automaticky rozpoznává kdy a které nástroje použít pro optimální psaní kódu
"""

import sys
from typing import Dict, Any, List
import re
from datetime import datetime

# Add project to path
sys.path.append('/Users/vojtechhamada/PycharmProjects/PythonProject2')

try:
    from github_copilot_mcp_tools import CopilotMCPInterface
    TOOLS_AVAILABLE = True
except ImportError:
    CopilotMCPInterface = None
    TOOLS_AVAILABLE = False

class GitHubCopilotAutoMCP:
    """Inteligentní auto-aktivace MCP nástrojů pro GitHub Copilot agenta"""

    def __init__(self):
        if TOOLS_AVAILABLE:
            self.mcp_tools = CopilotMCPInterface()
        else:
            self.mcp_tools = None

        # Advanced trigger patterns for GitHub Copilot code generation
        self.copilot_triggers = {
            'pattern_reuse_opportunity': [
                r'create.*class.*with',
                r'implement.*async.*function',
                r'add.*error.*handling',
                r'create.*data.*structure',
                r'build.*api.*endpoint',
                r'write.*database.*operation'
            ],
            'code_quality_enhancement': [
                r'optimize.*performance',
                r'improve.*code.*quality',
                r'refactor.*function',
                r'add.*type.*hints',
                r'enhance.*error.*handling',
                r'add.*logging'
            ],
            'token_optimization_needed': [
                r'similar.*to.*existing',
                r'like.*previous.*function',
                r'following.*same.*pattern',
                r'reuse.*existing.*logic',
                r'copy.*pattern.*from'
            ],
            'project_specific_context': [
                r'research.*function',
                r'biohacking.*data',
                r'peptide.*analysis',
                r'ai.*integration',
                r'cache.*results',
                r'academic.*scraping'
            ]
        }

        # Context patterns specific to this biohacking research project
        self.project_context_indicators = {
            'biohacking_research': ['peptide', 'research', 'biohacking', 'academic', 'compound', 'study'],
            'ai_ml_integration': ['ai', 'ollama', 'model', 'embedding', 'vector', 'llm', 'chat'],
            'async_operations': ['async', 'await', 'aiohttp', 'asyncio', 'concurrent'],
            'data_processing': ['data', 'cache', 'database', 'json', 'api', 'scraping'],
            'mcp_development': ['mcp', 'server', 'client', 'protocol', 'tools']
        }

        # Smart recommendations for different scenarios
        self.smart_recommendations = {
            'new_research_function': {
                'pattern': 'async_research_patterns',
                'reasoning': 'Research functions benefit from caching and async operations',
                'token_savings': 0.75,
                'quality_improvements': ['caching', 'error_handling', 'type_hints']
            },
            'data_structure_creation': {
                'pattern': 'data_validation_pattern',
                'reasoning': 'Dataclasses with validation improve code quality and maintainability',
                'token_savings': 0.60,
                'quality_improvements': ['validation', 'documentation', 'type_safety']
            },
            'ai_integration': {
                'pattern': 'ai_integration_pattern',
                'reasoning': 'AI operations need proper resource management and token optimization',
                'token_savings': 0.80,
                'quality_improvements': ['resource_management', 'token_optimization', 'fallback_logic']
            },
            'database_operation': {
                'pattern': 'database_operation_pattern',
                'reasoning': 'Database operations require connection pooling and transaction management',
                'token_savings': 0.65,
                'quality_improvements': ['connection_pooling', 'transactions', 'security']
            }
        }

    def should_activate_copilot_tools(self, user_message: str, file_context: Dict = None) -> Dict[str, Any]:
        """Rozhoduje zda aktivovat MCP nástroje pro GitHub Copilot"""
        activation_decision = {
            'should_activate': False,
            'confidence': 0.0,
            'reasons': [],
            'recommended_tools': [],
            'expected_benefits': {}
        }

        if not TOOLS_AVAILABLE:
            return activation_decision

        message_lower = user_message.lower()

        # Check for trigger patterns
        triggered_categories = []
        for category, patterns in self.copilot_triggers.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    triggered_categories.append(category)
                    activation_decision['reasons'].append(f"Matched {category}: {pattern}")

        # Analyze file context for additional triggers
        if file_context:
            context_score = self._analyze_file_context_for_activation(file_context, message_lower)
            if context_score > 0.5:
                triggered_categories.append('context_driven')
                activation_decision['reasons'].append(f"File context suggests MCP tools (score: {context_score:.2f})")

        # Make activation decision
        if triggered_categories:
            activation_decision['should_activate'] = True
            activation_decision['confidence'] = min(1.0, len(triggered_categories) * 0.3)
            activation_decision['recommended_tools'] = self._select_optimal_tools(triggered_categories, user_message, file_context)
            activation_decision['expected_benefits'] = self._estimate_benefits(triggered_categories, user_message)

        return activation_decision

    def enhance_copilot_code_generation(self, user_message: str, file_context: Dict = None) -> Dict[str, Any]:
        """Vylepšuje GitHub Copilot generování kódu pomocí MCP nástrojů"""

        # Decide if tools should be activated
        activation = self.should_activate_copilot_tools(user_message, file_context)

        if not activation['should_activate']:
            return {
                'status': 'no_enhancement_needed',
                'original_approach': 'standard_generation',
                'reasoning': 'No clear benefit from MCP tools detected'
            }

        enhancement = {
            'status': 'enhanced',
            'activation_confidence': activation['confidence'],
            'tools_used': [],
            'code_analysis': {},
            'pattern_suggestions': [],
            'quality_recommendations': [],
            'token_optimization': {},
            'implementation_guidance': {}
        }

        # Get current file info for analysis
        current_file = file_context.get('current_file', '') if file_context else ''
        file_content = file_context.get('content', '') if file_context else ''

        if current_file and file_content:
            try:
                # Run MCP analysis
                analysis_result = self.mcp_tools.analyze_for_code_generation(
                    current_file,
                    file_content,
                    user_message
                )

                enhancement['tools_used'].append('code_analyzer')
                enhancement['code_analysis'] = analysis_result['analysis']
                enhancement['pattern_suggestions'] = analysis_result['code_suggestions']
                enhancement['token_optimization'] = {
                    'efficiency_score': analysis_result['token_efficiency_score'],
                    'estimated_savings': self._calculate_estimated_savings(analysis_result)
                }

                # Generate implementation guidance
                enhancement['implementation_guidance'] = self._create_implementation_guidance(
                    analysis_result, user_message, file_context
                )

                # Add quality recommendations
                enhancement['quality_recommendations'] = self._generate_quality_recommendations(
                    analysis_result, user_message
                )

            except Exception as e:
                enhancement['analysis_error'] = str(e)
                enhancement['fallback_guidance'] = self._get_fallback_guidance(user_message)

        return enhancement

    def _analyze_file_context_for_activation(self, file_context: Dict, message: str) -> float:
        """Analyzuje kontext souboru pro rozhodnutí o aktivaci"""
        score = 0.0

        # Check visible files for project patterns
        visible_files = file_context.get('visible_files', [])
        files_text = ' '.join(visible_files).lower()

        for context_type, indicators in self.project_context_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in files_text)
            if matches > 0:
                score += matches * 0.1

        # Check current file content
        current_content = file_context.get('content', '').lower()
        for context_type, indicators in self.project_context_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in current_content)
            if matches > 0:
                score += matches * 0.05

        # Boost score if message relates to existing patterns in file
        if current_content:
            if 'async def' in current_content and 'async' in message:
                score += 0.2
            if 'class ' in current_content and 'class' in message:
                score += 0.2
            if 'cache' in current_content and any(word in message for word in ['cache', 'store', 'save']):
                score += 0.3

        return min(1.0, score)

    def _select_optimal_tools(self, triggered_categories: List[str], user_message: str, file_context: Dict) -> List[str]:
        """Vybírá optimální nástroje na základě spouštěčů"""
        tools = ['code_analyzer']  # Always include basic analysis

        if 'pattern_reuse_opportunity' in triggered_categories:
            tools.append('pattern_matcher')

        if 'code_quality_enhancement' in triggered_categories:
            tools.extend(['quality_checker', 'optimization_suggester'])

        if 'token_optimization_needed' in triggered_categories:
            tools.append('token_optimizer')

        if 'project_specific_context' in triggered_categories:
            tools.append('project_context_analyzer')

        return list(set(tools))  # Remove duplicates

    def _estimate_benefits(self, triggered_categories: List[str], user_message: str) -> Dict[str, Any]:
        """Odhaduje přínosy použití MCP nástrojů"""
        benefits = {
            'token_savings_estimate': '0-20%',
            'quality_improvements': [],
            'development_speed_boost': 'Low',
            'code_consistency': 'Standard'
        }

        if 'pattern_reuse_opportunity' in triggered_categories:
            benefits['token_savings_estimate'] = '40-75%'
            benefits['development_speed_boost'] = 'High'
            benefits['quality_improvements'].append('Pattern consistency')

        if 'code_quality_enhancement' in triggered_categories:
            benefits['quality_improvements'].extend(['Error handling', 'Type safety', 'Documentation'])

        if 'token_optimization_needed' in triggered_categories:
            benefits['token_savings_estimate'] = '60-80%'
            benefits['code_consistency'] = 'High'

        if 'project_specific_context' in triggered_categories:
            benefits['quality_improvements'].append('Project-specific best practices')
            benefits['code_consistency'] = 'Very High'

        return benefits

    def _create_implementation_guidance(self, analysis_result: Dict, user_message: str, file_context: Dict) -> Dict[str, Any]:
        """Vytváří návod pro implementaci"""
        guidance = {
            'recommended_approach': 'standard',
            'pattern_to_use': None,
            'customization_steps': [],
            'quality_checkpoints': [],
            'testing_recommendations': []
        }

        # Determine recommended approach based on analysis
        if analysis_result.get('code_suggestions'):
            best_suggestion = analysis_result['code_suggestions'][0]
            guidance['recommended_approach'] = 'pattern_based'
            guidance['pattern_to_use'] = best_suggestion['pattern_name']
            guidance['customization_steps'] = best_suggestion['customization_hints']

        # Add quality checkpoints
        guidance['quality_checkpoints'] = [
            'Verify type hints are complete',
            'Ensure proper error handling',
            'Add comprehensive docstrings',
            'Include input validation where needed',
            'Test edge cases thoroughly'
        ]

        # Add testing recommendations
        if 'async' in user_message.lower():
            guidance['testing_recommendations'].append('Use pytest-asyncio for async function testing')
        if 'database' in user_message.lower():
            guidance['testing_recommendations'].append('Mock database operations in unit tests')
        if 'api' in user_message.lower():
            guidance['testing_recommendations'].append('Mock external API calls')

        return guidance

    def _calculate_estimated_savings(self, analysis_result: Dict) -> str:
        """Vypočítává odhadované úspory tokenů"""
        efficiency_score = analysis_result.get('token_efficiency_score', 0.5)

        if efficiency_score > 0.8:
            return "60-80% token reduction through pattern reuse"
        elif efficiency_score > 0.6:
            return "40-60% token reduction through optimization"
        elif efficiency_score > 0.4:
            return "20-40% token reduction through smart suggestions"
        else:
            return "10-20% token reduction through quality improvements"

    def _generate_quality_recommendations(self, analysis_result: Dict, user_message: str) -> List[str]:
        """Generuje doporučení pro kvalitu kódu"""
        recommendations = []

        analysis = analysis_result.get('analysis', {})

        # Add recommendations based on analysis
        if analysis.get('copilot_recommendations'):
            recommendations.extend(analysis['copilot_recommendations'])

        # Add message-specific recommendations
        message_lower = user_message.lower()
        if 'async' in message_lower:
            recommendations.append("🔄 Use proper async/await patterns and context managers")
        if 'database' in message_lower:
            recommendations.append("🛡️ Use parameterized queries to prevent SQL injection")
        if 'api' in message_lower:
            recommendations.append("⚡ Implement retry logic and proper timeout handling")
        if 'cache' in message_lower:
            recommendations.append("📦 Set appropriate TTL and handle cache invalidation")

        return recommendations

    def _get_fallback_guidance(self, user_message: str) -> Dict[str, Any]:
        """Poskytuje záložní návod při selhání analýzy"""
        return {
            'approach': 'Follow Python best practices',
            'recommendations': [
                'Use type hints for better code clarity',
                'Add proper error handling with specific exceptions',
                'Include comprehensive docstrings',
                'Follow PEP 8 style guidelines',
                'Add unit tests for new functionality'
            ],
            'patterns_to_consider': list(self.smart_recommendations.keys())
        }

    def create_copilot_prompt_enhancement(self, user_message: str, file_context: Dict = None) -> str:
        """Vytváří vylepšený prompt pro GitHub Copilot"""

        # Get enhancement data
        enhancement = self.enhance_copilot_code_generation(user_message, file_context)

        if enhancement['status'] == 'no_enhancement_needed':
            return user_message

        # Build enhanced prompt
        enhanced_parts = [user_message]

        # Add context information
        if file_context:
            project_context = self._detect_project_context(file_context)
            if project_context:
                enhanced_parts.append(f"\nProject context: {project_context}")

        # Add pattern suggestions
        if enhancement.get('pattern_suggestions'):
            best_pattern = enhancement['pattern_suggestions'][0]
            enhanced_parts.append(f"\nSuggested pattern: {best_pattern['pattern_name']} (saves {best_pattern['expected_token_savings']})")

        # Add quality requirements
        if enhancement.get('quality_recommendations'):
            quality_reqs = enhancement['quality_recommendations'][:2]  # Top 2 recommendations
            enhanced_parts.append(f"\nQuality requirements: {', '.join(quality_reqs)}")

        # Add implementation guidance
        if enhancement.get('implementation_guidance', {}).get('recommended_approach') == 'pattern_based':
            enhanced_parts.append(f"\nImplementation approach: Use {enhancement['implementation_guidance']['pattern_to_use']} pattern")

        return '\n'.join(enhanced_parts)

    def _detect_project_context(self, file_context: Dict) -> Optional[str]:
        """Detekuje kontext projektu pro prompt enhancement"""
        visible_files = file_context.get('visible_files', [])
        files_text = ' '.join(visible_files).lower()

        for context_type, indicators in self.project_context_indicators.items():
            if sum(1 for indicator in indicators if indicator in files_text) >= 2:
                return context_type

        return None

# Global instance pro GitHub Copilot
_auto_mcp = GitHubCopilotAutoMCP()

def enhance_copilot_response(user_message: str, file_context: Dict = None) -> Dict[str, Any]:
    """Hlavní funkce pro vylepšení GitHub Copilot odpovědi"""
    global _auto_mcp

    # Get enhancement
    enhancement = _auto_mcp.enhance_copilot_code_generation(user_message, file_context)

    # Create enhanced prompt
    enhanced_prompt = _auto_mcp.create_copilot_prompt_enhancement(user_message, file_context)

    return {
        'original_message': user_message,
        'enhanced_prompt': enhanced_prompt,
        'enhancement_data': enhancement,
        'tools_available': TOOLS_AVAILABLE,
        'activation_decision': _auto_mcp.should_activate_copilot_tools(user_message, file_context)
    }

def get_copilot_recommendations(user_message: str, file_context: Dict = None) -> List[str]:
    """Vrací doporučení pro GitHub Copilot"""
    global _auto_mcp

    enhancement = _auto_mcp.enhance_copilot_code_generation(user_message, file_context)

    recommendations = []

    if enhancement['status'] == 'enhanced':
        # Add pattern recommendations
        if enhancement.get('pattern_suggestions'):
            best_pattern = enhancement['pattern_suggestions'][0]
            recommendations.append(f"🔄 Use {best_pattern['pattern_name']} for {best_pattern['expected_token_savings']} token savings")

        # Add quality recommendations
        recommendations.extend(enhancement.get('quality_recommendations', [])[:3])

        # Add token optimization
        if enhancement.get('token_optimization', {}).get('efficiency_score', 0) > 0.7:
            recommendations.append("⚡ High token optimization potential detected")

    return recommendations

if __name__ == "__main__":
    # Test the enhanced auto-MCP system
    test_message = "Create an async function to process peptide research data with caching"
    test_context = {
        'current_file': 'research_processor.py',
        'content': 'import asyncio\nfrom dataclasses import dataclass\n\nclass ResearchEngine:\n    pass',
        'visible_files': ['biohacking_research_engine.py', 'local_ai_adapter.py', 'cache.py']
    }

    result = enhance_copilot_response(test_message, test_context)

    print("🤖 Enhanced GitHub Copilot Auto-MCP System Test:")
    print(f"Tools Available: {result['tools_available']}")
    print(f"Should Activate: {result['activation_decision']['should_activate']}")
    print(f"Confidence: {result['activation_decision']['confidence']:.2f}")
    print(f"Enhancement Status: {result['enhancement_data']['status']}")

    if result['enhancement_data']['status'] == 'enhanced':
        print(f"Token Efficiency: {result['enhancement_data'].get('token_optimization', {}).get('efficiency_score', 0):.2f}")
        print(f"Pattern Suggestions: {len(result['enhancement_data'].get('pattern_suggestions', []))}")

    recommendations = get_copilot_recommendations(test_message, test_context)
    print(f"\nRecommendations ({len(recommendations)}):")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
