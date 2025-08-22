#!/usr/bin/env python3
"""
GitHub Copilot MCP Tools - Optimalizované nástroje pro GitHub Copilot agenta
Zaměřeno na zlepšení kvality kódu a snížení spotřeby tokenů při programování
"""

import ast
import re
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

class CopilotCodeAnalyzer:
    """Analyzátor kódu optimalizovaný pro GitHub Copilot agenta"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.patterns_db = self.project_root / "cache" / "copilot_patterns.db"
        self.init_patterns_db()

        # Specific patterns for this biohacking research project
        self.project_patterns = self._analyze_project_patterns()

    def init_patterns_db(self):
        """Inicializuje optimalizovanou databázi pro GitHub Copilot patterns"""
        self.patterns_db.parent.mkdir(exist_ok=True)
        conn = sqlite3.connect(self.patterns_db)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS copilot_patterns (
                pattern_hash TEXT PRIMARY KEY,
                pattern_type TEXT,
                code_template TEXT,
                usage_frequency REAL DEFAULT 1.0,
                success_rate REAL DEFAULT 1.0,
                token_efficiency REAL DEFAULT 1.0,
                project_context TEXT,
                last_used DATETIME,
                performance_score REAL DEFAULT 1.0
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS code_quality_metrics (
                file_path TEXT PRIMARY KEY,
                complexity_score REAL,
                maintainability_index REAL,
                code_patterns TEXT,
                optimization_opportunities TEXT,
                last_analyzed DATETIME
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS copilot_learning (
                interaction_id TEXT PRIMARY KEY,
                user_intent TEXT,
                context_hash TEXT,
                suggested_approach TEXT,
                actual_implementation TEXT,
                success_metrics TEXT,
                token_savings REAL,
                timestamp DATETIME
            )
        ''')
        conn.commit()
        conn.close()

    def _analyze_project_patterns(self) -> Dict[str, Any]:
        """Analyzuje patterns specifické pro tento biohacking research projekt"""
        patterns = {
            'async_research_patterns': {
                'template': '''async def {function_name}(self, {params}):
    """Research function with caching and error handling"""
    cache_key = f"{cache_prefix}_{hashlib.md5(str({params}).encode()).hexdigest()}"
    
    # Check cache first
    cached = await self.cache.get(cache_key)
    if cached:
        return cached
    
    try:
        # Actual research logic
        result = await self._perform_research({params})
        
        # Cache successful results
        await self.cache.set(cache_key, result, ttl=3600)
        return result
        
    except Exception as e:
        logger.error(f"Research failed: {e}")
        raise ResearchError(f"Failed to {function_name}: {e}")''',
                'use_cases': ['research functions', 'data processing', 'API calls'],
                'token_savings': 0.75
            },

            'data_validation_pattern': {
                'template': '''@dataclass
class {class_name}:
    """Validated data structure for {purpose}"""
    {fields}
    
    def __post_init__(self):
        """Validate data after initialization"""
        self._validate_fields()
    
    def _validate_fields(self):
        """Validate all fields with proper error messages"""
        {validation_logic}''',
                'use_cases': ['data structures', 'API models', 'configuration'],
                'token_savings': 0.60
            },

            'ai_integration_pattern': {
                'template': '''class {class_name}:
    """AI-enabled component with proper resource management"""
    
    async def __aenter__(self):
        self.ai_client = await self._init_ai_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'ai_client'):
            await self.ai_client.close()
    
    async def {method_name}(self, {params}):
        """AI-powered method with token optimization"""
        # Optimize prompt for token efficiency
        optimized_prompt = self._optimize_prompt({params})
        
        try:
            response = await self.ai_client.generate(optimized_prompt)
            return self._process_ai_response(response)
        except Exception as e:
            logger.error(f"AI operation failed: {e}")
            return self._fallback_response({params})''',
                'use_cases': ['AI integrations', 'LLM calls', 'async AI operations'],
                'token_savings': 0.80
            },

            'database_operation_pattern': {
                'template': '''async def {operation_name}(self, {params}):
    """Optimized database operation with connection pooling"""
    async with self.db_pool.acquire() as conn:
        try:
            async with conn.transaction():
                # Prepared statement for security and performance
                query = """
                {sql_query}
                """
                result = await conn.fetchrow(query, {query_params})
                
                # Log successful operations
                logger.debug(f"{operation_name} completed successfully")
                return result
                
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise DatabaseError(f"Failed {operation_name}: {e}")''',
                'use_cases': ['database operations', 'data persistence', 'queries'],
                'token_savings': 0.65
            }
        }
        return patterns

    def analyze_for_copilot_suggestions(self, file_path: str, file_content: str, user_intent: str) -> Dict[str, Any]:
        """Hlavní analýza pro GitHub Copilot návrhy"""
        try:
            # Parse the code
            tree = ast.parse(file_content) if file_path.endswith('.py') else None

            analysis = {
                'file_type': self._detect_file_type(file_path),
                'current_patterns': self._extract_current_patterns(file_content, tree),
                'optimization_opportunities': self._find_optimization_opportunities(file_content, tree),
                'suggested_patterns': self._suggest_relevant_patterns(user_intent, file_content),
                'quality_metrics': self._calculate_quality_metrics(file_content, tree),
                'token_optimization': self._analyze_token_optimization_potential(user_intent, file_content),
                'copilot_recommendations': []
            }

            # Generate specific recommendations for GitHub Copilot
            analysis['copilot_recommendations'] = self._generate_copilot_recommendations(analysis, user_intent)

            # Store learning data
            self._store_copilot_interaction(user_intent, analysis)

            return analysis

        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'fallback_suggestions': self._get_fallback_suggestions(user_intent)
            }

    def _suggest_relevant_patterns(self, user_intent: str, file_content: str) -> List[Dict[str, Any]]:
        """Navrhuje relevantní patterns na základě záměru a kontextu"""
        suggestions = []
        intent_lower = user_intent.lower()

        # Match patterns to user intent
        for pattern_name, pattern_data in self.project_patterns.items():
            if self._pattern_matches_intent(pattern_name, intent_lower, file_content):
                suggestions.append({
                    'pattern_name': pattern_name,
                    'template': pattern_data['template'],
                    'use_cases': pattern_data['use_cases'],
                    'token_savings': pattern_data['token_savings'],
                    'customization_hints': self._get_customization_hints(pattern_name, user_intent, file_content)
                })

        return suggestions

    def _pattern_matches_intent(self, pattern_name: str, intent: str, file_content: str) -> bool:
        """Určuje, zda pattern odpovídá záměru uživatele"""
        pattern_keywords = {
            'async_research_patterns': ['async', 'research', 'data', 'api', 'fetch', 'process'],
            'data_validation_pattern': ['dataclass', 'validate', 'model', 'structure', 'data'],
            'ai_integration_pattern': ['ai', 'llm', 'generate', 'model', 'chat', 'embedding'],
            'database_operation_pattern': ['database', 'db', 'query', 'select', 'insert', 'update']
        }

        keywords = pattern_keywords.get(pattern_name, [])
        has_intent_match = any(keyword in intent for keyword in keywords)

        # Also consider current file context
        has_context_match = any(keyword in file_content.lower() for keyword in keywords[:2])

        return has_intent_match or has_context_match

    def _generate_copilot_recommendations(self, analysis: Dict, user_intent: str) -> List[str]:
        """Generuje specifické doporučení pro GitHub Copilot"""
        recommendations = []

        # Pattern-based recommendations
        if analysis['suggested_patterns']:
            best_pattern = max(analysis['suggested_patterns'], key=lambda p: p['token_savings'])
            recommendations.append(
                f"🚀 Use {best_pattern['pattern_name']} pattern for {best_pattern['token_savings']:.0%} token savings"
            )

        # Quality-based recommendations
        quality = analysis['quality_metrics']
        if quality['complexity_score'] > 7:
            recommendations.append("⚠️ Consider breaking down complex functions for better maintainability")

        # Optimization recommendations
        for opt in analysis['optimization_opportunities'][:2]:
            recommendations.append(f"⚡ {opt}")

        # Token optimization recommendations
        token_opt = analysis['token_optimization']
        if token_opt['reusable_components'] > 0:
            recommendations.append(
                f"💡 Found {token_opt['reusable_components']} reusable components - leverage existing code"
            )

        return recommendations

    def _calculate_quality_metrics(self, file_content: str, tree: Optional[ast.AST]) -> Dict[str, Any]:
        """Vypočítává metriky kvality kódu"""
        metrics = {
            'complexity_score': 1,
            'maintainability_index': 100,
            'documentation_ratio': 0,
            'type_hints_coverage': 0,
            'error_handling_score': 0
        }

        if tree:
            # Calculate complexity
            complexity = self._calculate_cyclomatic_complexity(tree)
            metrics['complexity_score'] = complexity

            # Calculate maintainability index (simplified without numpy)
            loc = len(file_content.split('\n'))
            import math
            metrics['maintainability_index'] = max(0, int(171 - 5.2 * math.log(max(1, loc)) - 0.23 * complexity))

            # Documentation ratio
            docstrings = [node for node in ast.walk(tree) if isinstance(node, ast.Expr)
                         and isinstance(node.value, (ast.Str, ast.Constant))]
            functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if functions:
                metrics['documentation_ratio'] = int(len(docstrings) / len(functions)) if len(functions) > 0 else 0

            # Type hints coverage
            typed_functions = [node for node in ast.walk(tree)
                             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                             and node.returns is not None]
            if functions:
                metrics['type_hints_coverage'] = int(len(typed_functions) / len(functions)) if len(functions) > 0 else 0

        return metrics

    def _analyze_token_optimization_potential(self, user_intent: str, file_content: str) -> Dict[str, Any]:
        """Analyzuje potenciál pro optimalizaci tokenů"""
        return {
            'reusable_components': self._count_reusable_patterns(file_content),
            'template_opportunities': self._find_template_opportunities(user_intent, file_content),
            'redundancy_score': self._calculate_redundancy_score(file_content),
            'optimization_potential': self._estimate_optimization_potential(user_intent, file_content)
        }

    def _store_copilot_interaction(self, user_intent: str, analysis: Dict):
        """Ukládá interakci pro machine learning"""
        conn = sqlite3.connect(self.patterns_db)
        interaction_id = hashlib.md5(f"{user_intent}_{datetime.now()}".encode()).hexdigest()

        conn.execute(
            "INSERT OR REPLACE INTO copilot_learning VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                interaction_id,
                user_intent,
                hashlib.md5(str(analysis).encode()).hexdigest(),
                str(analysis.get('suggested_patterns', [])),
                '',  # actual_implementation - to be filled later
                str(analysis.get('quality_metrics', {})),
                analysis.get('token_optimization', {}).get('optimization_potential', 0),
                datetime.now().isoformat()
            )
        )
        conn.commit()
        conn.close()

# Simplified helper functions to avoid numpy dependency
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Simplified cyclomatic complexity calculation"""
        complexity = 1  # Base complexity
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor,
                               ast.ExceptHandler, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity

    def _count_reusable_patterns(self, file_content: str) -> int:
        """Počítá opakující se patterns v kódu"""
        # Simple heuristic - look for repeated function signatures, class patterns, etc.
        patterns = re.findall(r'def \w+\([^)]*\):', file_content)
        patterns += re.findall(r'class \w+\([^)]*\):', file_content)
        return len(set(patterns)) if len(patterns) > len(set(patterns)) else 0

    def _find_template_opportunities(self, user_intent: str, file_content: str) -> int:
        """Hledá příležitosti pro template použití"""
        # Count how many of our project patterns could be applied
        applicable_patterns = 0
        for pattern_name in self.project_patterns:
            if self._pattern_matches_intent(pattern_name, user_intent.lower(), file_content):
                applicable_patterns += 1
        return applicable_patterns

    def _calculate_redundancy_score(self, file_content: str) -> float:
        """Vypočítává skóre redundance v kódu"""
        lines = file_content.split('\n')
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(lines) == 0:
            return 0.0
        return 1.0 - (len(unique_lines) / len(lines))

    def _estimate_optimization_potential(self, user_intent: str, file_content: str) -> float:
        """Odhaduje potenciál optimalizace (0-1)"""
        factors = []

        # Pattern reuse potential
        reusable = self._count_reusable_patterns(file_content)
        factors.append(min(1.0, reusable / 3))

        # Template opportunities
        template_ops = self._find_template_opportunities(user_intent, file_content)
        factors.append(min(1.0, template_ops / 2))

        # Redundancy
        redundancy = self._calculate_redundancy_score(file_content)
        factors.append(redundancy)

        return sum(factors) / len(factors) if factors else 0.0

    def _detect_file_type(self, file_path: str) -> str:
        """Detekuje typ souboru"""
        if file_path.endswith('.py'):
            return 'python'
        elif file_path.endswith('.js'):
            return 'javascript'
        elif file_path.endswith('.md'):
            return 'markdown'
        elif file_path.endswith('.json'):
            return 'json'
        else:
            return 'unknown'

    def _extract_current_patterns(self, file_content: str, tree: Optional[ast.AST]) -> List[str]:
        """Extrahuje současné patterns v souboru"""
        patterns = []

        if 'async def' in file_content:
            patterns.append('async_functions')
        if 'dataclass' in file_content:
            patterns.append('dataclass_usage')
        if 'try:' in file_content and 'except' in file_content:
            patterns.append('error_handling')
        if 'class ' in file_content and '__enter__' in file_content:
            patterns.append('context_manager')
        if 'cache' in file_content.lower():
            patterns.append('caching')
        if 'logger' in file_content:
            patterns.append('logging')

        return patterns

    def _find_optimization_opportunities(self, file_content: str, tree: Optional[ast.AST]) -> List[str]:
        """Hledá příležitosti pro optimalizaci"""
        opportunities = []

        # Check for common optimization patterns
        if 'for ' in file_content and 'append(' in file_content:
            opportunities.append("Consider using list comprehension instead of for loop with append")

        if 'open(' in file_content and 'with ' not in file_content:
            opportunities.append("Use context managers (with statement) for file operations")

        if re.search(r'def \w+\([^)]*\):.*def \w+\([^)]*\):', file_content, re.DOTALL):
            if file_content.count('def ') > 5:
                opportunities.append("Consider breaking large files into smaller modules")

        if 'cache' not in file_content.lower() and ('research' in file_content.lower() or 'api' in file_content.lower()):
            opportunities.append("Consider adding caching for expensive operations")

        return opportunities

    def _get_fallback_suggestions(self, user_intent: str) -> List[str]:
        """Poskytuje záložní návrhy při selhání analýzy"""
        return [
            "Follow Python PEP 8 style guidelines",
            "Add type hints for better code clarity",
            "Include proper error handling",
            "Add docstrings to functions and classes",
            "Consider using dataclasses for data structures"
        ]

    def _get_customization_hints(self, pattern_name: str, user_intent: str, file_content: str) -> List[str]:
        """Poskytuje hinty pro přizpůsobení patterns"""
        hints_map = {
            'async_research_patterns': [
                "Replace {function_name} with descriptive function name",
                "Customize cache TTL based on data freshness requirements",
                "Add specific exception types for better error handling"
            ],
            'data_validation_pattern': [
                "Add field-specific validation logic",
                "Consider using Pydantic for advanced validation",
                "Add computed properties if needed"
            ],
            'ai_integration_pattern': [
                "Customize prompt optimization for your specific AI model",
                "Add retry logic for transient failures",
                "Implement proper token counting and rate limiting"
            ],
            'database_operation_pattern': [
                "Use parameterized queries to prevent SQL injection",
                "Add connection pooling configuration",
                "Implement proper transaction management"
            ]
        }
        return hints_map.get(pattern_name, [])

class CopilotMCPInterface:
    """Optimalizované rozhraní pro GitHub Copilot MCP nástroje"""

    def __init__(self, project_root: str = "/Users/vojtechhamada/PycharmProjects/PythonProject2"):
        self.project_root = project_root
        self.code_analyzer = CopilotCodeAnalyzer(project_root)

    def analyze_for_code_generation(self, file_path: str, file_content: str, user_request: str) -> Dict[str, Any]:
        """Hlavní funkce pro analýzu před generováním kódu"""
        analysis = self.code_analyzer.analyze_for_copilot_suggestions(file_path, file_content, user_request)

        # Create actionable recommendations
        result = {
            'analysis': analysis,
            'action_plan': self._create_action_plan(analysis, user_request),
            'code_suggestions': self._generate_code_suggestions(analysis, user_request),
            'token_efficiency_score': self._calculate_token_efficiency(analysis)
        }

        return result

    def _create_action_plan(self, analysis: Dict, user_request: str) -> Dict[str, Any]:
        """Vytváří akční plán pro implementaci"""
        plan = {
            'approach': 'standard',
            'recommended_patterns': [],
            'implementation_steps': [],
            'quality_checkpoints': []
        }

        # Determine approach based on analysis
        if analysis.get('suggested_patterns'):
            plan['approach'] = 'pattern_reuse'
            plan['recommended_patterns'] = [p['pattern_name'] for p in analysis['suggested_patterns']]

        # Add implementation steps
        plan['implementation_steps'] = [
            "1. Choose appropriate pattern template",
            "2. Customize template for specific use case",
            "3. Add proper error handling and logging",
            "4. Include type hints and documentation",
            "5. Test implementation thoroughly"
        ]

        return plan

    def _generate_code_suggestions(self, analysis: Dict, user_request: str) -> List[Dict[str, Any]]:
        """Generuje konkrétní návrhy kódu"""
        suggestions = []

        for pattern in analysis.get('suggested_patterns', []):
            suggestions.append({
                'pattern_name': pattern['pattern_name'],
                'code_template': pattern['template'],
                'customization_hints': pattern['customization_hints'],
                'expected_token_savings': f"{pattern['token_savings']:.0%}"
            })

        return suggestions

    def _calculate_token_efficiency(self, analysis: Dict) -> float:
        """Vypočítává celkovou efektivitu tokenů"""
        base_efficiency = 0.5

        # Pattern reuse bonus
        patterns_bonus = len(analysis.get('suggested_patterns', [])) * 0.15

        # Quality bonus
        quality = analysis.get('quality_metrics', {})
        quality_bonus = (quality.get('maintainability_index', 50) / 100) * 0.2

        # Optimization potential
        token_opt = analysis.get('token_optimization', {})
        opt_bonus = token_opt.get('optimization_potential', 0) * 0.15

        total_efficiency = base_efficiency + patterns_bonus + quality_bonus + opt_bonus
        return min(1.0, total_efficiency)

# Global instance pro GitHub Copilot
_copilot_tools = CopilotMCPInterface()

def get_copilot_tools():
    """Vrací optimalizované nástroje pro GitHub Copilot"""
    return _copilot_tools
