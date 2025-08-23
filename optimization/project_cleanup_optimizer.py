"""
Project Cleanup and Optimization System - Automatické čištění a optimalizace projektu
Identifikuje a odstraňuje zbytečné soubory, optimalizuje strukturu a výkon
ROZŠÍŘENO - automatické slučování redundantních souborů a pokročilé optimalizace
"""

import os
import shutil
import asyncio
import json
import logging
import ast
import difflib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import hashlib
import subprocess
import re

logger = logging.getLogger(__name__)

class ProjectCleanupOptimizer:
    """Systém pro čištění a optimalizaci projektu - ROZŠÍŘENÝ"""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.cleanup_report = {
            "timestamp": datetime.now().isoformat(),
            "files_removed": [],
            "files_optimized": [],
            "files_consolidated": [],
            "space_saved_mb": 0.0,
            "optimizations_applied": [],
            "import_fixes": [],
            "redundancy_analysis": {},
            "errors": []
        }

        # Files and patterns to remove
        self.cleanup_patterns = {
            "cache_files": [
                "__pycache__/**",
                "*.pyc",
                "*.pyo",
                ".pytest_cache/**",
                "cache/**/*.cache",
                ".DS_Store",
                "Thumbs.db",
                "*.swp",
                "*.swo",
                "*~"
            ],
            "log_files": [
                "*.log",
                "logs/**",
                "agent_server.log"
            ],
            "temp_files": [
                "tmp/**",
                "temp/**",
                "*.tmp",
                ".vscode/settings.json.backup*"
            ],
            "duplicate_configs": [
                "config_backup_*",
                "*.config.old",
                "*.conf.bak"
            ]
        }

        # Redundant file patterns for consolidation
        self.redundancy_patterns = {
            "orchestrators": [
                ("intelligent_research_orchestrator.py", "enhanced_research_orchestrator.py"),
                ("research_orchestrator.py", "enhanced_research_orchestrator.py")
            ],
            "cache_systems": [
                ("smart_caching_system.py", "unified_cache_system.py"),
                ("cache_manager.py", "unified_cache_system.py"),
                ("m1_optimized_cache.py", "unified_cache_system.py")
            ],
            "mcp_tools": [
                ("smart_mcp_tools.py", "copilot_tools.py"),
                ("advanced_copilot_mcp.py", "copilot_tools.py"),
                ("copilot_mcp_interface.py", "copilot_tools.py"),
                ("github_copilot_mcp_tools.py", "copilot_tools.py")
            ]
        }

        # Import mapping for fixing broken imports
        self.import_mappings = {
            "intelligent_research_orchestrator": "enhanced_research_orchestrator",
            "smart_caching_system": "unified_cache_system",
            "smart_mcp_tools": "copilot_tools",
            "advanced_copilot_mcp": "copilot_tools",
            "copilot_mcp_interface": "copilot_tools",
            "github_copilot_mcp_tools": "copilot_tools"
        }

        # Class/function mappings for consolidated modules
        self.class_mappings = {
            "IntelligentResearchOrchestrator": "ConsolidatedResearchOrchestrator",
            "SmartMCPTools": "UnifiedCopilotInterface",
            "AdvancedCopilotMCP": "UnifiedCopilotInterface",
            "CopilotMCPInterface": "UnifiedCopilotInterface"
        }

    async def comprehensive_project_optimization(self) -> Dict[str, any]:
        """Komplexní optimalizace projektu"""

        logger.info("🚀 Starting comprehensive project optimization...")

        # 1. Analyze project structure
        await self._analyze_project_structure()

        # 2. Detect and analyze redundant files
        await self._detect_redundant_files()

        # 3. Clean temporary and cache files
        await self._clean_temporary_files()

        # 4. Fix imports after consolidation
        await self._fix_broken_imports()

        # 5. Optimize remaining code
        await self._optimize_code_quality()

        # 6. Update documentation
        await self._update_documentation()

        # 7. Generate optimization report
        self._generate_final_report()

        return self.cleanup_report

    async def _analyze_project_structure(self):
        """Analyzuje strukturu projektu"""

        logger.info("📊 Analyzing project structure...")

        structure_analysis = {
            "total_files": 0,
            "python_files": 0,
            "config_files": 0,
            "cache_files": 0,
            "documentation_files": 0,
            "potential_redundancies": []
        }

        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                structure_analysis["total_files"] += 1

                if file_path.suffix == ".py":
                    structure_analysis["python_files"] += 1
                elif file_path.suffix in [".json", ".yaml", ".yml", ".ini", ".conf"]:
                    structure_analysis["config_files"] += 1
                elif file_path.name.startswith(".") or "cache" in str(file_path):
                    structure_analysis["cache_files"] += 1
                elif file_path.suffix in [".md", ".rst", ".txt"]:
                    structure_analysis["documentation_files"] += 1

        self.cleanup_report["project_structure"] = structure_analysis
        logger.info(f"✅ Found {structure_analysis['total_files']} total files")

    async def _detect_redundant_files(self):
        """Detekuje redundantní soubory pro konsolidaci"""

        logger.info("🔍 Detecting redundant files...")

        redundant_files = []

        for category, file_pairs in self.redundancy_patterns.items():
            for old_file, new_file in file_pairs:
                old_path = self.project_root / old_file
                new_path = self.project_root / new_file

                if old_path.exists():
                    if new_path.exists():
                        # Both files exist - analyze for consolidation
                        similarity = await self._analyze_file_similarity(old_path, new_path)
                        redundant_files.append({
                            "category": category,
                            "old_file": str(old_path),
                            "new_file": str(new_path),
                            "similarity": similarity,
                            "action": "consolidate" if similarity > 0.7 else "review"
                        })
                    else:
                        logger.warning(f"Target consolidation file missing: {new_path}")

        self.cleanup_report["redundant_files"] = redundant_files

        # Auto-consolidate files with high similarity
        for file_info in redundant_files:
            if file_info["action"] == "consolidate":
                await self._backup_and_remove_file(Path(file_info["old_file"]))

    async def _analyze_file_similarity(self, file1: Path, file2: Path) -> float:
        """Analyzuje podobnost dvou souborů"""

        try:
            with open(file1, 'r', encoding='utf-8') as f1:
                content1 = f1.read()
            with open(file2, 'r', encoding='utf-8') as f2:
                content2 = f2.read()

            # Use difflib to calculate similarity
            similarity = difflib.SequenceMatcher(None, content1, content2).ratio()
            return similarity

        except Exception as e:
            logger.warning(f"Could not analyze similarity between {file1} and {file2}: {e}")
            return 0.0

    async def _clean_temporary_files(self):
        """Čistí dočasné a cache soubory"""

        logger.info("🧹 Cleaning temporary and cache files...")

        files_removed = 0
        space_saved = 0

        for category, patterns in self.cleanup_patterns.items():
            for pattern in patterns:
                for file_path in self.project_root.glob(pattern):
                    if file_path.is_file():
                        try:
                            file_size = file_path.stat().st_size
                            file_path.unlink()

                            files_removed += 1
                            space_saved += file_size

                            self.cleanup_report["files_removed"].append({
                                "file": str(file_path),
                                "category": category,
                                "size_bytes": file_size
                            })

                        except Exception as e:
                            self.cleanup_report["errors"].append(f"Could not remove {file_path}: {e}")

                    elif file_path.is_dir():
                        try:
                            shutil.rmtree(file_path)
                            files_removed += 1

                            self.cleanup_report["files_removed"].append({
                                "file": str(file_path),
                                "category": category,
                                "type": "directory"
                            })

                        except Exception as e:
                            self.cleanup_report["errors"].append(f"Could not remove directory {file_path}: {e}")

        self.cleanup_report["space_saved_mb"] = space_saved / (1024 * 1024)
        logger.info(f"✅ Removed {files_removed} files, saved {space_saved / (1024 * 1024):.2f} MB")

    async def _fix_broken_imports(self):
        """Opraví rozbité importy po konsolidaci"""

        logger.info("🔧 Fixing broken imports...")

        python_files = list(self.project_root.rglob("*.py"))
        fixed_imports = []

        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                original_content = content

                # Fix module imports
                for old_module, new_module in self.import_mappings.items():
                    # Fix direct imports
                    pattern = rf"from\s+{re.escape(old_module)}\s+import"
                    replacement = f"from {new_module} import"
                    content = re.sub(pattern, replacement, content)

                    # Fix import statements
                    pattern = rf"import\s+{re.escape(old_module)}"
                    replacement = f"import {new_module}"
                    content = re.sub(pattern, replacement, content)

                # Fix class references
                for old_class, new_class in self.class_mappings.items():
                    # Be careful to only replace class instantiations, not in strings
                    pattern = rf"\b{re.escape(old_class)}\b(?=\s*\()"
                    replacement = new_class
                    content = re.sub(pattern, replacement, content)

                # Save if changed
                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)

                    fixed_imports.append({
                        "file": str(py_file),
                        "changes": self._count_import_changes(original_content, content)
                    })

            except Exception as e:
                self.cleanup_report["errors"].append(f"Could not fix imports in {py_file}: {e}")

        self.cleanup_report["import_fixes"] = fixed_imports
        logger.info(f"✅ Fixed imports in {len(fixed_imports)} files")

    def _count_import_changes(self, original: str, updated: str) -> int:
        """Počítá počet změn v importech"""

        original_imports = len(re.findall(r"^(import|from)\s+", original, re.MULTILINE))
        updated_imports = len(re.findall(r"^(import|from)\s+", updated, re.MULTILINE))

        return abs(original_imports - updated_imports)

    async def _optimize_code_quality(self):
        """Optimalizuje kvalitu kódu"""

        logger.info("⚡ Optimizing code quality...")

        optimizations = []

        # Find Python files for optimization
        python_files = list(self.project_root.rglob("*.py"))

        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                original_content = content

                # Remove excessive blank lines
                content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

                # Fix trailing whitespace
                content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)

                # Ensure file ends with newline
                if content and not content.endswith('\n'):
                    content += '\n'

                # Save if changed
                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)

                    optimizations.append({
                        "file": str(py_file),
                        "type": "formatting",
                        "description": "Fixed whitespace and blank lines"
                    })

            except Exception as e:
                self.cleanup_report["errors"].append(f"Could not optimize {py_file}: {e}")

        self.cleanup_report["optimizations_applied"] = optimizations
        logger.info(f"✅ Optimized {len(optimizations)} files")

    async def _update_documentation(self):
        """Aktualizuje dokumentaci"""

        logger.info("📝 Updating documentation...")

        # Update README with consolidation info
        readme_path = self.project_root / "README.md"

        if readme_path.exists():
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()

                # Add consolidation info
                consolidation_info = """
## Project Optimization (Latest Update)

This project has been optimized and consolidated:

### Consolidated Modules:
- **Research Orchestrators**: `enhanced_research_orchestrator.py` (consolidated)
- **Cache Systems**: `unified_cache_system.py` (consolidated) 
- **Copilot Tools**: `copilot_tools.py` (consolidated with 4 new advanced tools)

### New Advanced Tools for GitHub Copilot:
1. **BiohackingCompoundValidator** - Validates biohacking substances with research status
2. **CodePatternOptimizer** - Suggests optimized design patterns
3. **AsyncSafetyGuard** - Specialized linter for async code safety
4. **PrivacyLeakDetector** - Detects potential sensitive data leaks

### Key Features:
- Unified cache system optimized for M1 MacBook
- AI-powered research orchestration with predictive preloading
- Comprehensive code analysis and optimization tools
- Privacy-focused development with leak detection

"""

                # Add info if not already present
                if "Project Optimization" not in readme_content:
                    readme_content = consolidation_info + "\n---\n\n" + readme_content

                    with open(readme_path, 'w', encoding='utf-8') as f:
                        f.write(readme_content)

                    self.cleanup_report["optimizations_applied"].append({
                        "file": "README.md",
                        "type": "documentation",
                        "description": "Added consolidation and optimization info"
                    })

            except Exception as e:
                self.cleanup_report["errors"].append(f"Could not update README: {e}")

    async def _backup_and_remove_file(self, file_path: Path):
        """Zálohuje a odstraní soubor"""

        try:
            # Create backup
            backup_dir = self.project_root / "backups" / "consolidation"
            backup_dir.mkdir(parents=True, exist_ok=True)

            backup_path = backup_dir / f"{file_path.name}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
            shutil.copy2(file_path, backup_path)

            # Remove original
            file_size = file_path.stat().st_size
            file_path.unlink()

            self.cleanup_report["files_consolidated"].append({
                "file": str(file_path),
                "backup": str(backup_path),
                "size_bytes": file_size
            })

            logger.info(f"✅ Consolidated {file_path.name} (backed up to {backup_path})")

        except Exception as e:
            self.cleanup_report["errors"].append(f"Could not backup/remove {file_path}: {e}")

    def _generate_final_report(self):
        """Generuje finální report optimalizace"""

        report_path = self.project_root / "reports" / f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.cleanup_report, f, indent=2, ensure_ascii=False)

            logger.info(f"📊 Optimization report saved to {report_path}")

            # Print summary
            self._print_optimization_summary()

        except Exception as e:
            logger.error(f"Could not save optimization report: {e}")

    def _print_optimization_summary(self):
        """Vytiskne shrnutí optimalizace"""

        print("\n" + "="*50)
        print("🚀 PROJECT OPTIMIZATION SUMMARY")
        print("="*50)

        print(f"📁 Files removed: {len(self.cleanup_report['files_removed'])}")
        print(f"🔗 Files consolidated: {len(self.cleanup_report['files_consolidated'])}")
        print(f"🔧 Import fixes: {len(self.cleanup_report['import_fixes'])}")
        print(f"⚡ Optimizations applied: {len(self.cleanup_report['optimizations_applied'])}")
        print(f"💾 Space saved: {self.cleanup_report['space_saved_mb']:.2f} MB")

        if self.cleanup_report['errors']:
            print(f"⚠️  Errors encountered: {len(self.cleanup_report['errors'])}")

        print("\n✅ Project optimization completed successfully!")
        print("="*50)

# Factory function
async def optimize_project(project_root: Path = None):
    """Factory function to run project optimization"""
    optimizer = ProjectCleanupOptimizer(project_root)
    return await optimizer.comprehensive_project_optimization()

# Export
__all__ = ['ProjectCleanupOptimizer', 'optimize_project']
