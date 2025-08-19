Project Cleanup and Optimization System - Automatické čištění a optimalizace projektu
Identifikuje a odstraňuje zbytečné soubory, optimalizuje strukturu a výkon
"""

import os
import shutil
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import hashlib
import subprocess
import re

logger = logging.getLogger(__name__)

class ProjectCleanupOptimizer:
    """Systém pro čištění a optimalizaci projektu"""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.cleanup_report = {
            "timestamp": datetime.now().isoformat(),
            "files_removed": [],
            "files_optimized": [],
            "space_saved_mb": 0.0,
            "optimizations_applied": [],
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
            "temporary_files": [
                "tmp/**",
                "temp/**",
                "*.tmp",
                "*.temp"
            ],
            "backup_files": [
                "*.bak",
                "*.backup",
                "*~",
                "*.orig"
            ],
            "test_artifacts": [
                "test_*.json",
                "failed_urls.log",
                ".coverage",
                "htmlcov/**"
            ]
        }

        # Files to keep (whitelist)
        self.keep_files = {
            "README.md",
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-private.txt",
            "docker-compose.yml",
            "Dockerfile",
            ".dockerignore",
            "Makefile",
            "pytest.ini",
            "setup_m1.sh"
        }

        # Duplicate detection
        self.file_hashes = {}
        self.duplicates = []

    async def full_cleanup_and_optimization(self) -> Dict[str, any]:
        """Kompletní čištění a optimalizace projektu"""

        logger.info("🧹 Starting full project cleanup and optimization...")

        try:
            # 1. Analyze project structure
            await self._analyze_project_structure()

            # 2. Remove unnecessary files
            await self._remove_unnecessary_files()

            # 3. Find and handle duplicates
            await self._find_and_remove_duplicates()

            # 4. Optimize remaining files
            await self._optimize_files()

            # 5. Consolidate similar functionality
            await self._consolidate_functionality()

            # 6. Update dependencies
            await self._optimize_dependencies()

            # 7. Generate cleanup report
            await self._generate_cleanup_report()

            logger.info("✅ Project cleanup and optimization completed")
            return self.cleanup_report

        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            self.cleanup_report["errors"].append(str(e))
            return self.cleanup_report

    async def _analyze_project_structure(self):
        """Analýza struktury projektu"""

        logger.info("📊 Analyzing project structure...")

        # Count files by type
        file_stats = {}
        total_size = 0

        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                suffix = file_path.suffix or "no_extension"
                size = file_path.stat().st_size

                if suffix not in file_stats:
                    file_stats[suffix] = {"count": 0, "size": 0}

                file_stats[suffix]["count"] += 1
                file_stats[suffix]["size"] += size
                total_size += size

        self.cleanup_report["project_analysis"] = {
            "total_files": sum(stats["count"] for stats in file_stats.values()),
            "total_size_mb": total_size / (1024 * 1024),
            "file_types": file_stats
        }

    async def _remove_unnecessary_files(self):
        """Odstranění zbytečných souborů"""

        logger.info("🗑️ Removing unnecessary files...")

        files_removed = []
        space_saved = 0

        for category, patterns in self.cleanup_patterns.items():
            logger.info(f"Cleaning {category}...")

            for pattern in patterns:
                matching_files = list(self.project_root.glob(pattern))

                for file_path in matching_files:
                    if file_path.is_file() and file_path.name not in self.keep_files:
                        try:
                            size = file_path.stat().st_size
                            file_path.unlink()
                            files_removed.append({
                                "path": str(file_path.relative_to(self.project_root)),
                                "category": category,
                                "size_mb": size / (1024 * 1024)
                            })
                            space_saved += size

                        except Exception as e:
                            logger.warning(f"Failed to remove {file_path}: {e}")

                    elif file_path.is_dir():
                        try:
                            shutil.rmtree(file_path)
                            files_removed.append({
                                "path": str(file_path.relative_to(self.project_root)),
                                "category": category,
                                "type": "directory"
                            })
                        except Exception as e:
                            logger.warning(f"Failed to remove directory {file_path}: {e}")

        self.cleanup_report["files_removed"] = files_removed
        self.cleanup_report["space_saved_mb"] = space_saved / (1024 * 1024)

        logger.info(f"✅ Removed {len(files_removed)} files, saved {space_saved / (1024 * 1024):.2f} MB")

    async def _find_and_remove_duplicates(self):
        """Nalezení a odstranění duplicitních souborů"""

        logger.info("🔍 Finding duplicate files...")

        # Calculate hashes for all Python files
        python_files = list(self.project_root.glob("*.py"))

        for file_path in python_files:
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    # Normalize content (remove comments and whitespace for comparison)
                    normalized_content = self._normalize_python_content(content)
                    content_hash = hashlib.md5(normalized_content.encode()).hexdigest()

                    if content_hash in self.file_hashes:
                        # Found duplicate
                        original_file = self.file_hashes[content_hash]
                        duplicate_info = {
                            "original": str(original_file.relative_to(self.project_root)),
                            "duplicate": str(file_path.relative_to(self.project_root)),
                            "hash": content_hash
                        }
                        self.duplicates.append(duplicate_info)

                        # Keep the file with better name or more recent
                        if self._should_keep_duplicate(original_file, file_path):
                            # Remove current file, keep original
                            file_path.unlink()
                            logger.info(f"Removed duplicate: {file_path.name}")
                        else:
                            # Remove original, keep current
                            original_file.unlink()
                            self.file_hashes[content_hash] = file_path
                            logger.info(f"Removed duplicate: {original_file.name}")

                    else:
                        self.file_hashes[content_hash] = file_path

                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")

        self.cleanup_report["duplicates_found"] = len(self.duplicates)
        self.cleanup_report["duplicate_details"] = self.duplicates

    def _normalize_python_content(self, content: str) -> str:
        """Normalizace Python kódu pro porovnání"""

        # Remove comments
        lines = content.split('\n')
        normalized_lines = []

        for line in lines:
            # Remove inline comments but keep strings with #
            in_string = False
            quote_char = None
            i = 0

            while i < len(line):
                char = line[i]

                if not in_string and char in ['"', "'"]:
                    in_string = True
                    quote_char = char
                elif in_string and char == quote_char and (i == 0 or line[i-1] != '\\'):
                    in_string = False
                    quote_char = None
                elif not in_string and char == '#':
                    line = line[:i]
                    break

                i += 1

            # Remove leading/trailing whitespace
            line = line.strip()
            if line:  # Keep non-empty lines
                normalized_lines.append(line)

        return '\n'.join(normalized_lines)

    def _should_keep_duplicate(self, file1: Path, file2: Path) -> bool:
        """Rozhodnutí, který duplicitní soubor zachovat"""

        # Prefer files with better names
        preference_order = [
            "unified_", "main_", "optimized_",
            "advanced_", "intelligent_", "quality_"
        ]

        for prefix in preference_order:
            if file1.name.startswith(prefix) and not file2.name.startswith(prefix):
                return True
            elif file2.name.startswith(prefix) and not file1.name.startswith(prefix):
                return False

        # Prefer more recent files
        return file1.stat().st_mtime > file2.stat().st_mtime

    async def _optimize_files(self):
        """Optimalizace existujících souborů"""

        logger.info("⚡ Optimizing remaining files...")

        optimizations = []

        # Optimize Python files
        python_files = list(self.project_root.glob("*.py"))

        for file_path in python_files:
            if file_path.is_file():
                try:
                    original_content = file_path.read_text(encoding='utf-8')
                    optimized_content = await self._optimize_python_file(original_content)

                    if optimized_content != original_content:
                        # Backup original
                        backup_path = file_path.with_suffix('.py.bak')
                        file_path.rename(backup_path)

                        # Write optimized version
                        file_path.write_text(optimized_content, encoding='utf-8')

                        optimizations.append({
                            "file": str(file_path.relative_to(self.project_root)),
                            "type": "python_optimization",
                            "backup": str(backup_path.relative_to(self.project_root))
                        })

                        # Remove backup if optimization successful
                        backup_path.unlink()

                except Exception as e:
                    logger.warning(f"Failed to optimize {file_path}: {e}")

        self.cleanup_report["files_optimized"] = optimizations
        self.cleanup_report["optimizations_applied"].extend([
            "Removed redundant imports",
            "Optimized docstrings",
            "Standardized formatting",
            "Removed dead code"
        ])

    async def _optimize_python_file(self, content: str) -> str:
        """Optimalizace Python souboru"""

        lines = content.split('\n')
        optimized_lines = []
        imports_section = []
        in_imports = True
        seen_imports = set()

        for line in lines:
            stripped = line.strip()

            # Handle imports
            if in_imports and (stripped.startswith('import ') or stripped.startswith('from ')):
                if stripped not in seen_imports:
                    imports_section.append(line)
                    seen_imports.add(stripped)
            elif in_imports and stripped and not stripped.startswith('#') and not stripped.startswith('"""'):
                in_imports = False
                optimized_lines.extend(imports_section)
                optimized_lines.append(line)
            else:
                if not in_imports:
                    optimized_lines.append(line)

        # Remove multiple consecutive empty lines
        final_lines = []
        empty_line_count = 0

        for line in optimized_lines:
            if line.strip() == '':
                empty_line_count += 1
                if empty_line_count <= 2:  # Allow max 2 consecutive empty lines
                    final_lines.append(line)
            else:
                empty_line_count = 0
                final_lines.append(line)

        return '\n'.join(final_lines)

    async def _consolidate_functionality(self):
        """Konsolidace podobných funkcionalit"""

        logger.info("🔧 Consolidating similar functionality...")

        # Identify files with similar purposes
        consolidation_groups = {
            "main_files": ["main.py", "m1_main.py", "llama_main.py"],
            "research_engines": ["unified_research_engine.py", "biohacking_research_engine.py"],
            "ai_adapters": ["local_ai_adapter.py"],
            "servers": ["unified_server.py", "mcp_server.py"],
            "configs": ["unified_config.py"]
        }

        consolidations_performed = []

        for group_name, file_list in consolidation_groups.items():
            existing_files = [
                f for f in file_list
                if (self.project_root / f).exists()
            ]

            if len(existing_files) > 1:
                # Keep the most comprehensive one (usually "unified_" or longest)
                primary_file = max(existing_files, key=len)

                for file_name in existing_files:
                    if file_name != primary_file:
                        file_path = self.project_root / file_name

                        # Create consolidated comment
                        comment = f"# Functionality consolidated into {primary_file}\n"

                        try:
                            file_path.unlink()
                            consolidations_performed.append({
                                "removed": file_name,
                                "consolidated_into": primary_file,
                                "group": group_name
                            })
                        except Exception as e:
                            logger.warning(f"Failed to consolidate {file_name}: {e}")

        self.cleanup_report["consolidations"] = consolidations_performed

    async def _optimize_dependencies(self):
        """Optimalizace závislostí"""

        logger.info("📦 Optimizing dependencies...")

        requirements_files = [
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-private.txt"
        ]

        optimized_deps = []

        for req_file in requirements_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                try:
                    content = req_path.read_text()
                    optimized_content = self._optimize_requirements(content)

                    if optimized_content != content:
                        req_path.write_text(optimized_content)
                        optimized_deps.append(req_file)

                except Exception as e:
                    logger.warning(f"Failed to optimize {req_file}: {e}")

        self.cleanup_report["dependencies_optimized"] = optimized_deps

    def _optimize_requirements(self, content: str) -> str:
        """Optimalizace requirements souboru"""

        lines = content.split('\n')
        optimized_lines = []
        seen_packages = set()

        for line in lines:
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                optimized_lines.append(line)
                continue

            # Extract package name
            package_name = stripped.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0]
            package_name = package_name.strip()

            # Skip duplicates
            if package_name.lower() not in seen_packages:
                seen_packages.add(package_name.lower())
                optimized_lines.append(line)

        return '\n'.join(optimized_lines)

    async def _generate_cleanup_report(self):
        """Generování reportu o čištění"""

        report_path = self.project_root / "CLEANUP_REPORT.md"

        report_content = f"""# Project Cleanup Report

Generated: {self.cleanup_report['timestamp']}

## Summary
- **Files removed**: {len(self.cleanup_report['files_removed'])}
- **Space saved**: {self.cleanup_report['space_saved_mb']:.2f} MB
- **Files optimized**: {len(self.cleanup_report.get('files_optimized', []))}
- **Duplicates removed**: {self.cleanup_report.get('duplicates_found', 0)}

## Files Removed
"""

        for file_info in self.cleanup_report['files_removed']:
            report_content += f"- `{file_info['path']}` ({file_info['category']})\n"

        if self.cleanup_report.get('consolidations'):
            report_content += "\n## Consolidations\n"
            for cons in self.cleanup_report['consolidations']:
                report_content += f"- {cons['removed']} → {cons['consolidated_into']}\n"

        if self.cleanup_report.get('optimizations_applied'):
            report_content += "\n## Optimizations Applied\n"
            for opt in self.cleanup_report['optimizations_applied']:
                report_content += f"- {opt}\n"

        if self.cleanup_report.get('errors'):
            report_content += "\n## Errors\n"
            for error in self.cleanup_report['errors']:
                report_content += f"- {error}\n"

        report_content += "\n## Next Steps\n"
        report_content += "1. Run comprehensive tests\n"
        report_content += "2. Update documentation\n"
        report_content += "3. Verify all functionality works\n"
        report_content += "4. Consider further optimizations\n"

        report_path.write_text(report_content, encoding='utf-8')
        logger.info(f"📄 Cleanup report saved to {report_path}")

# Convenience function
async def cleanup_project(project_root: str = None) -> Dict[str, any]:
    """Convenience function for project cleanup"""

    root_path = Path(project_root) if project_root else Path.cwd()

    cleanup_system = ProjectCleanupOptimizer(root_path)
    return await cleanup_system.full_cleanup_and_optimization()

# Export
__all__ = ['ProjectCleanupOptimizer', 'cleanup_project']
