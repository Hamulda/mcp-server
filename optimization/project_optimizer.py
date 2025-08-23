#!/usr/bin/env python3
"""
Pokročilý optimalizátor projektu - konsolidace a čištění
"""

import os
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Set
import ast
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProjectOptimizer:
    """Optimalizátor pro čištění a konsolidaci projektu"""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.duplicates = []
        self.unused_files = []
        self.core_files = {
            'unified_server.py',
            'main.py',
            'security_manager.py',
            'mcp_handler.py',
            'academic_scraper.py'
        }

    async def analyze_project(self) -> Dict[str, List[str]]:
        """Analyzuje projekt a identifikuje problémy"""
        logger.info("🔍 Analyzuji projekt...")

        analysis = {
            'duplicates': await self._find_duplicates(),
            'unused': await self._find_unused_files(),
            'oversized': await self._find_oversized_files(),
            'test_files': await self._find_test_files()
        }

        return analysis

    async def _find_duplicates(self) -> List[str]:
        """Najde duplikátní soubory podle obsahu"""
        duplicates = []
        file_hashes = {}

        for py_file in self.project_root.glob("*.py"):
            try:
                content = py_file.read_text()
                content_hash = hash(content)

                if content_hash in file_hashes:
                    duplicates.append(str(py_file.name))
                    logger.warning(f"Duplikát nalezen: {py_file.name}")
                else:
                    file_hashes[content_hash] = py_file.name
            except Exception as e:
                logger.error(f"Chyba při čtení {py_file}: {e}")

        return duplicates

    async def _find_unused_files(self) -> List[str]:
        """Najde nepoužívané soubory"""
        unused = []
        all_imports = set()

        # Najdi všechny importy
        for py_file in self.project_root.glob("*.py"):
            try:
                content = py_file.read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            all_imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            all_imports.add(node.module)
            except:
                continue

        # Kontrola nepoužívaných souborů
        for py_file in self.project_root.glob("*.py"):
            module_name = py_file.stem
            if (module_name not in all_imports and
                py_file.name not in self.core_files and
                not module_name.startswith('test_')):
                unused.append(py_file.name)

        return unused

    async def _find_oversized_files(self) -> List[str]:
        """Najde příliš velké soubory (>1000 řádků)"""
        oversized = []

        for py_file in self.project_root.glob("*.py"):
            try:
                lines = len(py_file.read_text().splitlines())
                if lines > 1000:
                    oversized.append(f"{py_file.name} ({lines} řádků)")
            except:
                continue

        return oversized

    async def _find_test_files(self) -> List[str]:
        """Najde všechny testovací soubory"""
        test_files = []

        for py_file in self.project_root.glob("test_*.py"):
            test_files.append(py_file.name)

        return test_files

    async def cleanup_project(self, analysis: Dict[str, List[str]]) -> None:
        """Provede čištění projektu"""
        logger.info("🧹 Čistím projekt...")

        # Vytvořím backup
        backup_dir = self.project_root / "backup_before_cleanup"
        backup_dir.mkdir(exist_ok=True)

        # Přesunu duplicitní soubory
        for duplicate in analysis['duplicates']:
            src = self.project_root / duplicate
            if src.exists():
                shutil.move(str(src), str(backup_dir / duplicate))
                logger.info(f"Přesunut duplikát: {duplicate}")

        # Konsolidace testovacích souborů
        await self._consolidate_test_files(analysis['test_files'])

        # Organizace do složek
        await self._organize_into_folders()

    async def _consolidate_test_files(self, test_files: List[str]) -> None:
        """Konsoliduje testovací soubory"""
        test_dir = self.project_root / "tests"
        test_dir.mkdir(exist_ok=True)

        for test_file in test_files:
            src = self.project_root / test_file
            if src.exists():
                shutil.move(str(src), str(test_dir / test_file))
                logger.info(f"Přesunut test: {test_file}")

    async def _organize_into_folders(self) -> None:
        """Organizuje soubory do logických složek"""

        folders = {
            'core': ['unified_server.py', 'main.py', 'mcp_handler.py'],
            'security': ['security_manager.py'],
            'scrapers': ['academic_scraper.py', 'advanced_source_aggregator.py'],
            'optimization': [f for f in os.listdir(self.project_root)
                           if f.endswith('.py') and 'optimizer' in f.lower()],
            'monitoring': ['advanced_monitoring_system.py', 'performance_monitoring_system.py'],
            'ai': ['local_ai_adapter.py', 'semantic_search_system.py'],
            'cache': ['smart_caching_system.py', 'unified_cache_system.py']
        }

        for folder_name, files in folders.items():
            folder_path = self.project_root / folder_name
            folder_path.mkdir(exist_ok=True)

            for file_name in files:
                src = self.project_root / file_name
                if src.exists() and src.is_file():
                    dest = folder_path / file_name
                    if not dest.exists():
                        shutil.move(str(src), str(dest))
                        logger.info(f"Přesunut {file_name} -> {folder_name}/")

async def main():
    """Hlavní funkce optimalizátora"""
    project_root = Path("/Users/vojtechhamada/PycharmProjects/PythonProject2")
    optimizer = ProjectOptimizer(project_root)

    # Analýza
    analysis = await optimizer.analyze_project()

    # Výpis analýzy
    print("\n📊 ANALÝZA PROJEKTU")
    print("=" * 50)
    print(f"Duplikáty: {len(analysis['duplicates'])}")
    print(f"Nepoužívané: {len(analysis['unused'])}")
    print(f"Příliš velké: {len(analysis['oversized'])}")
    print(f"Testovací soubory: {len(analysis['test_files'])}")

    # Čištění
    await optimizer.cleanup_project(analysis)

    print("\n✅ Optimalizace dokončena!")

if __name__ == "__main__":
    asyncio.run(main())
