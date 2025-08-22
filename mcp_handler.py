#!/usr/bin/env python3
"""
MCP Handler - Opravený handler pro Model Context Protocol
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

# Simplified MCP implementation without external dependencies
class ToolExecuteRequest(BaseModel):
    """Request model for tool execution"""
    name: str
    parameters: Dict[str, Any] = {}

class ToolExecuteResponse(BaseModel):
    """Response model for tool execution"""
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: Optional[int] = None
    success: bool = True

class BaseTool:
    """Base class for MCP tools"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def execute(self, req: ToolExecuteRequest) -> ToolExecuteResponse:
        raise NotImplementedError

class ReadFileTool(BaseTool):
    """Nástroj pro čtení souborů."""

    def __init__(self):
        super().__init__("read_file", "Přečte obsah souboru v projektu.")

    async def execute(self, req: ToolExecuteRequest) -> ToolExecuteResponse:
        try:
            file_path_str = req.parameters.get("query") or req.parameters.get("path")
            if not file_path_str:
                return ToolExecuteResponse(
                    stderr="Chybí cesta k souboru v parametru 'query' nebo 'path'.",
                    success=False
                )

            # Bezpečnostní omezení - povolujeme přístup pouze v rámci projektu
            project_root = Path("/app").resolve()
            target_path = (project_root / file_path_str).resolve()

            if not target_path.is_relative_to(project_root):
                return ToolExecuteResponse(
                    stderr="Chyba: Pokus o přístup mimo adresář projektu.",
                    success=False
                )

            if not target_path.exists():
                return ToolExecuteResponse(
                    stderr=f"Chyba: Soubor neexistuje na cestě: {file_path_str}",
                    success=False
                )

            content = target_path.read_text(encoding="utf-8")
            return ToolExecuteResponse(stdout=content, success=True)
        except Exception as e:
            return ToolExecuteResponse(stderr=f"Nastala chyba při čtení souboru: {e}", success=False)

class WriteFileTool(BaseTool):
    """Nástroj pro zápis do souborů."""

    def __init__(self):
        super().__init__("write_file", "Zapíše text do souboru. Pokud soubor neexistuje, vytvoří ho.")

    async def execute(self, req: ToolExecuteRequest) -> ToolExecuteResponse:
        try:
            file_path_str = req.parameters.get("path")
            content = req.parameters.get("content")

            if not file_path_str:
                return ToolExecuteResponse(
                    stderr="Chybí cesta k souboru v parametru 'path'.",
                    success=False
                )
            if content is None:
                return ToolExecuteResponse(
                    stderr="Chybí obsah pro zápis v parametru 'content'.",
                    success=False
                )

            project_root = Path("/app").resolve()
            target_path = (project_root / file_path_str).resolve()

            if not target_path.is_relative_to(project_root):
                return ToolExecuteResponse(
                    stderr="Chyba: Pokus o přístup mimo adresář projektu.",
                    success=False
                )

            # Vytvoříme adresáře, pokud neexistují
            target_path.parent.mkdir(parents=True, exist_ok=True)

            target_path.write_text(content, encoding="utf-8")
            return ToolExecuteResponse(
                stdout=f"Soubor '{file_path_str}' byl úspěšně uložen.",
                success=True
            )
        except Exception as e:
            return ToolExecuteResponse(stderr=f"Nastala chyba při zápisu do souboru: {e}", success=False)

class RunInTerminalTool(BaseTool):
    """Nástroj pro spouštění příkazů v terminálu."""

    def __init__(self):
        super().__init__("run_in_terminal", "Spustí bezpečný příkaz v sandboxu (uvnitř Docker kontejneru).")

    async def execute(self, req: ToolExecuteRequest) -> ToolExecuteResponse:
        try:
            command = req.parameters.get("query") or req.parameters.get("command")
            if not command:
                return ToolExecuteResponse(
                    stderr="Chybí příkaz ke spuštění v parametru 'query' nebo 'command'.",
                    success=False
                )

            # Spuštění příkazu v /app adresáři uvnitř kontejneru
            result = await asyncio.to_thread(
                subprocess.run,
                command,
                shell=True,
                cwd="/app",
                capture_output=True,
                text=True,
                check=False
            )

            return ToolExecuteResponse(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                success=result.returncode == 0
            )
        except Exception as e:
            return ToolExecuteResponse(stderr=f"Nastala chyba při spouštění příkazu: {e}", success=False)

class ResearchTool(BaseTool):
    """Nástroj pro research pomocí academic scraper."""

    def __init__(self):
        super().__init__("research", "Provede akademický research na zadané téma.")

    async def execute(self, req: ToolExecuteRequest) -> ToolExecuteResponse:
        try:
            query = req.parameters.get("query")
            if not query:
                return ToolExecuteResponse(
                    stderr="Chybí dotaz v parametru 'query'.",
                    success=False
                )

            # Import academic scraper
            try:
                from academic_scraper import create_scraping_orchestrator
                orchestrator = await create_scraping_orchestrator()
                results = await orchestrator.scrape_all_sources(query, max_results=5)

                # Format results
                formatted_results = []
                for result in results:
                    if result.success:
                        formatted_results.append({
                            "source": result.source,
                            "data": result.data,
                            "success": result.success
                        })

                return ToolExecuteResponse(
                    stdout=json.dumps(formatted_results, indent=2),
                    success=True
                )

            except ImportError:
                return ToolExecuteResponse(
                    stderr="Academic scraper není dostupný",
                    success=False
                )

        except Exception as e:
            return ToolExecuteResponse(stderr=f"Chyba při research: {e}", success=False)

# Vytvoření MCP serveru s našimi nástroji
mcp_router = APIRouter(prefix="/mcp", tags=["MCP"])

# Registrace nástrojů
tools = [
    ReadFileTool(),
    WriteFileTool(),
    RunInTerminalTool(),
    ResearchTool()
]

@mcp_router.get("/tools")
async def list_tools():
    """Seznam dostupných nástrojů"""
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in tools
        ]
    }

@mcp_router.post("/tools/{tool_name}/execute")
async def execute_tool(tool_name: str, request: ToolExecuteRequest):
    """Spuštění konkrétního nástroje"""
    tool = next((t for t in tools if t.name == tool_name), None)
    if not tool:
        return {"error": f"Nástroj '{tool_name}' nebyl nalezen"}

    try:
        result = await tool.execute(request)
        return result.model_dump()
    except Exception as e:
        return {"error": f"Chyba při spouštění nástroje: {e}", "success": False}

@mcp_router.get("/status")
async def mcp_status():
    """Status MCP serveru"""
    return {
        "status": "running",
        "tools_count": len(tools),
        "available_tools": [tool.name for tool in tools]
    }
