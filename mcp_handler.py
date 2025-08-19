# mcp_handler.py
import asyncio
import os
import subprocess
from pathlib import Path
from fastapi import APIRouter
from mcp.server.main import create_mcp_server
from mcp.server.tools import Tool, ToolExecuteRequest, ToolExecuteResponse
from subprocess_tee import run


# --- Definice nástrojů ---

class ReadFileTool(Tool):
    """Nástroj pro čtení souborů."""
    name = "read_file"
    description = "Přečte obsah souboru v projektu."

    async def execute(self, req: ToolExecuteRequest) -> ToolExecuteResponse:
        try:
            file_path_str = req.parameters.get("query")
            if not file_path_str:
                return ToolExecuteResponse(stderr="Chybí cesta k souboru v parametru 'query'.")

            # Bezpečnostní omezení - povolujeme přístup pouze v rámci projektu
            project_root = Path("/app").resolve()
            target_path = (project_root / file_path_str).resolve()

            if not target_path.is_relative_to(project_root):
                return ToolExecuteResponse(stderr="Chyba: Pokus o přístup mimo adresář projektu.")

            if not target_path.exists():
                return ToolExecuteResponse(stderr=f"Chyba: Soubor neexistuje na cestě: {file_path_str}")

            content = target_path.read_text(encoding="utf-8")
            return ToolExecuteResponse(stdout=content)
        except Exception as e:
            return ToolExecuteResponse(stderr=f"Nastala chyba při čtení souboru: {e}")


class WriteFileTool(Tool):
    """Nástroj pro zápis do souborů."""
    name = "write_file"
    description = "Zapíše text do souboru. Pokud soubor neexistuje, vytvoří ho."

    async def execute(self, req: ToolExecuteRequest) -> ToolExecuteResponse:
        try:
            file_path_str = req.parameters.get("path")
            content = req.parameters.get("content")

            if not file_path_str:
                return ToolExecuteResponse(stderr="Chybí cesta k souboru v parametru 'path'.")
            if content is None:
                return ToolExecuteResponse(stderr="Chybí obsah pro zápis v parametru 'content'.")

            project_root = Path("/app").resolve()
            target_path = (project_root / file_path_str).resolve()

            if not target_path.is_relative_to(project_root):
                return ToolExecuteResponse(stderr="Chyba: Pokus o přístup mimo adresář projektu.")

            # Vytvoříme adresáře, pokud neexistují
            target_path.parent.mkdir(parents=True, exist_ok=True)

            target_path.write_text(content, encoding="utf-8")
            return ToolExecuteResponse(stdout=f"Soubor '{file_path_str}' byl úspěšně uložen.")
        except Exception as e:
            return ToolExecuteResponse(stderr=f"Nastala chyba při zápisu do souboru: {e}")


class RunInTerminalTool(Tool):
    """Nástroj pro spouštění příkazů v terminálu."""
    name = "run_in_terminal"
    description = "Spustí bezpečný příkaz v sandboxu (uvnitř Docker kontejneru)."

    async def execute(self, req: ToolExecuteRequest) -> ToolExecuteResponse:
        try:
            command = req.parameters.get("query")
            if not command:
                return ToolExecuteResponse(stderr="Chybí příkaz ke spuštění v parametru 'query'.")

            # Spuštění příkazu v /app adresáři uvnitř kontejneru
            result = await asyncio.to_thread(
                run, command, shell=True, cwd="/app", check=False
            )

            output = result.stdout.decode('utf-8')
            error = result.stderr.decode('utf-8')

            return ToolExecuteResponse(
                stdout=output,
                stderr=error,
                exit_code=result.returncode
            )
        except Exception as e:
            return ToolExecuteResponse(stderr=f"Nastala chyba při spouštění příkazu: {e}")


class ResearchTool(Tool):
    """Nástroj pro research pomocí academic scraper."""
    name = "research"
    description = "Provede akademický research na zadané téma."

    async def execute(self, req: ToolExecuteRequest) -> ToolExecuteResponse:
        try:
            query = req.parameters.get("query")
            if not query:
                return ToolExecuteResponse(stderr="Chybí dotaz v parametru 'query'.")

            # Import academic scraper
            try:
                from academic_scraper import create_scraping_orchestrator
                orchestrator = create_scraping_orchestrator()
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

                import json
                return ToolExecuteResponse(stdout=json.dumps(formatted_results, indent=2))

            except ImportError:
                return ToolExecuteResponse(stderr="Academic scraper není dostupný")

        except Exception as e:
            return ToolExecuteResponse(stderr=f"Chyba při research: {e}")


# Vytvoření MCP serveru s našimi nástroji
try:
    from fastapi import APIRouter

    mcp_router = APIRouter()

    # Registrace nástrojů
    tools = [
        ReadFileTool(),
        WriteFileTool(),
        RunInTerminalTool(),
        ResearchTool()
    ]

    @mcp_router.get("/tools")
    async def list_tools():
        """Seznam dostupných MCP nástrojů"""
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description
                } for tool in tools
            ]
        }

    @mcp_router.post("/tools/{tool_name}/execute")
    async def execute_tool(tool_name: str, request: dict):
        """Spuštění MCP nástroje"""
        for tool in tools:
            if tool.name == tool_name:
                # Simulate ToolExecuteRequest
                class MockRequest:
                    def __init__(self, params):
                        self.parameters = params

                mock_req = MockRequest(request.get("parameters", {}))
                result = await tool.execute(mock_req)

                return {
                    "stdout": getattr(result, 'stdout', ''),
                    "stderr": getattr(result, 'stderr', ''),
                    "exit_code": getattr(result, 'exit_code', 0)
                }

        return {"error": f"Tool {tool_name} not found"}

    @mcp_router.get("/")
    async def mcp_root():
        """MCP Server root endpoint"""
        return {
            "name": "Academic Research MCP Server",
            "version": "1.0.0",
            "tools_count": len(tools),
            "status": "running"
        }

except ImportError as e:
    print(f"⚠️ MCP dependencies not available: {e}")
    mcp_router = None
