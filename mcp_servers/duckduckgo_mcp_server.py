#!/usr/bin/env python3
"""
DuckDuckGo Search MCP Server - Zdarma bez API klíče
"""

import asyncio
from typing import Any, Dict, List
from ddgs import DDGS
import logging

# Nastavení loggingu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DuckDuckGoMCPServer:
    def __init__(self):
        self.ddgs = DDGS()
        logger.info("🔍 DuckDuckGo MCP Server inicializován")

    async def search_web(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Vyhledává na webu pomocí DuckDuckGo"""
        try:
            results = []
            for result in self.ddgs.text(query, max_results=max_results):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "source": "DuckDuckGo"
                })

            logger.info(f"Nalezeno {len(results)} výsledků pro dotaz: {query}")
            return results

        except Exception as e:
            logger.error(f"Chyba při vyhledávání: {e}")
            return [{"error": f"Chyba při vyhledávání: {str(e)}"}]

    async def search_news(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Vyhledává zprávy pomocí DuckDuckGo"""
        try:
            results = []
            for result in self.ddgs.news(query, max_results=max_results):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("body", ""),
                    "date": result.get("date", ""),
                    "source": result.get("source", "DuckDuckGo News")
                })

            logger.info(f"Nalezeno {len(results)} zpráv pro dotaz: {query}")
            return results

        except Exception as e:
            logger.error(f"Chyba při vyhledávání zpráv: {e}")
            return [{"error": f"Chyba při vyhledávání zpráv: {str(e)}"}]

    async def search_images(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Vyhledává obrázky pomocí DuckDuckGo"""
        try:
            results = []
            for result in self.ddgs.images(query, max_results=max_results):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("image", ""),
                    "thumbnail": result.get("thumbnail", ""),
                    "source": result.get("source", "DuckDuckGo Images"),
                    "width": result.get("width", ""),
                    "height": result.get("height", "")
                })

            logger.info(f"Nalezeno {len(results)} obrázků pro dotaz: {query}")
            return results

        except Exception as e:
            logger.error(f"Chyba při vyhledávání obrázků: {e}")
            return [{"error": f"Chyba při vyhledávání obrázků: {str(e)}"}]

    async def run(self):
        """Spustí MCP server"""
        logger.info("🔍 Spouštím DuckDuckGo MCP Server...")
        logger.info("✅ DuckDuckGo Search je připraven - bez potřeby API klíče!")

        # Udržuje server běžící
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 DuckDuckGo MCP Server ukončen")

def main():
    """Hlavní funkce pro spuštění serveru"""
    server = DuckDuckGoMCPServer()
    asyncio.run(server.run())

if __name__ == "__main__":
    main()
