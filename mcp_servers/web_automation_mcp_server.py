#!/usr/bin/env python3
"""
Web Automation MCP Server - Pro web scraping a automatizaci
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional
from urllib.parse import urljoin
import aiohttp
from playwright.async_api import async_playwright, Browser
from bs4 import BeautifulSoup
from mcp.server import Server
import ssl
import certifi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebAutomationMCPServer:
    def __init__(self):
        self.server = Server("web-automation")
        self.browser: Optional[Browser] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._setup_ssl_context()
        self.setup_tools()

    def _setup_ssl_context(self):
        """Nastaví SSL kontext pro bezpečné spojení"""
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    async def _get_session(self) -> aiohttp.ClientSession:
        """Získá nebo vytvoří HTTP session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(ssl=self.ssl_context, limit=100)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
            )
        return self.session

    async def _get_browser(self) -> Browser:
        """Získá nebo vytvoří Playwright browser"""
        if self.browser is None:
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--memory-pressure-off'
                ]
            )
        return self.browser

    def setup_tools(self):
        """Nastaví dostupné nástroje pro web automatizaci"""

        async def scrape_url(url: str, selector: Optional[str] = None, method: str = "requests") -> Dict[str, Any]:
            """
            Stáhne a parsuje obsah webové stránky
            Args:
                url: URL adresa stránky
                selector: CSS selektor pro extrakci specifických elementů
                method: Metoda scraping ('requests' nebo 'playwright')
            """
            try:
                if method == "playwright":
                    return await self._scrape_with_playwright(url, selector)
                else:
                    return await self._scrape_with_requests(url, selector)
            except Exception as e:
                logger.error(f"Chyba při scraping URL {url}: {e}")
                return {
                    "error": str(e),
                    "url": url,
                    "success": False
                }

        async def extract_links(url: str, filter_pattern: Optional[str] = None) -> Dict[str, Any]:
            """
            Extrahuje všechny odkazy ze stránky
            Args:
                url: URL adresa stránky
                filter_pattern: Regex pattern pro filtrování odkazů
            """
            try:
                session = await self._get_session()
                async with session.get(url) as response:
                    content = await response.text()

                soup = BeautifulSoup(content, 'html.parser')
                links = []

                for link in soup.find_all('a', href=True):
                    href = link['href']
                    absolute_url = urljoin(url, href)
                    link_text = link.get_text(strip=True)

                    if filter_pattern:
                        import re
                        if not re.search(filter_pattern, absolute_url):
                            continue

                    links.append({
                        "url": absolute_url,
                        "text": link_text,
                        "title": link.get('title', '')
                    })

                return {
                    "url": url,
                    "links": links,
                    "count": len(links),
                    "success": True
                }

            except Exception as e:
                logger.error(f"Chyba při extrakci odkazů z {url}: {e}")
                return {
                    "error": str(e),
                    "url": url,
                    "success": False
                }

        async def browser_screenshot(url: str, selector: Optional[str] = None,
                                   full_page: bool = False) -> Dict[str, Any]:
            """
            Pořídí screenshot webové stránky
            Args:
                url: URL adresa stránky
                selector: CSS selektor pro screenshot specifického elementu
                full_page: Zda pořídit screenshot celé stránky
            """
            try:
                browser = await self._get_browser()
                page = await browser.new_page()

                await page.goto(url, wait_until='networkidle')

                if selector:
                    element = await page.query_selector(selector)
                    if element:
                        screenshot = await element.screenshot()
                    else:
                        return {"error": f"Element s selektorem '{selector}' nenalezen", "success": False}
                else:
                    screenshot = await page.screenshot(full_page=full_page)

                await page.close()

                import base64
                screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')

                return {
                    "url": url,
                    "screenshot": screenshot_b64,
                    "success": True
                }

            except Exception as e:
                logger.error(f"Chyba při pořizování screenshotu {url}: {e}")
                return {
                    "error": str(e),
                    "url": url,
                    "success": False
                }

        async def get_page_info(url: str) -> Dict[str, Any]:
            """
            Získá detailní informace o webové stránce
            Args:
                url: URL adresa stránky
            """
            try:
                session = await self._get_session()
                start_time = time.time()

                async with session.get(url) as response:
                    content = await response.text()
                    load_time = time.time() - start_time

                soup = BeautifulSoup(content, 'html.parser')

                # Extrakce meta informací
                meta_info = {}
                for meta in soup.find_all('meta'):
                    if meta.get('name'):
                        meta_info[meta['name']] = meta.get('content', '')
                    elif meta.get('property'):
                        meta_info[meta['property']] = meta.get('content', '')

                # Počítání různých elementů
                images = len(soup.find_all('img'))
                links = len(soup.find_all('a'))
                forms = len(soup.find_all('form'))
                scripts = len(soup.find_all('script'))

                return {
                    "url": url,
                    "title": soup.title.string if soup.title else "",
                    "meta_info": meta_info,
                    "load_time": round(load_time, 2),
                    "status_code": response.status,
                    "content_length": len(content),
                    "element_counts": {
                        "images": images,
                        "links": links,
                        "forms": forms,
                        "scripts": scripts
                    },
                    "success": True
                }

            except Exception as e:
                logger.error(f"Chyba při získávání informací o stránce {url}: {e}")
                return {
                    "error": str(e),
                    "url": url,
                    "success": False
                }

        # Registrace toolů
        self.tools = {
            'scrape_url': scrape_url,
            'extract_links': extract_links,
            'browser_screenshot': browser_screenshot,
            'get_page_info': get_page_info
        }

    async def _scrape_with_requests(self, url: str, selector: Optional[str] = None) -> Dict[str, Any]:
        """Scraping pomocí aiohttp a BeautifulSoup"""
        session = await self._get_session()
        async with session.get(url) as response:
            content = await response.text()

        soup = BeautifulSoup(content, 'html.parser')

        if selector:
            elements = soup.select(selector)
            content_data = [elem.get_text(strip=True) for elem in elements]
        else:
            content_data = soup.get_text(strip=True)

        return {
            "url": url,
            "title": soup.title.string if soup.title else "",
            "content": content_data,
            "success": True,
            "method": "requests"
        }

    async def _scrape_with_playwright(self, url: str, selector: Optional[str] = None) -> Dict[str, Any]:
        """Scraping pomocí Playwright pro dynamické stránky"""
        browser = await self._get_browser()
        page = await browser.new_page()

        await page.goto(url, wait_until='networkidle')

        if selector:
            elements = await page.query_selector_all(selector)
            content_data = []
            for element in elements:
                text = await element.text_content()
                content_data.append(text.strip() if text else "")
        else:
            content_data = await page.content()

        title = await page.title()
        await page.close()

        return {
            "url": url,
            "title": title,
            "content": content_data,
            "success": True,
            "method": "playwright"
        }

    async def cleanup(self):
        """Vyčištění zdrojů"""
        if self.session and not self.session.closed:
            await self.session.close()

        if self.browser:
            await self.browser.close()

    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Spustí nástroj podle jména"""
        if tool_name in self.tools:
            return await self.tools[tool_name](**kwargs)
        else:
            return {"error": f"Nástroj '{tool_name}' není dostupný", "success": False}

async def main():
    """Hlavní funkce pro spuštění serveru"""
    server = WebAutomationMCPServer()

    # Testování funkcionalit
    print("🔧 Testování Web Automation MCP Server...")

    try:
        # Test session
        session = await server._get_session()
        print("✅ HTTP session vytvořena")

        # Test cleanup
        await server.cleanup()
        print("✅ Cleanup dokončen")

        print("🎉 Web Automation MCP Server je funkční!")

    except Exception as e:
        print(f"❌ Chyba při testování: {e}")

if __name__ == "__main__":
    asyncio.run(main())
