"""
Unified FastAPI Server - Čistý a funkční server pro Academic Research Tool s MCP
"""

import asyncio
import logging
import time
import os
import hashlib
import json
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY

# Import sjednoceného cache systému
from unified_cache_system import get_cache_manager

# Helper functions
def etag_for_response(data: Any) -> str:
    """Generate ETag for response data"""
    if isinstance(data, dict):
        content = json.dumps(data, sort_keys=True)
    elif hasattr(data, 'model_dump'):
        content = json.dumps(data.model_dump(), sort_keys=True)
    else:
        content = str(data)
    return hashlib.md5(content.encode()).hexdigest()

# Import unified components s fallbackem pro robustnost
try:
    from unified_config import get_config
    UNIFIED_CONFIG_AVAILABLE = True
    print("✅ Unified server using unified configuration")
except ImportError as e:
    UNIFIED_CONFIG_AVAILABLE = False
    get_config = None
    print(f"⚠️  Unified config unavailable: {e}")

try:
    from academic_scraper import create_scraping_orchestrator, ScrapingResult
    SCRAPER_AVAILABLE = True
except ImportError as e:
    create_scraping_orchestrator = None
    ScrapingResult = None
    SCRAPER_AVAILABLE = False
    print(f"⚠️  Academic scraper unavailable: {e}")

# Try to import MCP handler, but don't fail if it's not available
try:
    from mcp_handler import mcp_router
    MCP_AVAILABLE = True
    print("✅ MCP handler available")
except ImportError as e:
    mcp_router = None
    MCP_AVAILABLE = False
    print(f"⚠️  MCP handler unavailable: {e}")

# Globální metriky (singletony)
scrape_counter = Counter('scrape_requests_total', 'Počet scraping požadavků', ['source'], registry=REGISTRY)
scrape_duration = Histogram('scrape_duration_seconds', 'Doba trvání scraping požadavků', ['source'], registry=REGISTRY)
cache_hits = Counter('cache_hits_total', 'Počet cache hitů', registry=REGISTRY)
cache_misses = Counter('cache_misses_total', 'Počet cache missů', registry=REGISTRY)
error_counter = Counter('scrape_errors_total', 'Počet chyb při scrapování/researchi', ['endpoint', 'source', 'error_code'], registry=REGISTRY)
research_duration = Histogram('research_duration_seconds', 'Doba trvání research požadavků', ['source'], registry=REGISTRY)

# Pydantic models pro API
class ScrapeRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    sources: Optional[List[str]] = Field(
        default=None,
        description="List of sources to scrape (e.g., 'wikipedia', 'pubmed'). If None, all sources will be used."
    )

class ScrapeResponse(BaseModel):
    success: bool
    query: str
    results: List[Dict[str, Any]]
    total_sources: int
    successful_sources: int
    execution_time: float

class HealthResponse(BaseModel):
    status: str
    uptime: float
    version: str
    environment: str
    scrapers_available: List[str]

class SourcesResponse(BaseModel):
    available_sources: List[str]
    source_configs: Dict[str, Dict[str, Any]]

class UnifiedServer:
    """Unified FastAPI server pro Academic Research Tool"""

    def __init__(self):
        self.app = FastAPI(
            title="Academic Research Tool API",
            description="Unified API for academic content scraping and research, now with MCP support.",
            version="2.1.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        # Oprava: Správné načítání konfigurace
        if UNIFIED_CONFIG_AVAILABLE:
            self.config = get_config()
        else:
            # Fallback konfigurace
            self.config = type('Config', (), {
                'api': type('API', (), {'cors_enabled': True, 'host': '0.0.0.0', 'port': 8000})(),
                'environment': type('Env', (), {'value': 'development'})()
            })()

        self._setup_middleware()
        self.cache = get_cache_manager()
        self._setup_routes()
        self.startup_time = time.time()
        self.logger = logging.getLogger(__name__)

    def _setup_middleware(self):
        """Konfigurace middleware (CORS, Rate Limiting)"""
        # CORS - oprava přístupu k config
        cors_enabled = True
        if UNIFIED_CONFIG_AVAILABLE and hasattr(self.config, 'api'):
            cors_enabled = self.config.api.cors_enabled

        if cors_enabled:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"], # Pro jednoduchost, v produkci omezit
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Rate Limiting
        self.limiter = Limiter(key_func=lambda request: request.client.host)
        self.app.state.limiter = self.limiter
        self.app.add_exception_handler(RateLimitExceeded, self._rate_limit_handler)
        self.app.add_middleware(SlowAPIMiddleware)

    async def _rate_limit_handler(self, request, exc):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"}
        )

    def _setup_routes(self):
        """Setup all API routes"""

        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Landing page with API documentation links"""
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Research Tool API + MCP</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f4f6f9; color: #333; }}
                    h1 {{ color: #2c3e50; }}
                    ul {{ line-height: 1.8; }}
                    a {{ color: #3498db; text-decoration: none; font-weight: bold; }}
                    a:hover {{ text-decoration: underline; }}
                    .container {{ background: #fff; padding: 20px 40px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🔬 Research Tool API & MCP Server</h1>
                    <p>Unified API for academic scraping and an MCP server for AI agent tools.</p>
                    <h2>📚 API Documentation</h2>
                    <ul>
                        <li><a href="/docs">Interactive API Docs (Swagger UI)</a></li>
                        <li><a href="/redoc">Alternative Docs (ReDoc)</a></li>
                    </ul>
                    <h2>🛠️ MCP (Model Context Protocol)</h2>
                     <ul>
                        <li>MCP server is running at <code>/mcp</code></li>
                        <li>Your AI agent can connect to this base URL to access tools.</li>
                    </ul>
                    <h2>🔗 Other Endpoints</h2>
                    <ul>
                        <li><a href="/health">💚 Health Check</a></li>
                        <li><a href="/api/v1/sources">📋 Available Sources</a></li>
                    </ul>
                </div>
            </body>
            </html>
            """

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            available_scrapers = []
            if create_scraping_orchestrator:
                try:
                    orchestrator = create_scraping_orchestrator()
                    available_scrapers = list(orchestrator.scrapers.keys())
                except Exception:
                    available_scrapers = ["Error initializing scrapers"]

            return HealthResponse(
                status="healthy",
                uptime=time.time() - self.startup_time,
                version="2.1.0",
                environment=self.config.environment.value if UNIFIED_CONFIG_AVAILABLE else "fallback",
                scrapers_available=available_scrapers
            )

        @self.app.get("/api/v1/sources", response_model=SourcesResponse)
        async def get_sources():
            """Get available sources and their configurations"""
            # Implement logic to get sources, similar to health_check
            return {"available_sources": [], "source_configs": {}}

        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

        inflight: Dict[str, asyncio.Task] = {}

        @self.app.post("/api/v1/scrape", response_model=ScrapeResponse)
        @self.limiter.limit(os.getenv('RATE_LIMIT_SCRAPE', '30/minute'))
        async def scrape_sources(body: ScrapeRequest, request: Request):
            """Main scraping endpoint"""
            start_time = time.time()
            cache_key = f"{body.query.lower().strip()}|{','.join(sorted(body.sources or []))}"

            # Check cache first
            cached = await self.cache.get(cache_key)
            if cached:
                cache_hits.inc()
                self.logger.info(f"Cache hit for {cache_key}")
                etag = etag_for_response(cached)
                if_none_match = request.headers.get("if-none-match")
                if if_none_match and if_none_match == etag:
                    return Response(status_code=304, headers={"ETag": etag})
                # Return cached response directly if it's already a dict
                if isinstance(cached, dict):
                    return Response(content=json.dumps(cached), media_type="application/json", headers={"ETag": etag})
                return Response(content=cached.model_dump_json(), media_type="application/json", headers={"ETag": etag})

            cache_misses.inc()

            # Check if request is already in flight
            if cache_key in inflight:
                result = await inflight[cache_key]
                etag = etag_for_response(result)
                return Response(content=result.model_dump_json(), media_type="application/json", headers={"ETag": etag})

            try:
                self.logger.info(f"Scraping request: query='{body.query}', sources={body.sources}")

                # If unified modules are available, use them
                if UNIFIED_CONFIG_AVAILABLE and create_scraping_orchestrator:
                    orchestrator = create_scraping_orchestrator()

                    async def _run():
                        return await orchestrator.scrape_all_sources(body.query, body.sources)

                    task = asyncio.create_task(_run())
                    inflight[cache_key] = task

                    try:
                        results = await task
                    finally:
                        inflight.pop(cache_key, None)

                    successful_results = [r for r in results if r.success]
                    execution_time = time.time() - start_time
                    results_data = []

                    for r in results:
                        scrape_counter.labels(r.source).inc()
                        if r.response_time:
                            scrape_duration.labels(r.source).observe(r.response_time)
                        if not r.success:
                            error_counter.labels(endpoint="scrape", source=r.source, error_code=str(r.error or 'unknown')).inc()
                        results_data.append(r.__dict__)

                    response_data = ScrapeResponse(
                        success=len(successful_results) > 0,
                        query=body.query,
                        results=results_data,
                        total_sources=len(results),
                        successful_sources=len(successful_results),
                        execution_time=execution_time
                    )
                else:
                    # Fallback mode - return mock data
                    execution_time = time.time() - start_time
                    response_data = ScrapeResponse(
                        success=True,
                        query=body.query,
                        results=[{
                            "source": "fallback",
                            "query": body.query,
                            "success": True,
                            "data": {"message": "Fallback mode - unified modules not available"},
                            "response_time": execution_time
                        }],
                        total_sources=1,
                        successful_sources=1,
                        execution_time=execution_time
                    )

                # Cache successful results
                if response_data.successful_sources > 0:
                    await self.cache.set(cache_key, response_data.model_dump())

                etag = etag_for_response(response_data)
                return Response(
                    content=response_data.model_dump_json(),
                    media_type="application/json",
                    headers={"ETag": etag}
                )

            except Exception as e:
                error_counter.labels(endpoint="scrape", source="all", error_code=type(e).__name__).inc()
                self.logger.error(f"Scraping failed: {e}")
                raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

        # Optional: Add MCP router if available
        if MCP_AVAILABLE and mcp_router:
            self.app.include_router(mcp_router)
            print("✅ MCP router registered at /mcp")
        else:
            print("⚠️  MCP router not available")

    def get_app(self) -> FastAPI:
        """Get FastAPI application instance"""
        return self.app

# Factory funkce pro Uvicorn a Docker
def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    server = UnifiedServer()
    return server.get_app()

app = create_app()

# Pro přímé spuštění (lokální vývoj)
if __name__ == "__main__":
    import uvicorn
    # Použijeme port 8001 místo 8000 kvůli konfliktu
    port = 8001
    host = "0.0.0.0"

    if UNIFIED_CONFIG_AVAILABLE:
        config = get_config()
        host = getattr(config.api, 'host', '0.0.0.0')
        port = getattr(config.api, 'port', 8001)

    print(f"🚀 Starting server on {host}:{port}")
    uvicorn.run(
        "unified_server:app",
        host=host,
        port=port,
        reload=True
    )