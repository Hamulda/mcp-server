"""
Unified FastAPI Server - Čistý a funkční server pro Academic Research Tool
"""

import asyncio
import logging
import time
import os
import hashlib
import json
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import prometheus_client
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from fastapi import Response
from cache_manager import CacheManager
from typing import Optional

# Import unified components
try:
    from unified_config import get_config
    from academic_scraper import create_scraping_orchestrator, ScrapingResult
    UNIFIED_CONFIG_AVAILABLE = True
    print("✅ Unified server using unified configuration")
except ImportError as e:
    UNIFIED_CONFIG_AVAILABLE = False
    print(f"⚠️  Unified config unavailable: {e}")

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
        description="List of sources to scrape (wikipedia, pubmed). If None, all sources will be used."
    )

class ScrapeResponse(BaseModel):
    success: bool
    query: str
    results: List[Dict[str, Any]]
    total_sources: int
    successful_sources: int
    execution_time: float

class ResearchRequestModel(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    strategy: str = Field(default="balanced")
    domain: str = Field(default="general")
    sources: Optional[List[str]] = None
    max_results: int = 10
    user_id: str = "default"
    budget_limit: Optional[float] = None

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
            description="Unified API for academic content scraping and research",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        # Load configuration
        if UNIFIED_CONFIG_AVAILABLE:
            self.config = get_config()
            cors_enabled = self.config.api.cors_enabled
        else:
            cors_enabled = True

        # Setup CORS
        if cors_enabled:
            allowed_origins = ["*"]
            if UNIFIED_CONFIG_AVAILABLE and self.config.environment.value == 'production':
                # In production default to restrictive CORS; override via env CORS_ORIGINS
                origins_env = os.getenv('CORS_ORIGINS', '')
                if origins_env:
                    allowed_origins = [o.strip() for o in origins_env.split(',') if o.strip()]
                else:
                    allowed_origins = []  # no wildcard in prod by default
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=allowed_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Rate limiting
        self.limiter = Limiter(key_func=lambda request: request.client.host)
        self.app.state.limiter = self.limiter
        self.app.add_exception_handler(RateLimitExceeded, self._rate_limit_handler)
        self.app.add_middleware(SlowAPIMiddleware)

        # Initialize cache manager (shared cache adapter)
        self.cache = CacheManager()

        # Setup routes
        self._setup_routes()

        # Startup time tracking
        self.startup_time = time.time()

        # Setup logging
        self.logger = logging.getLogger(__name__)

    async def _rate_limit_handler(self, request, exc):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    def _setup_routes(self):
        """Setup all API routes"""

        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Landing page with API documentation links"""
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Academic Research Tool API</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h1 { color: #2c3e50; }
                    ul { line-height: 1.6; }
                    a { color: #3498db; text-decoration: none; }
                    a:hover { text-decoration: underline; }
                    .status { background: #ecf0f1; padding: 20px; border-radius: 5px; }
                </style>
            </head>
            <body>
                <h1>🔬 Academic Research Tool API</h1>
                <p>Unified API for academic content scraping and research.</p>
                <h2>📚 Documentation</h2>
                <ul>
                    <li><a href="/docs">Interactive API Documentation (Swagger UI)</a></li>
                    <li><a href="/redoc">Alternative Documentation (ReDoc)</a></li>
                </ul>
                <h2>🔗 API Endpoints</h2>
                <ul>
                    <li><a href="/health">💚 Health Check</a></li>
                    <li><a href="/api/v1/sources">📋 Available Sources</a></li>
                    <li>🔍 POST /api/v1/scrape - Main scraping endpoint</li>
                </ul>
                <div class="status">
                    <p><em>Status: ✅ Server is running | Powered by FastAPI + Academic Scraper</em></p>
                </div>
            </body>
            </html>
            """

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            try:
                # Test orchestrator availability
                orchestrator = create_scraping_orchestrator()
                available_scrapers = list(orchestrator.scrapers.keys())

                uptime = time.time() - self.startup_time

                return HealthResponse(
                    status="healthy",
                    uptime=uptime,
                    version="2.0.0",
                    environment=self.config.environment.value if UNIFIED_CONFIG_AVAILABLE else "unknown",
                    scrapers_available=available_scrapers
                )

            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

        @self.app.get("/api/v1/sources", response_model=SourcesResponse)
        async def get_sources():
            """Get available sources and their configurations"""
            try:
                orchestrator = create_scraping_orchestrator()
                available_sources = list(orchestrator.scrapers.keys())

                source_configs = {}
                if UNIFIED_CONFIG_AVAILABLE:
                    for source in available_sources:
                        source_config = self.config.get_source_config(source)
                        if source_config:
                            source_configs[source] = {
                                "name": source_config.name,
                                "enabled": source_config.enabled,
                                "rate_limit_delay": source_config.rate_limit_delay,
                                "base_url": source_config.base_url
                            }

                return SourcesResponse(
                    available_sources=available_sources,
                    source_configs=source_configs
                )

            except Exception as e:
                self.logger.error(f"Failed to get sources: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to retrieve sources: {str(e)}")

        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

        # In-flight de-duplication map
        inflight: Dict[str, asyncio.Task] = {}

        @self.app.post("/api/v1/scrape", response_model=ScrapeResponse)
        @self.limiter.limit(os.getenv('RATE_LIMIT_SCRAPE', '30/minute'))
        async def scrape_sources(body: ScrapeRequest, request: Request):
            """Main scraping endpoint"""
            start_time = time.time()
            cache_key = f"{body.query.lower().strip()}|{','.join(body.sources or [])}"
            cached = self.cache.get(cache_key)
            if cached:
                cache_hits.inc()
                self.logger.info(f"Cache hit for {cache_key}")
                etag = etag_for_response(cached.dict() if hasattr(cached, 'dict') else cached)
                if_none_match = request.headers.get("if-none-match")
                if if_none_match and if_none_match == etag:
                    return Response(status_code=304, headers={"ETag": etag})
                return Response(content=cached.json() if hasattr(cached, 'json') else json.dumps(cached), media_type="application/json", headers={"ETag": etag})
            cache_misses.inc()
            # In-flight de-duplication
            if cache_key in inflight:
                result = await inflight[cache_key]
                etag = etag_for_response(result.dict() if hasattr(result, 'dict') else result)
                if_none_match = request.headers.get("if-none-match")
                if if_none_match and if_none_match == etag:
                    return Response(status_code=304, headers={"ETag": etag})
                return Response(content=result.json() if hasattr(result, 'json') else json.dumps(result), media_type="application/json", headers={"ETag": etag})
            try:
                self.logger.info(f"Scraping request: query='{body.query}', sources={body.sources}")

                # Create orchestrator and perform scraping
                orchestrator = create_scraping_orchestrator()
                async def _run():
                    return await orchestrator.scrape_all_sources(body.query, body.sources)
                task = asyncio.create_task(_run())
                inflight[cache_key] = task
                try:
                    results = await task
                finally:
                    inflight.pop(cache_key, None)

                # Process results
                successful_results = [r for r in results if r.success]
                execution_time = time.time() - start_time

                # Convert ScrapingResult objects to dictionaries
                results_data = []
                for result in results:
                    scrape_counter.labels(result.source).inc()
                    if result.response_time is not None:
                        scrape_duration.labels(result.source).observe(result.response_time)
                    if not result.success:
                        error_code = str(result.error) if result.error else "unknown"
                        error_counter.labels(endpoint="scrape", source=result.source, error_code=error_code).inc()
                    result_dict = {
                        "source": result.source,
                        "success": result.success,
                        "data": result.data,
                        "error": result.error,
                        "execution_time": result.response_time,  # Použij response_time místo execution_time
                        "cached": False
                    }
                    results_data.append(result_dict)

                response = ScrapeResponse(
                    success=len(successful_results) > 0,
                    query=body.query,
                    results=results_data,
                    total_sources=len(results),
                    successful_sources=len(successful_results),
                    execution_time=execution_time
                )
                self.cache.set(cache_key, response)
                self.logger.info(f"Scraping completed: {len(successful_results)}/{len(results)} sources successful")
                etag = etag_for_response(response.dict())
                if_none_match = request.headers.get("if-none-match")
                if if_none_match and if_none_match == etag:
                    return Response(status_code=304, headers={"ETag": etag})
                return Response(content=response.json(), media_type="application/json", headers={"ETag": etag})
            except Exception as e:
                self.logger.error(f"Scraping failed: {e}")
                execution_time = time.time() - start_time

                response = ScrapeResponse(
                    success=False,
                    query=body.query,
                    results=[],
                    total_sources=0,
                    successful_sources=0,
                    execution_time=execution_time
                )
                etag = etag_for_response(response.dict())
                return Response(content=response.json(), media_type="application/json", headers={"ETag": etag})

        # Unified Research endpoint using UnifiedResearchEngine
        @self.app.post("/api/v1/research")
        @self.limiter.limit(os.getenv('RATE_LIMIT_RESEARCH', '10/minute'))
        async def research_endpoint(payload: ResearchRequestModel, request: Request):
            try:
                from unified_research_engine import UnifiedResearchEngine, ResearchRequest
                engine = UnifiedResearchEngine()
                req = ResearchRequest(
                    query=payload.query,
                    strategy=payload.strategy,
                    domain=payload.domain,
                    sources=payload.sources,
                    max_results=payload.max_results,
                    user_id=payload.user_id,
                    budget_limit=payload.budget_limit
                )
                rkey = f"research|{payload.query.lower().strip()}|{payload.strategy}|{payload.domain}|{','.join(payload.sources or [])}"
                start_time = time.time()
                if rkey in inflight:
                    result = await inflight[rkey]
                else:
                    async def _run_research():
                        return await engine.research(req)
                    task = asyncio.create_task(_run_research())
                    inflight[rkey] = task
                    try:
                        result = await task
                    finally:
                        inflight.pop(rkey, None)
                duration = time.time() - start_time
                # Histogram pro research_duration
                for src in (result.sources_found if hasattr(result, 'sources_found') and isinstance(result.sources_found, list) else []):
                    research_duration.labels(src).observe(duration)
                # Error counter pro research
                if not result.success:
                    error_code = str(getattr(result, 'error', 'unknown'))
                    for src in (result.sources_found if hasattr(result, 'sources_found') and isinstance(result.sources_found, list) else ['unknown']):
                        error_counter.labels(endpoint="research", source=src, error_code=error_code).inc()
                response_dict = {
                    'query_id': result.query_id,
                    'success': result.success,
                    'sources_found': result.sources_found,
                    'total_tokens': result.total_tokens,
                    'cost': result.cost,
                    'execution_time': result.execution_time,
                    'summary': result.summary,
                    'key_findings': result.key_findings,
                    'cached': result.cached,
                    'timestamp': result.timestamp.isoformat(),
                    'detailed_results': result.detailed_results,
                }
                etag = etag_for_response(response_dict)
                if_none_match = request.headers.get("if-none-match")
                if if_none_match and if_none_match == etag:
                    return Response(status_code=304, headers={"ETag": etag})
                return Response(content=json.dumps(response_dict), media_type="application/json", headers={"ETag": etag})
            except Exception as e:
                self.logger.error(f"Research endpoint failed: {e}")
                error_counter.labels(endpoint="research", source="unknown", error_code=str(e)).inc()
                raise HTTPException(status_code=500, detail=str(e))

    def get_app(self) -> FastAPI:
        """Get FastAPI application instance"""
        return self.app

# Create server instance
def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    server = UnifiedServer()
    return server.get_app()

# For direct running
if __name__ == "__main__":
    import uvicorn

    app = create_app()

    # Load config for port
    try:
        config = get_config()
        port = config.api.port
        host = config.api.host
    except:
        port = 8000
        host = "0.0.0.0"

    print(f"🚀 Starting Academic Research Tool API on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=True)
