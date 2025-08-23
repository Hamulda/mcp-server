"""
Unified FastAPI Server - Enterprise-ready server s pokročilými optimalizacemi
"""

import asyncio
import logging
import time
import os
import hashlib
import json
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY

# Import optimalizovaných komponent
from cache.intelligent_cache_system import cache_manager, initialize_cache
from security.security_manager import security_manager
from optimization.advanced_rate_limiter import rate_limiter, rate_limited
from core.remote_mcp_handler import remote_mcp_handler, setup_remote_mcp_routes

# Import existujících komponent s fallbackem
try:
    from scrapers.academic_scraper import create_scraping_orchestrator, ScrapingResult
    SCRAPER_AVAILABLE = True
except ImportError as e:
    create_scraping_orchestrator = None
    ScrapingResult = None
    SCRAPER_AVAILABLE = False
    print(f"⚠️ Academic scraper unavailable: {e}")

try:
    from mcp_handler import mcp_router
    MCP_AVAILABLE = True
    print("✅ MCP handler available")
except ImportError as e:
    mcp_router = None
    MCP_AVAILABLE = False
    print(f"⚠️ MCP handler unavailable: {e}")

# Nastavení loggingu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metriky
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')
CACHE_HITS = Counter('cache_hits_total', 'Cache hits', ['cache_type'])
RATE_LIMIT_EXCEEDED = Counter('rate_limit_exceeded_total', 'Rate limit exceeded', ['source'])

# Pydantic modely
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    components: Dict[str, bool]

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Vyhledávací dotaz")
    sources: List[str] = Field(default=["wikipedia", "pubmed"], description="Zdroje pro vyhledávání")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximální počet výsledků")
    use_cache: bool = Field(default=True, description="Použít cache")

class AuthRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

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

def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    return security_manager.get_client_ip(request)

# FastAPI aplikace
app = FastAPI(
    title="🔬 Academic Research Tool - Enterprise Edition",
    description="Pokročilá platforma pro akademický výzkum s MCP integrací",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware konfigurace
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # V produkci specifikovat konkrétní domény
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # V produkci specifikovat konkrétní hosty
)

# Rate limiting middleware
limiter = Limiter(key_func=get_client_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=429,
    content={"detail": "Rate limit exceeded", "retry_after": exc.retry_after}
))
app.add_middleware(SlowAPIMiddleware)

# Middleware pro metriky a security
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware pro Prometheus metriky a security"""
    start_time = time.time()
    client_ip = get_client_ip(request)

    # Rate limiting check
    if not security_manager.check_rate_limit(client_ip):
        RATE_LIMIT_EXCEEDED.labels(source="global").inc()
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"}
        )

    # Process request
    response = await call_next(request)

    # Record metrics
    process_time = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_DURATION.observe(process_time)

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    return response

# Startup a shutdown eventy
@app.on_event("startup")
async def startup_event():
    """Inicializace při spuštění"""
    logger.info("🚀 Spouštím Academic Research Tool Enterprise Edition")

    # Inicializace cache systému
    await initialize_cache()
    logger.info("✅ Cache systém inicializován")

    # Registrace MCP nástrojů do remote handleru
    if SCRAPER_AVAILABLE:
        remote_mcp_handler.register_tool("academic_search", academic_search_tool)
        logger.info("✅ MCP nástroje registrovány")

    logger.info("🎯 Server je připraven k použití")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup při vypnutí"""
    logger.info("🔄 Vypínám server...")
    # Zde by bylo cleanup připojení k databázím, cache atd.

# Authentication endpoints
@app.post("/auth/login", response_model=TokenResponse)
async def login(auth_request: AuthRequest):
    """Přihlášení uživatele"""
    # V produkci by zde byla validace proti databázi
    if auth_request.username == "admin" and auth_request.password == "admin":
        access_token = security_manager.create_access_token(
            data={"sub": auth_request.username, "permissions": ["admin"]}
        )
        refresh_token = security_manager.create_refresh_token(
            data={"sub": auth_request.username}
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=3600
        )
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/auth/refresh")
async def refresh_token(refresh_token: str):
    """Obnovení access tokenu"""
    try:
        payload = security_manager.verify_token(refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")

        new_access_token = security_manager.create_access_token(
            data={"sub": payload["sub"], "permissions": ["admin"]}
        )

        return {"access_token": new_access_token, "token_type": "bearer"}
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    components = {
        "cache": cache_manager.redis_client is not None,
        "scraper": SCRAPER_AVAILABLE,
        "mcp": MCP_AVAILABLE,
        "security": True
    }

    return HealthResponse(
        status="healthy" if all(components.values()) else "degraded",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        version="2.0.0",
        components=components
    )

# Academic search endpoint s pokročilými funkcemi
@app.post("/api/search")
@rate_limited("api")
async def enhanced_search(
    search_request: SearchRequest,
    request: Request,
    current_user = Depends(security_manager.get_current_user)
):
    """Pokročilé akademické vyhledávání s cache a rate limiting"""

    # Sanitizace vstupu
    query = security_manager.sanitize_input(search_request.query)
    query = security_manager.validate_input_length(query, 500)

    # Cache lookup
    cache_key = f"search_{hashlib.md5(f'{query}_{search_request.sources}'.encode()).hexdigest()}"

    if search_request.use_cache:
        cached_result = await cache_manager.get_research_data("combined", cache_key)
        if cached_result:
            CACHE_HITS.labels(cache_type="search").inc()
            return JSONResponse(content=cached_result)

    # ETag handling
    etag = etag_for_response(search_request.model_dump())
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304)

    if not SCRAPER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Search service unavailable")

    try:
        # Vytvoř orchestrátor a spusť vyhledávání
        orchestrator = await create_scraping_orchestrator()
        results = []

        for source in search_request.sources:
            try:
                source_results = await orchestrator.scrape_source(
                    source=source,
                    query=query,
                    max_results=search_request.max_results
                )
                results.extend(source_results)
            except Exception as e:
                logger.error(f"Error scraping {source}: {e}")
                continue

        # Připrav response
        response_data = {
            "query": query,
            "sources": search_request.sources,
            "results": [result.model_dump() if hasattr(result, 'model_dump') else result for result in results[:search_request.max_results]],
            "total_results": len(results),
            "cached": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }

        # Cache result
        if search_request.use_cache and results:
            await cache_manager.cache_research_data("combined", cache_key, response_data)

        response = JSONResponse(content=response_data)
        response.headers["ETag"] = etag
        return response

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# MCP tool pro remote handler
async def academic_search_tool(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """MCP nástroj pro akademické vyhledávání"""
    query = parameters.get("query", "")
    sources = parameters.get("sources", ["wikipedia"])
    max_results = parameters.get("max_results", 10)

    if not SCRAPER_AVAILABLE:
        return {"error": "Scraper service unavailable"}

    try:
        orchestrator = await create_scraping_orchestrator()
        results = []

        for source in sources:
            source_results = await orchestrator.scrape_source(
                source=source,
                query=query,
                max_results=max_results
            )
            results.extend(source_results)

        return {
            "query": query,
            "results": [result.model_dump() if hasattr(result, 'model_dump') else result for result in results],
            "total_results": len(results)
        }
    except Exception as e:
        return {"error": str(e)}

# Statistics endpoint
@app.get("/api/stats")
async def get_statistics(current_user = Depends(security_manager.get_current_user)):
    """Získá statistiky systému"""
    try:
        stats = {
            "cache": await cache_manager.get_stats(),
            "rate_limiter": await rate_limiter.get_statistics(),
            "system": {
                "uptime": time.time() - startup_time if 'startup_time' in globals() else 0,
                "version": "2.0.0"
            }
        }
        return stats
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

# Prometheus metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

# Setup remote MCP routes
setup_remote_mcp_routes(app)

# Include MCP router if available
if MCP_AVAILABLE and mcp_router:
    app.include_router(mcp_router, prefix="/mcp", tags=["MCP"])

# Global startup time tracking
startup_time = time.time()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "unified_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )