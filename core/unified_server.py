"""
Unified Server - Hlavní FastAPI server s integrovanými optimalizacemi
Obsahuje všechny security, monitoring a performance optimalizace
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Importy našich optimalizovaných komponent
from core.main import UnifiedBiohackingResearchTool
from security.enhanced_security_manager import get_security_manager
from monitoring.advanced_monitoring_system import get_monitoring_system, start_monitoring, stop_monitoring
from cache.redis_cache_system import get_redis_cache, cleanup_cache
from ai.semantic_search_system import get_semantic_search, cleanup_semantic_search
from unified_config import get_config

logger = logging.getLogger(__name__)

# Pydantic modely pro API
class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Research query")
    research_type: str = Field("comprehensive", description="Type of research")
    evidence_level: str = Field("high", description="Required evidence level")
    include_safety: bool = Field(True, description="Include safety assessment")
    output_format: str = Field("detailed", description="Output format")

class PeptideRequest(BaseModel):
    peptide_name: str = Field(..., min_length=1, max_length=50, description="Peptide name")
    research_focus: str = Field("comprehensive", description="Research focus area")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    components: Dict[str, Any]

# Security dependency
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Ověření uživatele - volitelné pro public API"""
    if credentials:
        security_manager = get_security_manager()
        payload = security_manager.verify_token(credentials.credentials)
        if payload:
            return payload.get("sub", "authenticated_user")
    return "anonymous_user"

async def check_security_and_rate_limit(request: Request, user_id: str = Depends(get_current_user)):
    """Middleware pro security a rate limiting"""
    security_manager = get_security_manager()
    monitoring = get_monitoring_system()

    # Získej IP adresu
    client_ip = request.client.host
    endpoint = request.url.path

    # Rate limiting check
    rate_limit_result = await security_manager.check_rate_limit(
        identifier=user_id or client_ip,
        endpoint=endpoint.split('/')[-1],  # Posledni část path
        ip_address=client_ip
    )

    if not rate_limit_result["allowed"]:
        monitoring.record_request(endpoint, request.method, 429, 0.0)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "reason": rate_limit_result["reason"],
                "retry_after": rate_limit_result.get("retry_after", 60)
            }
        )

    return {"user_id": user_id, "ip_address": client_ip, "rate_limit": rate_limit_result}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management pro aplikaci"""
    logger.info("🚀 Starting Unified Biohacking Research Server")

    # Spuštění monitoring systému
    await start_monitoring()

    # Inicializace cache a semantic search
    try:
        await get_redis_cache()
        await get_semantic_search()
        logger.info("✅ All systems initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ Some components failed to initialize: {e}")

    yield

    # Cleanup při shutdown
    logger.info("🛑 Shutting down server")
    await stop_monitoring()
    await cleanup_cache()
    await cleanup_semantic_search()

# Vytvoření FastAPI aplikace
app = FastAPI(
    title="Unified Biohacking Research API",
    description="Advanced biohacking research tool with AI, caching, and monitoring",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware konfigurace
config = get_config()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # V produkci nastavit konkrétní domény
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # V produkci nastavit konkrétní hosty
)

# Request/Response middleware pro monitoring
@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    """Middleware pro monitoring všech requestů"""
    start_time = time.time()
    monitoring = get_monitoring_system()

    # Zpracování requestu
    response = await call_next(request)

    # Zaznamenání metriky
    duration = time.time() - start_time
    monitoring.record_request(
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code,
        duration=duration
    )

    # Přidání performance headers
    response.headers["X-Response-Time"] = f"{duration:.3f}s"
    response.headers["X-Server"] = "Unified-Biohack-API"

    return response

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Comprehensive health check endpoint"""
    monitoring = get_monitoring_system()
    health_status = monitoring.get_health_status()

    return HealthResponse(
        status=health_status["overall_status"],
        timestamp=datetime.now().isoformat(),
        uptime_seconds=health_status["uptime_seconds"],
        components=health_status
    )

# Prometheus metrics endpoint
@app.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    monitoring = get_monitoring_system()
    return monitoring.get_prometheus_metrics()

# Security report endpoint
@app.get("/security/report", tags=["Security"])
async def security_report(
    hours: int = 24,
    security_context = Depends(check_security_and_rate_limit)
):
    """Security events report"""
    security_manager = get_security_manager()
    report = security_manager.get_security_report(hours=hours)
    return report

# Main research endpoint
@app.post("/research", tags=["Research"])
async def research_endpoint(
    request: ResearchRequest,
    security_context = Depends(check_security_and_rate_limit)
):
    """
    Advanced research endpoint with full optimization stack
    """
    monitoring = get_monitoring_system()
    security_manager = get_security_manager()
    user_id = security_context["user_id"]

    # Input validation
    validation_result = security_manager.validate_input(request.query)
    if not validation_result["valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid input",
                "reason": validation_result["reason"],
                "patterns": validation_result.get("patterns", [])
            }
        )

    # Cache check
    cache = await get_redis_cache()
    cached_result = await cache.get_research_cache(
        query=validation_result["sanitized"],
        source="api_research"
    )

    if cached_result:
        monitoring.record_cache_operation("research", hit=True)
        monitoring.record_research_query(request.research_type, success=True)
        return {
            "result": cached_result["result"],
            "cached": True,
            "cached_at": cached_result["cached_at"]
        }

    monitoring.record_cache_operation("research", hit=False)

    # Provedení research
    try:
        async with UnifiedBiohackingResearchTool(user_id=user_id) as research_tool:
            result = await research_tool.research(
                query=validation_result["sanitized"],
                research_type=request.research_type,
                evidence_level=request.evidence_level,
                include_safety=request.include_safety,
                output_format=request.output_format
            )

        # Cache výsledek
        await cache.set_research_cache(
            query=validation_result["sanitized"],
            result=result,
            source="api_research"
        )

        monitoring.record_research_query(request.research_type, success=True)

        return {
            "result": result,
            "cached": False,
            "processed_at": datetime.now().isoformat()
        }

    except Exception as e:
        monitoring.record_research_query(request.research_type, success=False)
        logger.error(f"Research failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Research failed", "message": str(e)}
        )

# Peptide research endpoint
@app.post("/peptide", tags=["Research"])
async def peptide_research_endpoint(
    request: PeptideRequest,
    security_context = Depends(check_security_and_rate_limit)
):
    """
    Specialized peptide research endpoint
    """
    monitoring = get_monitoring_system()
    security_manager = get_security_manager()
    user_id = security_context["user_id"]

    # Validate peptide name
    validation_result = security_manager.validate_input(request.peptide_name, max_length=50)
    if not validation_result["valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid peptide name",
                "reason": validation_result["reason"]
            }
        )

    # Peptide cache check
    cache = await get_redis_cache()
    cached_result = await cache.get_peptide_cache(validation_result["sanitized"])

    if cached_result:
        monitoring.record_cache_operation("peptide", hit=True)
        return {
            "result": cached_result["research_data"],
            "cached": True,
            "cached_at": cached_result["cached_at"]
        }

    monitoring.record_cache_operation("peptide", hit=False)

    # Provedení peptide research
    try:
        async with UnifiedBiohackingResearchTool(user_id=user_id) as research_tool:
            result = await research_tool.peptide_research(
                peptide_name=validation_result["sanitized"],
                research_focus=request.research_focus
            )

        # Cache výsledek
        await cache.set_peptide_cache(
            peptide_name=validation_result["sanitized"],
            research_data=result
        )

        monitoring.record_research_query("peptide", success=True)

        return {
            "result": result,
            "cached": False,
            "processed_at": datetime.now().isoformat()
        }

    except Exception as e:
        monitoring.record_research_query("peptide", success=False)
        logger.error(f"Peptide research failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Peptide research failed", "message": str(e)}
        )

# Semantic search endpoint
@app.post("/search/semantic", tags=["Search"])
async def semantic_search_endpoint(
    query: str,
    n_results: int = 10,
    source_filter: Optional[str] = None,
    security_context = Depends(check_security_and_rate_limit)
):
    """
    Semantic search endpoint
    """
    security_manager = get_security_manager()

    # Input validation
    validation_result = security_manager.validate_input(query)
    if not validation_result["valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid search query",
                "reason": validation_result["reason"]
            }
        )

    try:
        semantic_search = await get_semantic_search()
        results = await semantic_search.search(
            query=validation_result["sanitized"],
            n_results=min(n_results, 50),  # Limit max results
            source_filter=source_filter
        )

        return {
            "query": validation_result["sanitized"],
            "results": results,
            "total_found": len(results)
        }

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Semantic search failed", "message": str(e)}
        )

# Performance stats endpoint
@app.get("/stats/performance", tags=["Monitoring"])
async def performance_stats(
    security_context = Depends(check_security_and_rate_limit)
):
    """
    Current performance statistics
    """
    monitoring = get_monitoring_system()
    cache = await get_redis_cache()
    semantic_search = await get_semantic_search()

    return {
        "monitoring": monitoring.get_health_status(),
        "cache_stats": await cache.get_cache_stats(),
        "semantic_search_stats": await semantic_search.get_collection_stats(),
        "timestamp": datetime.now().isoformat()
    }

# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """
    API root with basic information
    """
    return {
        "service": "Unified Biohacking Research API",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Advanced research queries",
            "Peptide-specific research",
            "Semantic search",
            "Redis caching",
            "Rate limiting",
            "Security monitoring",
            "Prometheus metrics"
        ],
        "endpoints": {
            "health": "/health",
            "research": "/research",
            "peptide": "/peptide",
            "semantic_search": "/search/semantic",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    # Server konfigurace
    config = get_config()
    port = getattr(config, 'port', 8000)

    uvicorn.run(
        "unified_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # V produkci vypnout
        log_level="info"
    )
