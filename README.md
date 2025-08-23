# 🧬 Advanced Academic Research Tool with MCP Integration

> **Production-ready akademický výzkumný nástroj s Model Context Protocol (MCP) podporou, optimalizovaný pro M1 MacBook a enterprise nasazení.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![MCP](https://img.shields.io/badge/MCP-enabled-green.svg)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## ✨ Klíčové funkce

### 🔬 **Pokročilý akademický výzkum**
- **Multi-source scraping**: Wikipedia, PubMed, OpenAlex s inteligentním rate limitingem
- **AI-powered analýza**: Automatické hodnocení kvality zdrojů a důkazů
- **Safety assessment**: Inteligentní bezpečnostní hodnocení výzkumných témat
- **Intelligent caching**: Pokročilý cache systém s kompresí a automatickou expirací

### 🤖 **Model Context Protocol (MCP) integrace**
- **5 pokročilých MCP serverů**: Web automation, Brave Search, GitHub, Fetch, Puppeteer
- **Tool-based architektura** pro seamless AI agent integraci
- **RESTful API** s komprehenzivními endpointy
- **Real-time výzkumné schopnosti**

### ⚡ **M1 MacBook optimalizace**
- **Memory-efficient caching** s detekci tlaku paměti
- **Energy-optimized** retry logika a connection pooling
- **Thread-safe implementace** napříč všemi komponenty
- **Minimální resource footprint**

### 🏭 **Production-ready funkce**
- **Docker kontejnerizace** s health checks a multi-stage buildy
- **Prometheus monitoring** a Grafana dashboardy
- **Comprehensive error handling** a structured logging
- **Scalable unified architektura**

## 🚀 Rychlý start

### Předpoklady
- Python 3.11+
- Docker & Docker Compose
- Git

### 1. Instalace a setup
```bash
git clone https://github.com/Hamulda/advanced-academic-research-tool.git
cd advanced-academic-research-tool
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Základní použití
```bash
# Rychlý výzkumný dotaz
python core/main.py --query "BPC-157 dosing protocol" --type comprehensive --verbose

# Bezpečnostní hodnocení
python core/main.py --query "Nootropics safety" --type safety --evidence high

# JSON výstup pro API integraci
python core/main.py --query "Peptide research" --format json
```

### 3. Docker deployment
```bash
# Development
docker-compose up -d

# Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 📖 Pokročilé použití

### Research módy
- `quick` - Rychlé vyhledávání s cache prioritou
- `balanced` - Vyvážený přístup rychlost/kvalita
- `comprehensive` - Hloubkový výzkum všech zdrojů
- `safety` - Zaměřeno na bezpečnostní aspekty
- `dosage` - Specializace na dávkování a protokoly

### Evidence levely
- `high` - Pouze peer-reviewed a high-impact zdroje
- `medium` - Včetně kvalitních sekundárních zdrojů
- `all` - Všechny dostupné zdroje s hodnocením

### Output formáty
- `brief` - Stručné shrnutí s klíčovými body
- `detailed` - Kompletní analýza s odkazy
- `expert` - Technický formát pro odborníky
- `json` - Strukturovaná data pro API

## 🔧 Architektura

```
┌─────────────────┬──────────────────┬─────────────────┐
│   Core Engine   │   MCP Servers    │   Monitoring    │
├─────────────────┼──────────────────┼─────────────────┤
│ • Main App      │ • Web Automation │ • Prometheus    │
│ • Research Orch │ • Brave Search   │ • Grafana       │
│ • Cache System  │ • GitHub Tools   │ • Health Checks │
│ • Config Mgmt   │ • Fetch Tools    │ • Performance   │
└─────────────────┴──────────────────┴─────────────────┘
```

### Klíčové komponenty
- **UnifiedBiohackingResearchTool**: Hlavní orchestrátor
- **IntelligentResearchOrchestrator**: AI-powered research engine
- **UnifiedCacheManager**: Pokročilý cache systém
- **WebAutomationMCPServer**: Web scraping a automatizace
- **AcademicScraper**: Specializovaný scraper pro vědecké zdroje

## 📊 Performance optimalizace

### Cache systém
```python
# Automatické cache s TTL
@cached(ttl=3600, key_prefix="research")
async def research_function(query: str):
    return await expensive_operation(query)

# Manuální cache operace
cache = get_cache_manager()
await cache.set("key", data, ttl=1800)
result = await cache.get("key")
```

### Rate limiting
- Exponential backoff pro API calls
- Per-domain rate limiting
- Circuit breaker pattern pro API ochranu
- Intelligent retry s jitter

## 🔐 Bezpečnost

- **Input sanitization**: Ochrana proti injection útokům
- **SSL/TLS**: Secure komunikace s external APIs
- **Non-root Docker**: Bezpečnostní hardening kontejnerů
- **Rate limiting**: Ochrana proti abuse
- **Audit logging**: Kompletní audit trail

## 📈 Monitoring a metriky

### Prometheus metriky
- Request latency a throughput
- Cache hit/miss ratios
- Error rates po komponentách
- Resource utilization

### Grafana dashboardy
- Real-time performance monitoring
- System health overview
- Research query analytics
- Cache performance metrics

Přístup na `http://localhost:3000` (admin/admin123)

## 🧪 Testování

```bash
# Spuštění všech testů
python optimized_test_suite.py

# Performance benchmark
python -c "from optimized_test_suite import run_performance_benchmark; run_performance_benchmark()"

# Integration testy
pytest tests/ -v

# Type checking
mypy core/ mcp_servers/ cache/
```

## 🛠️ Development

### Struktura projektu
```
├── core/                   # Hlavní aplikační logika
├── mcp_servers/           # MCP server implementace
├── cache/                 # Cache systémy
├── monitoring/            # Monitoring konfigurace
├── optimization/          # Performance optimalizace
├── tests/                 # Test suites
└── docker-compose.yml     # Orchestrace služeb
```

### Přidání nových funkcí
1. Vytvořte feature branch
2. Implementujte s testy
3. Ověřte performance impact
4. Aktualizujte dokumentaci
5. Vytvořte pull request

## 📝 API dokumentace

### REST endpoints
```bash
# Health check
GET /health

# Research endpoint
POST /research
{
  "query": "research topic",
  "type": "comprehensive",
  "evidence_level": "high"
}

# Cache statistics
GET /cache/stats

# Performance metrics
GET /metrics
```

### MCP Tools
- `scrape_url`: Web scraping s CSS selektory
- `extract_links`: Link extrakce s filtering
- `browser_screenshot`: Screenshot capture
- `get_page_info`: Detailní page analýza
- `search_academic`: Akademické databáze search

## 🚀 Production deployment

### Docker Compose služby
- **research-app**: Hlavní aplikace
- **redis**: Cache a session storage
- **prometheus**: Metrics collection
- **grafana**: Visualization
- **nginx**: Reverse proxy a load balancing

### Environment variables
```bash
ENVIRONMENT=production
REDIS_URL=redis://redis:6379
DATABASE_URL=sqlite:///data/app.db
PROMETHEUS_URL=http://prometheus:9090
```

## 🤝 Přispívání

1. Fork repository
2. Vytvořte feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit změny (`git commit -m 'Add AmazingFeature'`)
4. Push do branch (`git push origin feature/AmazingFeature`)
5. Otevřete Pull Request

## 📄 Licence

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Poděkování

- [Model Context Protocol](https://modelcontextprotocol.io/) za MCP framework
- [FastAPI](https://fastapi.tiangolo.com/) za výkonný web framework
- [Playwright](https://playwright.dev/) za browser automation
- [Prometheus](https://prometheus.io/) a [Grafana](https://grafana.com/) za monitoring

## 📞 Kontakt

Vojtěch Hamada - [@Hamulda](https://github.com/Hamulda)

Project Link: [https://github.com/Hamulda/advanced-academic-research-tool](https://github.com/Hamulda/advanced-academic-research-tool)

---

**⚠️ Disclaimer**: Tento nástroj je určen pouze pro výzkumné účely. Vždy konzultujte s odborníky před implementací jakýchkoli doporučení.
