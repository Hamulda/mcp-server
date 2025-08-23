# Unified Biohacking Research API - Produkční Příručka

## 🔒 KRITICKÉ BEZPEČNOSTNÍ NASTAVENÍ PRO PRODUKCI

### ⚠️ PŘED NASAZENÍM DO PRODUKCE ZMĚŇ:

1. **Grafana heslo** - v docker-compose.yml změň `GRAFANA_ADMIN_PASSWORD`
2. **SECRET_KEY** - nastav environment proměnnou `SECRET_KEY`
3. **CORS origins** - omezobyj `CORS_ORIGINS` na konkrétní domény
4. **Trusted hosts** - nastav `TrustedHostMiddleware` allowed_hosts

```bash
# Nastavení produkčních proměnných
export GRAFANA_ADMIN_PASSWORD="your_secure_password_here"
export SECRET_KEY="your_secure_secret_key_here"
export ENVIRONMENT="production"
```

## 🚀 Implementované Optimalizace

### ✅ Bezpečnostní Vylepšení
- ✅ **Fixované verze závislostí** - odstraněny `>=` operátory
- ✅ **.dockerignore** - optimalizace Docker image velikosti
- ✅ **Bezpečnostní validace CLI** - sanitizace všech vstupů
- ✅ **Rate limiting** - pokročilý systém s IP blokováním  
- ✅ **JWT authentication** - bezpečná správa tokenů
- ✅ **Input validation** - ochrana proti injection útokům
- ✅ **Audit logging** - kompletní bezpečnostní log

### ✅ Performance Optimalizace
- ✅ **Redis cache systém** - nahrazuje SQLite pro lepší výkon
- ✅ **Sémantické vyhledávání** - ChromaDB + sentence-transformers
- ✅ **Connection pooling** - optimalizované DB připojení
- ✅ **Async everywhere** - plně asynchronní architektura
- ✅ **Smart caching strategies** - inteligentní TTL management

### ✅ Monitoring & Observability  
- ✅ **Prometheus metriky** - kompletní monitoring stack
- ✅ **Health checks** - detailní kontrola všech komponent
- ✅ **Performance tracking** - real-time statistiky
- ✅ **Error tracking** - centralizované error handling
- ✅ **Security monitoring** - real-time threat detection

### ✅ Docker Optimalizace
- ✅ **Multi-stage builds** - minimální image velikost
- ✅ **Sjednocené porty** - konzistentní 8000 napříč systémem
- ✅ **Bezpečné environment proměnné** - .env.example template
- ✅ **Optimalizované layery** - rychlejší build časy

## 🏗️ Architektura

```
unified_biohack_api/
├── core/                          # Hlavní aplikační logika
│   ├── main.py                   # CLI interface s security
│   └── unified_server.py         # FastAPI server
├── security/                      # Bezpečnostní komponenty
│   └── enhanced_security_manager.py
├── monitoring/                    # Monitoring & metriky
│   └── advanced_monitoring_system.py
├── cache/                        # Cache systémy
│   ├── redis_cache_system.py    # Redis implementace
│   └── unified_cache_system.py  # Fallback cache
├── ai/                           # AI komponenty
│   ├── semantic_search_system.py # Sémantické vyhledávání
│   └── local_ai_adapter.py      # Ollama integrace
└── docker/                      # Docker konfigurace
    ├── Dockerfile               # Optimalizovaný multi-stage
    ├── docker-compose.yml       # Bezpečná konfigurace
    └── .dockerignore            # Optimalizace image
```

## 🔧 Rychlý Start

### 1. Klonování a Instalace
```bash
git clone <repository>
cd PythonProject2

# Nastavení environmentu
cp .env.example .env
# Edituj .env s produkčními hodnotami

# Instalace závislostí (fixované verze)
pip install -r requirements.txt
```

### 2. Spuštění Development
```bash
# Spuštění s Dockerem (doporučeno)
docker-compose up -d

# Nebo lokálně
python core/unified_server.py
```

### 3. Ověření Funkčnosti
```bash
# Health check
curl http://localhost:8000/health

# Test research
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "BPC-157 benefits", "research_type": "comprehensive"}'

# Monitoring
curl http://localhost:8000/metrics
```

## 📊 Monitoring Endpoints

| Endpoint | Popis | Autentifikace |
|----------|-------|---------------|
| `/health` | Kompletní health check | Ne |
| `/metrics` | Prometheus metriky | Ne |
| `/security/report` | Bezpečnostní report | Rate limited |
| `/stats/performance` | Performance statistiky | Rate limited |

## 🔐 Rate Limiting

Implementované limity pro různé endpointy:

| Endpoint | Per Minute | Per Hour | Burst |
|----------|------------|----------|-------|
| `/research` | 20 | 100 | 5 |
| `/peptide` | 15 | 75 | 3 |
| `/health` | 60 | 300 | 10 |
| `/auth` | 5 | 20 | 2 |
| Default | 30 | 150 | 5 |

## 🧬 API Usage Examples

### Research Query
```python
import requests

response = requests.post("http://localhost:8000/research", json={
    "query": "NAD+ supplementation longevity",
    "research_type": "comprehensive",
    "evidence_level": "high",
    "include_safety": True,
    "output_format": "detailed"
})

result = response.json()
```

### Peptide Research
```python
response = requests.post("http://localhost:8000/peptide", json={
    "peptide_name": "BPC-157",
    "research_focus": "dosage"
})

result = response.json()
```

### Semantic Search
```python
response = requests.post("http://localhost:8000/search/semantic", params={
    "query": "peptide healing properties",
    "n_results": 10,
    "source_filter": "peptide_research"
})

results = response.json()
```

## 📈 Performance Metriky

Systém automaticky trackuje:

- **Request metrics**: latence, throughput, error rate
- **Cache metrics**: hit rate, miss rate, evictions  
- **System metrics**: CPU, paměť, disk usage
- **Security metrics**: rate limits, blocked IPs, suspicious activity
- **AI metrics**: model usage, response times

## 🐛 Troubleshooting

### Časté Problémy

1. **Redis nedostupný**
   - Systém automaticky fallback na in-memory cache
   - Check: `docker-compose logs redis`

2. **Semantic search nefunguje**
   - Zkontroluj závislosti: `pip install sentence-transformers chromadb`
   - Check: ChromaDB data directory permissions

3. **Rate limiting příliš přísný**
   - Upravit v `enhanced_security_manager.py`
   - Restart aplikace pro aplikování změn

4. **Ollama AI nedostupný**
   - Systém běží v "basic mode"
   - Install Ollama a stáhni modely: `ollama pull llama3.1:8b`

## 🔄 Deployment Checklist

### Production Readiness
- [ ] SECRET_KEY nastaven
- [ ] GRAFANA_ADMIN_PASSWORD změněn
- [ ] CORS_ORIGINS omezen na produkční domény
- [ ] Rate limits zkontrolovány
- [ ] Monitoring nakonfigurován
- [ ] Backup strategie definována
- [ ] SSL/TLS certifikáty
- [ ] Log rotation nastaven

### Performance Tuning
- [ ] Redis cluster pro high availability
- [ ] Load balancing konfigurace
- [ ] Database connection pooling optimalizace
- [ ] CDN pro statické soubory
- [ ] Compression middleware

## 📝 Changelog

### v2.0.0 - Kompletní Optimalizace
- ✅ Fixované verze závislostí
- ✅ .dockerignore optimalizace
- ✅ Bezpečnostní validace CLI
- ✅ Redis cache systém
- ✅ Sémantické vyhledávání
- ✅ Enterprise monitoring
- ✅ Rate limiting & security
- ✅ Sjednocené porty (8000)
- ✅ Multi-stage Docker builds

## 🤝 Contributing

Pro development změny:

1. Fork repository
2. Vytvoř feature branch
3. Implementuj změny s testy
4. Spusť security scan: `python -m safety check`
5. Update dokumentaci
6. Vytvoř pull request

## 📞 Support

Pro podporu nebo bug reports:
- Zkontroluj logs: `docker-compose logs api`
- Security events: GET `/security/report`
- Health status: GET `/health`
- Performance stats: GET `/stats/performance`
