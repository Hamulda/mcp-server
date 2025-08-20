# 🔬 Academic Research Tool with MCP Integration

> **Production-ready academic content scraping and research platform with Model Context Protocol (MCP) support, optimized for M1 MacBook.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![MCP](https://img.shields.io/badge/MCP-enabled-green.svg)](https://modelcontextprotocol.io/)

## ✨ Features

### 🔍 **Academic Research**
- **Multi-source scraping**: Wikipedia, PubMed, OpenAlex
- **Intelligent rate limiting** with exponential backoff
- **Circuit breaker pattern** for API protection
- **Async/concurrent processing** for optimal performance

### 🤖 **MCP Integration**
- **Model Context Protocol** server for AI agents
- **Tool-based architecture** for seamless AI integration
- **RESTful API** with comprehensive endpoints
- **Real-time research capabilities**
- **5 Advanced MCP Servers**: Brave Search, Puppeteer, Git, Fetch, GitHub

### 🔧 **M1 MacBook Optimized**
- **Memory-efficient caching** with pressure detection
- **Energy-optimized** retry logic and connection pooling
- **Thread-safe implementations** across all components
- **Minimal resource footprint**

### 📊 **Production Features**
- **Docker containerization** with health checks
- **Prometheus monitoring** and Grafana dashboards
- **Comprehensive error handling** and logging
- **Scalable unified architecture**

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git

### 1. Clone & Setup
```bash
git clone https://github.com/Hamulda/mcp-server.git
cd mcp-server
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run with Docker (Recommended)
```bash
# Start all services (API + Monitoring)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f academic-research-api
```

### 3. Local Development
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run locally
python unified_server.py
```

## 📡 API Endpoints

### Core Endpoints
- **Health Check**: `GET /health`
- **Scraping**: `POST /api/v1/scrape`
- **Sources**: `GET /api/v1/sources`
- **Metrics**: `GET /metrics` (Prometheus)

### MCP Server
- **MCP Tools**: `/mcp/*` (for AI agents)

### Monitoring
- **API**: http://localhost:8080
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9091

## 📖 Usage Examples

### Basic Research Query
```bash
curl -X POST http://localhost:8080/api/v1/scrape \
  -H "Content-Type: application/json" \
  -d '{
    "query": "peptides metabolism",
    "sources": ["wikipedia", "pubmed"]
  }'
```

### Python Integration
```python
import aiohttp
import asyncio

async def research_peptides():
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:8080/api/v1/scrape',
            json={
                'query': 'biohacking peptides',
                'sources': ['wikipedia', 'pubmed', 'openalex']
            }
        ) as response:
            return await response.json()

# Run research
results = asyncio.run(research_peptides())
```

### MCP Integration (for AI Agents)
```json
{
  "mcpServers": {
    "academic-research": {
      "command": "http",
      "args": ["http://localhost:8080/mcp"]
    }
  }
}
```

## 🏗️ Architecture

```
┌──��──────────────┬─────────────────┬─────────────────┐
│   FastAPI       │   MCP Server    │   Monitoring    │
│   (Port 8080)   │   (/mcp)        │   (Grafana)     │
└─────────────────┴─────────────────┴─────────────────┘
┌─────────────────────────────────────────────────────┐
│              Unified Orchestrator                   │
├─────────────────┬─────────────────┬─────────────────┤
│  Wikipedia      │    PubMed       │   OpenAlex      │
│  Scraper        │    Scraper      │   Scraper       │
└─────────────────┴─────────────────┴─────────────────┘
┌─────────────────────────────────────────────────────┐
│              Intelligent Cache System               │
│         (Memory + SQLite + M1 Optimized)           │
└─────────────────────────────────────────────────────┘
```

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=development

# Rate Limiting
RATE_LIMIT_SCRAPE=30/minute

# Cache Settings
CACHE_TTL=3600
CACHE_MAX_SIZE=1000

# Enable/Disable Sources
WIKIPEDIA_ENABLED=true
PUBMED_ENABLED=true
OPENALEX_ENABLED=true
```

### Docker Configuration
Edit `docker-compose.yml` for production deployment:
- Resource limits
- Environment variables
- Volume mounts
- Network settings

## 📊 Monitoring & Metrics

### Available Metrics
- **Request counts** by source and endpoint
- **Response times** and performance
- **Cache hit/miss ratios**
- **Error rates** and circuit breaker status
- **System resources** (CPU, memory)

### Grafana Dashboards
Pre-configured dashboards for:
- API performance monitoring
- Scraper health and success rates
- Cache efficiency metrics
- System resource utilization

## 🤖 MCP Servers

### 📦 Installed MCP Servers
Your project now includes 5 powerful MCP servers for AI agent integration:

1. **🔍 Brave Search** - Web search, news, images, videos
2. **🕷️ Puppeteer** - Web scraping and browser automation  
3. **📦 Git** - Git repository operations and management
4. **🌐 Fetch** - HTTP requests and API calls
5. **🐙 GitHub** - GitHub API integration (requires Go)

### Quick MCP Setup
```bash
# List all available MCP servers
./mcp-servers.sh list

# Build all servers
./mcp-servers.sh build

# Start a specific server
./mcp-servers.sh start brave-search
```

See [MCP_SERVERS.md](MCP_SERVERS.md) for detailed setup and configuration.

## 🛠️ Development

### Running Tests
```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

### Project Structure
```
├── unified_server.py          # Main FastAPI server
├── academic_scraper.py        # Scraping orchestrator
├── unified_config.py          # Configuration management
├── unified_cache_system.py    # Intelligent caching
├── mcp_handler.py            # MCP server implementation
├── docker-compose.yml        # Docker services
├── monitoring/               # Prometheus & Grafana
├── tests/                    # Test suite
└── docs/                     # Documentation
```

## 🚀 Deployment

### Production Checklist
- [ ] Set environment variables
- [ ] Configure resource limits in Docker
- [ ] Enable SSL/TLS
- [ ] Set up proper logging
- [ ] Configure monitoring alerts
- [ ] Test all endpoints
- [ ] Verify MCP integration

### Docker Production
```bash
# Production build
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale academic-research-api=3
```

## 📈 Performance

### Benchmarks (M1 MacBook Air)
- **API Response Time**: ~400ms average
- **Memory Usage**: <200MB base
- **Concurrent Requests**: 50+ simultaneous
- **Cache Hit Rate**: 85%+ typical

### Optimization Features
- **Circuit breakers** prevent cascade failures
- **Exponential backoff** for rate limiting
- **Connection pooling** reduces overhead
- **Smart caching** with memory pressure detection

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/Hamulda/mcp-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Hamulda/mcp-server/discussions)

## 🙏 Acknowledgments

- **FastAPI** for the excellent web framework
- **Model Context Protocol** for AI integration standards
- **OpenAlex**, **PubMed**, **Wikipedia** for academic data access

---

**Built with ❤️ for academic research and AI integration**
