# 🔬 Academic Research Tool - Unified Version

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Production-ready akademický výzkumný nástroj s optimalizovanou unified architekturou pro MacBook Air M1 a lokální AI**

## 🎯 Klíčové výhody

- **💰 100% úspora nákladů**: Kompletně offline s lokálním AI (Ollama + Llama 3.1 8B)
- **🔒 Maximální privacy**: Žádné externí API, vše lokálně
- **⚡ M1 optimalizace**: Speciálně optimalizováno pro Apple Silicon
- **🌐 Univerzální domény**: Medicína, technologie, věda, byznys
- **🚀 Sjednocená architektura**: Jeden vstupní bod, minimální konfigurace

## 🏗️ Unified Architecture

### ✅ Hlavní komponenty
- `unified_main.py` - Hlavní vstupní bod a CLI
- `unified_config.py` - Centralizovaná konfigurace
- `unified_server.py` - FastAPI server
- `unified_research_engine.py` - Research engine s lokálním AI
- `academic_scraper.py` - Optimalizovaný scraper
- `local_ai_adapter.py` - Ollama/Llama 3.1 adapter

## 🚀 Quick Start

### 1. Instalace závislostí
```bash
pip install -r requirements.txt
```

### 2. Nastavení lokálního AI (Ollama + Llama 3.1)
```bash
# Instalace Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Stažení Llama 3.1 8B (doporučeno pro M1)
ollama pull llama3.1:8b

# Alternativně rychlejší model pro testing
ollama pull phi3:mini
```

### 3. Spuštění

#### CLI Interface
```bash
# Zobrazení konfigurace
python unified_main.py config show

# Scraping z CLI
python unified_main.py scrape "machine learning algorithms" --output results.json

# Spuštění serveru
python unified_main.py server

# System status
python unified_main.py status
```

#### Web Interface
```bash
# Spuštění unified serveru (doporučeno)
python unified_main.py server --type unified

# Server běží na http://localhost:8000
# API dokumentace: http://localhost:8000/docs
```

## 🔧 Konfigurace

Projekt používá unified konfiguraci v `unified_config.py` s automatickým načítáním z environment variables:

```bash
# API konfigurace
export API_HOST=localhost
export API_PORT=8000
export API_DEBUG=false

# Ollama konfigurace
export OLLAMA_HOST=http://localhost:11434
export PRIMARY_MODEL=llama3.1:8b

# Environment
export ENVIRONMENT=production
```

## 📊 Podporované zdroje

- **PubMed** - Medicínské výzkumy
- **arXiv** - Vědecké preprinty
- **Semantic Scholar** - Akademické publikace
- **OpenAlex** - Otevřené vědecké data
- **Wikipedia** - Obecné informace
- **CrossRef** - Publikační metadata

## 🎯 Use Cases

### 🏥 Medicína
```bash
python unified_main.py scrape "nootropika pro ADHD 2024"
```

### 💻 Technologie
```bash
python unified_main.py scrape "React performance optimization"
```

### 🔬 Věda
```bash
python unified_main.py scrape "climate change machine learning"
```

## 🧪 Testing

```bash
# Spuštění testů
python unified_main.py test

# Nebo přímo pytest
pytest -v
```

## 📈 Performance & Optimalizace

### M1 MacBook Air optimalizace:
- **RAM management**: Inteligentní model switching
- **Energy efficiency**: Auto-unload timeout
- **Neural Engine**: GPU acceleration pro Llama 3.1
- **Memory mapping**: Optimalizace pro větší modely

### Caching:
- Agresivní disk cache pro AI odpovědi
- ETags pro HTTP cache
- Persistent cache přes restarty

## 🔒 Privacy & Security

- **100% offline**: Žádné externí API calls
- **Lokální AI**: Všechny dotazy zůstávají na zařízení
- **No logging**: Žádné logování uživatelských dotazů
- **Local storage**: Všechna data lokálně

## 📁 Struktura projektu

```
├── unified_main.py          # Hlavní vstupní bod
├── unified_config.py        # Centralizovaná konfigurace
├── unified_server.py        # FastAPI server
├── unified_research_engine.py # Research engine
├── academic_scraper.py      # Optimalizovaný scraper
├── local_ai_adapter.py      # Ollama adapter
├── cache_manager.py         # Cache management
├── requirements.txt         # Python závislosti
└── tests/                   # Test suite
```

## 🚀 Development

```bash
# Development mode
python unified_main.py server --env development

# Production deployment
python unified_main.py server --env production
```

## 📚 API Documentation

Po spuštění serveru je dostupná interaktivní dokumentace:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🤝 Contributing

1. Fork repository
2. Vytvořte feature branch
3. Commitujte změny
4. Vytvořte Pull Request

## 📄 License

MIT License - viz LICENSE soubor pro detaily.
