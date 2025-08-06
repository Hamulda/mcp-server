# ResearchTool

Pokročilý nástroj pro výzkum a analýzu akademických zdrojů s optimalizací nákladů a vysokým výkonem.

## Funkce

- 🔍 **Inteligentní vyhledávání** - Google Scholar, PubMed, Semantic Scholar
- 🧠 **AI analýza** - Gemini API pro zpracování a sumarizaci textů
- 💰 **Optimalizace nákladů** - Pokročilé sledování a omezování výdajů
- ⚡ **Vysoký výkon** - Paralelní zpracování a inteligentní cache
- 📊 **Monitoring** - Grafana dashboardy a metriky
- 🎯 **Specializace na medicínu** - Optimalizováno pro nootropika, peptidy, medikace

## Rychlý start

### Požadavky
- Python 3.8+
- Docker a Docker Compose (pro monitoring)
- Gemini API klíč

### Instalace

1. Klonujte repozitář:
```bash
git clone https://github.com/yourusername/ResearchTool.git
cd ResearchTool
```

2. Vytvořte virtuální prostředí:
```bash
python -m venv .venv
source .venv/bin/activate  # Na Windows: .venv\Scripts\activate
```

3. Nainstalujte závislosti:
```bash
pip install -r requirements.txt
```

4. Nastavte API klíče:
```bash
cp config.py config_personal.py
# Upravte config_personal.py s vašimi API klíči
```

### Spuštění

#### Streamlit UI
```bash
streamlit run streamlit_app.py
```

#### Command Line
```bash
python main_fast.py
```

#### S monitoringem (Docker)
```bash
docker-compose up -d
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

## Konfigurace

Hlavní konfigurace v `config_personal.py`:

- `DAILY_COST_LIMIT`: Denní limit nákladů
- `GEMINI_RATE_LIMIT`: Omezení requestů za minutu
- `MAX_CONCURRENT_REQUESTS`: Paralelní požadavky
- `LOCAL_MODE`: Optimalizace pro lokální použití

## Struktura projektu

```
├── streamlit_app.py           # Web UI
├── main_fast.py              # CLI rozhraní
├── research_engine.py        # Hlavní výzkumný engine
├── gemini_manager.py         # Gemini API management
├── web_scraper.py           # Web scraping
├── academic_scraper.py      # Akademické zdroje
├── text_analyzer.py         # Analýza textu
├── cost_tracker.py          # Sledování nákladů
├── database_manager.py      # Databázové operace
├── monitoring/              # Grafana a Prometheus config
└── tests/                   # Testy
```

## Optimalizace nákladů

- **Token optimalizace**: Inteligentní zkracování textů
- **Caching**: Perzistentní cache pro opakované dotazy
- **Batch processing**: Skupinové zpracování pro efektivitu
- **Rate limiting**: Kontrola frekvence API volání

## Testování

```bash
# Jednotkové testy
pytest test_core_components.py

# Integrační testy
pytest test_integration.py

# Všechny testy
pytest
```

## Monitoring

Projekt obsahuje kompletní monitoring stack:

- **Prometheus**: Sběr metrik
- **Grafana**: Vizualizace dashboardů
- **Custom metriky**: Náklady, výkon, chyby

## Vývoj

### Pre-commit hooks
```bash
pip install -r requirements-dev.txt
pre-commit install
```

### Přidání nového scraperu
1. Vytvořte třídu dědící z `BaseScraper`
2. Implementujte `_scrape_search_results`
3. Přidejte do `DEFAULT_SOURCES` v config

## Licence

MIT License - viz LICENSE soubor

## Podpora

Pro otázky a problémy vytvořte GitHub issue.
