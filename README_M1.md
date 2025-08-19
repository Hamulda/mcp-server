# M1 MacBook Research Tool - Kompletní Průvodce

## 🍎 Optimalizováno pro MacBook Air M1 + Phi-3 Mini

Tento projekt je kompletně optimalizován pro **MacBook Air M1** s lokálním AI modelem **Phi-3 Mini**. Prioritizuje privacy, offline funkčnost a energetickou efektivitu.

## ⚡ Quick Start

```bash
# 1. Spusť automatický setup (doporučeno)
chmod +x setup_m1.sh
./setup_m1.sh

# 2. Aktivuj prostředí
./start_research.sh

# 3. Test AI připojení
python m1_main.py --test-ai

# 4. Spusť research
python m1_main.py "machine learning applications"
```

## 🏗️ Architektura M1 Optimalizací

### Memory Management
- **Aktivní memory monitoring** s automatickým cleanup
- **Cache komprese** pro větší objekty
- **Emergency cleanup** při nízkém RAM
- **Garbage collection** po každé operaci

### AI Optimalizace (Phi-3 Mini)
- **Kratší context length** (2048 tokenů)
- **4 vlákna** pro M1 P-cores
- **Memory mapping** pro efektivitu
- **Auto-unload** po 3 minutách neaktivity
- **Streamování** pro rychlejší UX

### Síťové optimalizace
- **Omezené concurrent requesty** (2-3 max)
- **Kratší timeouty** (30s)
- **Agresivní caching** (1-2 hodiny TTL)
- **Connection pooling** optimalizovaný pro M1

## 🧠 Phi-3 Mini - Proč je ideální pro M1

### Výhody pro MacBook Air M1:
- **Velikost**: 3.8B parametrů = ~2.2GB RAM
- **Rychlost**: Optimalizován pro Apple Silicon
- **Kvalita**: Výborné výsledky i na malém modelu
- **Efektivita**: Nízká spotřeba energie

### Srovnání s alternativami:
```
Model          Velikost  RAM     Rychlost  Kvalita
Phi-3 Mini     3.8B     2.2GB   ⭐⭐⭐⭐⭐   ⭐⭐⭐⭐
Llama 3.1 8B   8B       4.5GB   ⭐⭐⭐     ⭐⭐⭐⭐⭐
Mistral 7B     7B       4.0GB   ⭐⭐⭐     ⭐⭐⭐⭐
```

## 🚀 Použití

### Základní příkazy
```bash
# Interaktivní mód
python m1_main.py

# Jeden dotaz
python m1_main.py "quantum computing"

# S parametry
python m1_main.py "AI ethics" --strategy fast --output results.json

# Bez AI analýzy (jen scraping)
python m1_main.py "research topic" --no-ai
```

### Strategie výzkumu
- **`fast`**: 2 zdroje, 30s timeout, základní AI
- **`balanced`**: 3 zdroje, 45s timeout, plná AI analýza
- **`thorough`**: Všechny zdroje, 60s timeout, detailní analýza

### Podporované zdroje
- **Wikipedia**: Rychlé obecné informace
- **PubMed**: Medicínské a vědecké články  
- **OpenAlex**: Akademické publikace

## ⚙️ Konfigurace M1

### Automatické optimalizace podle prostředí:

**Development** (výchozí):
```yaml
max_context_length: 2048
concurrent_sources: 2
memory_threshold: 1.5GB
cache_ttl: 30min
```

**Production**:
```yaml
max_context_length: 4096
concurrent_sources: 3  
memory_threshold: 1.0GB
cache_ttl: 2h
```

### Manuální nastavení (.env):
```bash
# Memory optimalizace
MEMORY_THRESHOLD_GB=1.5
MAX_CONCURRENT_REQUESTS=2
AUTO_CLEANUP=true

# AI model optimalizace
PRIMARY_MODEL=phi3:mini
MAX_CONTEXT_LENGTH=2048
STREAM_RESPONSES=true
LOW_MEMORY_MODE=true

# Privacy settings
OFFLINE_MODE=true
USE_EXTERNAL_APIs=false
LOG_QUERIES=false
```

## 📊 Monitoring a Diagnostika

### Systémové informace
```bash
python m1_main.py --system-info
```

### Performance test
```bash
python m1_performance_test.py
```

### Cache statistiky
```python
# V interaktivním módu
>>> stats
```

### Očekávané hodnoty pro M1:
- **AI odpověď**: 2-5 sekund
- **Scraping**: 10-20 sekund  
- **Memory usage**: 200-500MB
- **Cache hit rate**: 60-80%

## 🔧 Troubleshooting M1

### Pomalý AI
```bash
# Restartuj Ollama
brew services restart ollama

# Ověř model
ollama list

# Reinstaluj model
ollama rm phi3:mini
ollama pull phi3:mini
```

### Vysoká spotřeba paměti
```bash
# Vyčisti cache
python m1_main.py
>>> clear

# Restartuj aplikaci
# Zkontroluj ostatní aplikace
```

### Síťové problémy
```bash
# Zkontroluj internetové připojení
curl -I https://wikipedia.org

# Test lokálního AI
curl http://localhost:11434/api/tags
```

## 🎯 Best Practices pro M1

### Memory Management
1. **Zavři nepotřebné aplikace** před větším research
2. **Používej 'fast' strategii** pro běžné dotazy
3. **Vyčisti cache** občas (`clear` příkaz)
4. **Monitoruj memory usage** (--system-info)

### Performance  
1. **Krátké, specifické dotazy** jsou rychlejší
2. **Cache se automaticky optimalizuje** - opakuj dotazy
3. **Batch processing** - více dotazů najednou
4. **Používej offline mód** pro maximální rychlost

### Energy Efficiency
1. **Auto-unload modelu** po 3 minutách
2. **Přepínání mezi strategiemi** podle potřeby  
3. **Streamování odpovědí** pro rychlejší UX
4. **Komprese cache** šetří prostor

## 📁 Struktura Projektu (M1 Optimized)

```
PythonProject2/
├── m1_main.py              # 🎯 Hlavní vstupní bod pro M1
├── local_ai_adapter.py     # 🧠 Phi-3 Mini integrace
├── unified_config.py       # ⚙️  M1 optimalizovaná konfigurace
├── unified_research_engine.py # 🔍 Research engine pro M1
├── cache_manager.py        # 💾 Memory-efficient caching
├── academic_scraper.py     # 📚 Optimalizovaný scraping
├── setup_m1.sh            # 🛠️  Automatický M1 setup
├── start_research.sh       # ⚡ Quick start script
├── requirements.txt        # 📦 Minimální dependencies
├── .env                    # 🔐 M1 konfigurace
└── README_M1.md           # 📖 Tento soubor
```

## 🔐 Privacy & Security

### Lokální-first přístup:
- ✅ **Žádná data neopouštějí MacBook**
- ✅ **Phi-3 Mini běží offline**
- ✅ **Žádné API klíče třetích stran**
- ✅ **Žádné logování dotazů**
- ✅ **Cache pouze lokálně**

### Volitelné externí služby:
- ⚠️ **Scraping zdrojů** (Wikipedia, PubMed) - lze vypnout
- ⚠️ **Externí API** jsou defaultně zakázané

## 🆚 Srovnání s původním projektem

| Vlastnost | Původní | M1 Optimized |
|-----------|---------|--------------|
| AI Backend | OpenAI/Gemini | Phi-3 Mini (local) |
| Memory usage | ~1GB+ | ~200-500MB |
| Startup time | 10-20s | 2-5s |
| Privacy | Externí API | 100% lokální |
| Dependencies | 20+ balíčků | 10 balíčků |
| Setup | Složitý | `./setup_m1.sh` |

## 🚀 Roadmap & Optimalizace

### Krátký termín:
- [ ] **Vícejazyčné AI odpovědi**
- [ ] **PDF export výsledků**
- [ ] **Grafické uživatelské rozhraní**
- [ ] **Automatické aktualizace modelů**

### Dlouhý termín:
- [ ] **Vlastní fine-tuned model**
- [ ] **Lokální embeddings database**
- [ ] **M1 Neural Engine integrace**
- [ ] **iOS/iPadOS companion app**

## 💡 Tipy pro pokročilé uživatele

### Vlastní prompt templates:
```python
# V local_ai_adapter.py
custom_prompt = f"""
Kontext: {context}
Dotaz: {query}
Odpověz jako expert ve formátu:
1. Klíčové body
2. Praktické aplikace  
3. Doporučení

Odpověď:
"""
```

### Batch processing:
```python
queries = ["AI trends", "ML applications", "quantum computing"]
for query in queries:
    result = await tool.research_query(query, strategy="fast")
```

### Cache warming:
```python
# Předloaduj časté dotazy
common_queries = ["machine learning", "AI ethics", "quantum computing"]
for query in common_queries:
    await tool.research_query(query, strategy="fast")
```

---

**🎯 M1 MacBook Research Tool - Maximální výkon, minimální spotřeba, absolutní privacy**
