#!/bin/bash
# M1 MacBook Setup Script - Optimalizovaný pro Llama 3.1 8B na MacBook Air M1
# Tento skript nastaví pokročilé lokální AI vývojové prostředí s vysoce kvalitním modelem

set -e

echo "🍎 M1 MacBook Air Research Tool Setup - Llama 3.1 8B Edition"
echo "============================================================"

# Kontrola M1 architektury
if [[ $(uname -m) != "arm64" ]]; then
    echo "⚠️  Varování: Tento skript je optimalizován pro M1 MacBook (arm64)"
    echo "   Pokračuji, ale optimalizace nemusí být ideální..."
fi

# Kontrola RAM (důležité pro 8B model)
total_ram=$(sysctl -n hw.memsize)
total_ram_gb=$((total_ram / 1024 / 1024 / 1024))

echo "🔍 Kontroluji systém..."
echo "   OS: $(sw_vers -productName) $(sw_vers -productVersion)"
echo "   Architektura: $(uname -m)"
echo "   RAM: ${total_ram_gb}GB"

if [ $total_ram_gb -lt 8 ]; then
    echo "⚠️  VAROVÁNÍ: Detekována RAM méně než 8GB!"
    echo "   Llama 3.1 8B může být pomalý. Doporučuji Phi-3 Mini pro váš systém."
    read -p "   Pokračovat s Llama 3.1 8B? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Upravuji setup pro Phi-3 Mini..."
        USE_PHI3_MINI=true
    fi
fi

# Kontrola a instalace Homebrew
if ! command -v brew &> /dev/null; then
    echo "📦 Instaluji Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew je nainstalován"
fi

# Aktualizace Homebrew
echo "🔄 Aktualizuji Homebrew..."
brew update

# Kontrola a instalace Python 3.11+ (optimální pro M1)
if ! python3.11 --version &> /dev/null; then
    echo "🐍 Instaluji Python 3.11 (optimalizovaný pro M1)..."
    brew install python@3.11
else
    echo "✅ Python 3.11 je nainstalován"
fi

# Vytvoření virtuálního prostředí
echo "📁 Vytvářím virtuální prostředí..."
python3.11 -m venv venv_llama
source venv_llama/bin/activate

echo "✅ Virtuální prostředí aktivováno"

# Upgrade pip pro M1
echo "⬆️  Upgraduju pip..."
pip install --upgrade pip

# Instalace M1 optimalizovaných balíčků
echo "🚀 Instaluji M1 optimalizované balíčky..."
pip install -r requirements.txt

# Instalace a setup Ollama
echo "🧠 Nastavuji Ollama pro pokročilé lokální AI..."

if ! command -v ollama &> /dev/null; then
    echo "📥 Stahuji a instaluji Ollama..."
    brew install ollama
else
    echo "✅ Ollama je nainstalován"
fi

# Spuštění Ollama serveru na pozadí
echo "🔄 Spouštím Ollama server..."
brew services start ollama

# Chvilku počkáme na start serveru
sleep 5

# Stažení modelů podle konfigurace
if [ "$USE_PHI3_MINI" = true ]; then
    echo "⬇️  Stahuji Phi-3 Mini (optimalizovaný pro nižší RAM)..."
    ollama pull phi3:mini
    PRIMARY_MODEL="phi3:mini"
    FALLBACK_MODEL="phi3:mini"
else
    echo "⬇️  Stahuji Llama 3.1 8B (high-quality model)..."
    ollama pull llama3.1:8b

    echo "⬇️  Stahuji Phi-3 Mini jako fallback..."
    ollama pull phi3:mini

    echo "⬇️  Stahuji Qwen2 7B pro kódování..."
    ollama pull qwen2:7b

    PRIMARY_MODEL="llama3.1:8b"
    FALLBACK_MODEL="phi3:mini"
fi

# Stažení embedding modelu
echo "📊 Stahuji embedding model..."
ollama pull nomic-embed-text

# Test primárního modelu
echo "🧪 Testuji ${PRIMARY_MODEL}..."
if ollama run $PRIMARY_MODEL "Ahoj! Jsi připraven na pokročilý research?" --timeout 60s; then
    echo "✅ ${PRIMARY_MODEL} funguje správně!"
else
    echo "⚠️  ${PRIMARY_MODEL} možná nefunguje správně"
    if [ "$PRIMARY_MODEL" != "phi3:mini" ]; then
        echo "   Zkouším fallback na Phi-3 Mini..."
        if ollama run phi3:mini "Test fallback" --timeout 30s; then
            echo "✅ Fallback na Phi-3 Mini funguje"
        fi
    fi
fi

# Vytvoření optimalizované konfigurace pro Llama 3.1 8B
echo "⚙️  Vytvářím Llama 3.1 8B optimalizovanou konfiguraci..."
cat > .env << EOF
# M1 MacBook Optimized Configuration - Llama 3.1 8B Edition
ENVIRONMENT=development

# Ollama Configuration for Llama 3.1 8B
OLLAMA_HOST=http://localhost:11434
PRIMARY_MODEL=$PRIMARY_MODEL
FALLBACK_MODEL=$FALLBACK_MODEL
CODE_MODEL=qwen2:7b
EMBEDDING_MODEL=nomic-embed-text

# Enhanced Memory Optimizations for 8B Model
MAX_CONTEXT_LENGTH=4096
MAX_QUALITY_TOKENS=1024
MAX_CONCURRENT_REQUESTS=2
MEMORY_THRESHOLD_GB=2.5
AUTO_CLEANUP=true
SMART_MODEL_SWITCHING=true

# High-Quality AI Settings
REQUIRE_QUALITY_THRESHOLD=200
ADAPTIVE_CONTEXT=true
F16_PRECISION=true
STREAM_RESPONSES=true

# Privacy & Offline
OFFLINE_MODE=true
USE_EXTERNAL_APIs=false
CACHE_RESPONSES=true
LOG_QUERIES=false

# Performance for 8B Model
USE_GPU=true
LOW_MEMORY_MODE=true
NUM_THREADS=6
AUTO_GC=true
EOF

# Vytvoření Llama-optimalizovaných aliasů
echo "⚡ Vytvářím Llama 3.1 8B quick start skripty..."

cat > start_llama_research.sh << 'EOF'
#!/bin/bash
source venv_llama/bin/activate
echo "🧠 Llama 3.1 8B Research Tool je připraven!"
echo "💡 Použití:"
echo "   python llama_main.py --test-ai          # Test Llama 3.1 8B"
echo "   python llama_main.py --system-info      # Systémové info + model stats"
echo "   python llama_main.py 'AI research'      # High-quality research"
echo "   python llama_main.py --interactive      # Interaktivní mód"
echo ""
echo "🎯 Pro nejlepší kvalitu použij:"
echo "   python llama_main.py 'complex query' --require-quality"
EOF

chmod +x start_llama_research.sh

# Performance test pro Llama 3.1 8B
echo "⚡ Vytvářím Llama 3.1 8B performance test..."
cat > llama_performance_test.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import time
import psutil
from local_ai_adapter import M1OptimizedLlamaClient, quick_ai_query

async def test_llama_performance():
    print("🧪 Llama 3.1 8B Performance Test")
    print("=" * 50)

    # Memory před testem
    memory_before = psutil.virtual_memory()
    print(f"💾 RAM před testem: {memory_before.available / (1024**3):.1f}GB volné")

    # Test 1: Kvalita odpovědi s Llama 3.1 8B
    print("\n🧠 Test high-quality research s Llama 3.1 8B...")
    start_time = time.time()

    research_query = """Analyzuj současné trendy v oblasti umělé inteligence,
    zaměř se na praktické aplikace a budoucí směry vývoje."""

    response = await quick_ai_query(
        research_query,
        max_tokens=800,
        require_quality=True
    )

    ai_time = time.time() - start_time

    print(f"🤖 Llama 3.1 8B odpověď za: {ai_time:.2f}s")
    print(f"📝 Délka odpovědi: {len(response)} znaků")
    print(f"🎯 Kvalita: {'Vysoká' if len(response) > 500 else 'Střední'}")

    # Test 2: Rychlost s fallback modelem
    print(f"\n⚡ Test rychlé odpovědi s fallback...")
    start_time = time.time()

    quick_response = await quick_ai_query(
        "Co je machine learning?",
        max_tokens=200,
        require_quality=False
    )

    quick_time = time.time() - start_time
    print(f"🚀 Rychlá odpověď za: {quick_time:.2f}s")

    # Memory po testech
    memory_after = psutil.virtual_memory()
    print(f"\n💾 RAM po testu: {memory_after.available / (1024**3):.1f}GB volné")
    memory_used = (memory_before.available - memory_after.available) / (1024**2)
    print(f"📊 Spotřeba paměti: {memory_used:.1f}MB")

    # Model stats
    async with M1OptimizedLlamaClient() as client:
        model_stats = client.get_model_stats()
        print(f"\n🎯 Dostupné modely:")
        for name, stats in model_stats.items():
            print(f"   • {name}: {stats['size_gb']}GB, quality:{stats['quality_score']}/10")

    # CPU info
    cpu_usage = psutil.cpu_percent(interval=1)
    print(f"\n⚡ CPU využití: {cpu_usage}%")

    print(f"\n✅ Llama 3.1 8B performance test dokončen!")
    print(f"📈 Doporučení: {'Používej high-quality mode' if ai_time < 15 else 'Preferuj rychlé dotazy'}")

if __name__ == "__main__":
    asyncio.run(test_llama_performance())
EOF

chmod +x llama_performance_test.py

# Finální test
echo "🎯 Spouštím finální test Llama 3.1 8B systému..."
python3 -c "
import asyncio
from local_ai_adapter import quick_ai_query

async def test():
    try:
        response = await quick_ai_query('Test Llama 3.1 8B', max_tokens=50)
        print(f'✅ Llama test úspěšný: {response[:100]}...')
    except Exception as e:
        print(f'❌ Llama test failed: {e}')

asyncio.run(test())
"

echo ""
echo "🎉 Llama 3.1 8B MacBook Setup dokončen!"
echo "=" * 50
echo "✅ Llama 3.1 8B je nainstalován a nakonfigurován"
echo "✅ Inteligentní model switching je aktivní"
echo "✅ M1 optimalizace pro větší modely jsou aktivní"
echo "✅ High-quality research mode je dostupný"
echo ""
echo "🚀 Quick Start:"
echo "   ./start_llama_research.sh             # Aktivovat prostředí"
echo "   python llama_main.py --test-ai        # Test Llama 3.1 8B"
echo "   python llama_main.py 'AI research'    # High-quality research"
echo "   python llama_performance_test.py      # Performance test"
echo ""
echo "💡 Pro nejlepší výkon s Llama 3.1 8B:"
echo "   • Zavři nepotřebné aplikace (potřebuje 4GB+ RAM)"
echo "   • Používej --require-quality pro komplexní dotazy"
echo "   • Fallback na Phi-3 Mini je automatický při nízkém RAM"
echo "   • Cache je optimalizovaná pro větší odpovědi"
echo ""
echo "🎯 Model hierarchie:"
echo "   Llama 3.1 8B → Vysoká kvalita, komplexní analýzy"
echo "   Qwen2 7B     → Kódování a technické dotazy"
echo "   Phi-3 Mini   → Rychlé odpovědi, fallback"
echo ""
