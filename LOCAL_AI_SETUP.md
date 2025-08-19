# 🏠 Private Research Tool - Local AI Optimization

## 🎯 Optimalizace pro soukromé použití

### Lokální AI Model Setup
- **Platform**: Ollama (nejlepší pro macOS)
- **Doporučené modely**:
  - `llama3.1:8b` - Nejlepší kvalita (5GB RAM)
  - `phi3:14b` - Rychlá inference (7GB RAM) 
  - `codellama:7b` - Technické analýzy (4GB RAM)

### Instalace Ollama:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended models
ollama pull llama3.1:8b
ollama pull phi3:14b
ollama pull codellama:7b
```

### MacBook Air M1 Optimalizace:
- **Memory Management**: Smart batching pro omezený RAM
- **Energy Efficiency**: Optimalizované inference scheduling
- **Storage**: Lokální cache pro modely a embedding
- **Network**: Offline-first s optional online fallback

## 🚀 Implementované optimalizace:
1. **Local AI Integration** - Ollama API wrapper
2. **Memory Optimization** - Dynamic model switching
3. **Offline Operation** - Complete local functionality
4. **Privacy First** - No external API calls by default
5. **Energy Efficient** - M1 optimized processing
