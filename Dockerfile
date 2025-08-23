# Multi-stage build pro optimální velikost a bezpečnost
FROM python:3.11-slim as builder

# Nastavení pracovního adresáře
WORKDIR /app

# Kopírování requirements pro cache optimalizaci
COPY requirements.txt .

# Instalace závislostí
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Vytvoření non-root uživatele pro bezpečnost
RUN useradd --create-home --shell /bin/bash app

# Nastavení pracovního adresáře
WORKDIR /home/app

# Kopírování závislostí z builder stage
COPY --from=builder /root/.local /home/app/.local

# Přidání local bin do PATH
ENV PATH=/home/app/.local/bin:$PATH

# Instalace system dependencies pro Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Kopírování zdrojového kódu
COPY --chown=app:app . .

# Playwright setup pro headless browser
RUN python -m playwright install chromium
RUN python -m playwright install-deps chromium

# Vytvoření potřebných adresářů
RUN mkdir -p cache chroma_data data/cache logs \
    && chown -R app:app cache chroma_data data logs

# Přepnutí na non-root user
USER app

# Nastavení environment variables
ENV PYTHONPATH=/home/app
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import asyncio; from core.main import UnifiedBiohackingResearchTool; print('OK')" || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "core/main.py", "--query", "health-check", "--type", "quick"]
