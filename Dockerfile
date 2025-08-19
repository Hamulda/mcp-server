# Multi-stage build pro optimalizaci velikosti
FROM python:3.11-slim as builder

# Nastavení pracovního adresáře
WORKDIR /app

# Instalace systémových závislostí
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Kopírování requirements souborů
COPY requirements.txt ./

# Instalace Python závislostí
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Nastavení pracovního adresáře
WORKDIR /app

# Kopírování nainstalovaných balíčků z builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Instalace curl pro health check
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Kopírování aplikačního kódu
COPY . .

# Vytvoření cache a data adresářů
RUN mkdir -p cache/unified data

# Nastavení environmentálních proměnných
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "unified_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
