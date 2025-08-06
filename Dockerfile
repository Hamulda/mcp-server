# Použijeme oficiální Python 3.11 image
FROM python:3.11-slim

# Nastavení pracovního adresáře
WORKDIR /app

# Nastavení proměnných prostředí
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Instalace systémových závislostí
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Kopírování requirements.txt a instalace Python závislostí
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Kopírování zdrojového kódu
COPY . .

# Vytvoření potřebných adresářů
RUN mkdir -p data raw_data processed_data reports logs

# Nastavení oprávnění
RUN chmod +x main.py

# Expo portů
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Spuštění aplikace
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
