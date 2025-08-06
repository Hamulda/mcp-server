# Research Tool Configuration
import os
from pathlib import Path
import json

# Základní nastavení
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"

# Vytvořit složky pokud neexistují
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# API klíče (nastavit v .env souboru)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Fallback
SERP_API_KEY = os.getenv("SERP_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Nastavení pro web scraping
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Nastavení pro databáze
DATABASE_PATH = BASE_DIR / "research_database.db"

# Nastavení pro analýzu - optimalizováno pro Gemini
MAX_CONCURRENT_REQUESTS = 8  # Sníženo pro rate limiting
REQUEST_DELAY = 1.5  # Zvýšeno pro Gemini limits
GEMINI_RATE_LIMIT = 60  # requests per minute

# Token optimization nastavení
MAX_TOKENS_PER_REQUEST = 25000  # Gemini Pro limit
OPTIMAL_BATCH_SIZE = 5  # Optimální počet textů na batch
TOKEN_BUFFER = 2000  # Bezpečnostní buffer

# Gemini model konfigurace
GEMINI_MODEL_CONFIG = {
    'model': 'gemini-pro',
    'temperature': 0.1,  # Nízká pro konzistentní výsledky
    'top_p': 0.8,
    'top_k': 20,
    'max_output_tokens': 4000
}

# Cost optimization
ENABLE_COST_TRACKING = True
DAILY_COST_LIMIT = 5.0  # USD limit per day
COST_ALERT_THRESHOLD = 0.8  # Alert at 80% of limit

# Zdroje pro research
ACADEMIC_SOURCES = {
    'arxiv': 'https://arxiv.org/search/',
    'pubmed': 'https://pubmed.ncbi.nlm.nih.gov/',
    'google_scholar': 'https://scholar.google.com/',
    'semantic_scholar': 'https://www.semanticscholar.org/api/v1/',
}

WEB_SOURCES = {
    'news_apis': ['newsapi.org', 'gnews.io'],
    'rss_feeds': [
        'https://feeds.feedburner.com/oreilly/radar',
        'https://techcrunch.com/feed/',
        'https://www.wired.com/feed/rss',
    ]
}

# AI analýza typy optimalizované pro Gemini
ANALYSIS_TYPES = {
    'quick': {
        'max_tokens_per_text': 1000,
        'batch_size': 8,
        'description': 'Rychlá analýza - sentiment a klíčová témata'
    },
    'summary': {
        'max_tokens_per_text': 2000,
        'batch_size': 5,
        'description': 'Shrnutí a hlavní poznatky'
    },
    'comprehensive': {
        'max_tokens_per_text': 3000,
        'batch_size': 3,
        'description': 'Komplexní analýza se všemi metrikami'
    },
    'keywords_only': {
        'max_tokens_per_text': 500,
        'batch_size': 10,
        'description': 'Pouze extrakce klíčových slov'
    }
}

# Research depths s token optimalizací
RESEARCH_DEPTHS = {
    'shallow': {
        'max_sources': 20,
        'analysis_type': 'quick',
        'token_budget': 15000
    },
    'medium': {
        'max_sources': 50,
        'analysis_type': 'summary',
        'token_budget': 40000
    },
    'deep': {
        'max_sources': 100,
        'analysis_type': 'comprehensive',
        'token_budget': 80000
    }
}

# Načtení konfigurace MCP
CONFIG_PATH = os.path.expanduser('~/.config/github-copilot/intellij/mcp.json')

def load_mcp_config():
    try:
        with open(CONFIG_PATH, 'r') as config_file:
            config = json.load(config_file)
            return config
    except FileNotFoundError:
        print(f"Konfigurační soubor {CONFIG_PATH} nebyl nalezen.")
        return None
    except json.JSONDecodeError:
        print("Chyba při dekódování JSON souboru.")
        return None

mcp_config = load_mcp_config()
if mcp_config:
    print("MCP konfigurace načtena:", mcp_config)
