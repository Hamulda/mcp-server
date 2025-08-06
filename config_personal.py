# === ÚSPORNÉ NASTAVENÍ PRO MINIMÁLNÍ NÁKLADY ===
DAILY_COST_LIMIT = 2.0  # Pouze 2 USD denně - úspora oproti Perplexity ($20/měsíc)
MONTHLY_TARGET_COST = 15.0  # Cílové náklady za měsíc
GEMINI_RATE_LIMIT = 30  # Pomalejší ale levnější
MAX_CONCURRENT_REQUESTS = 3  # Méně paralelních požadavků = nižší náklady

# === AGRESIVNÍ ÚSPORY ===
BATCH_SIZE = 8  # Menší batche = levnější
MAX_TOKEN_LIMIT = 2000  # Nižší limit tokenů
PREFER_CACHE_OVER_API = True  # Preferuj cache před API voláními
MIN_CACHE_HIT_RATE = 0.7  # Minimálně 70% cache hit rate

# === OPTIMALIZACE PRO ÚSPORY ===
CACHE_EXPIRY_HOURS = 72  # Delší cache = méně API volání
ENABLE_AGGRESSIVE_DEDUPLICATION = True  # Odstranění duplikátů
SMART_QUERY_REDUCTION = True  # Inteligentní zkracování dotazů

# === TIMEOUT NASTAVENÍ PRO ÚSPORY ===
GOOGLE_SCHOLAR_TIMEOUT = 8  # Delší timeout pro méně opakování
PUBMED_TIMEOUT = 8
MAX_RETRY_ATTEMPTS = 1  # Pouze 1 pokus = nižší náklady

# === VÝSLEDKY - MÉNĚ JE VÍCE ===
MAX_RESULTS_PER_SOURCE = 30  # Méně výsledků = levnější
DEFAULT_MAX_RESULTS = 15  # Standardně jen 15 výsledků
QUALITY_OVER_QUANTITY = True  # Kvalita před kvantitou

# === RESEARCH DEPTHS - ÚSPORNÉ ===
RESEARCH_DEPTHS = {
    'shallow': {
        'max_sources': 10,
        'token_budget': 1000,
        'analysis_type': 'quick'
    },
    'medium': {
        'max_sources': 20,
        'token_budget': 2000,
        'analysis_type': 'summary'
    },
    'deep': {
        'max_sources': 30,  # Sníženo z 100
        'token_budget': 3000,  # Sníženo z 8000
        'analysis_type': 'comprehensive'
    }
}
