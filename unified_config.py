"""
Unified Configuration System - Sjednocuje všechny konfigurace
Nahrazuje config.py, settings.py, config_personal.py
Integruje stávající nastavení do unified architektury
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import threading

class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class ResearchStrategy(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"

@dataclass
class DatabaseConfig:
    """Databázová konfigurace"""
    type: str = "sqlite"  # sqlite, cosmos, postgresql
    url: Optional[str] = None
    connection_pool_size: int = 10
    timeout: int = 30

@dataclass
class ScrapingConfig:
    """Konfigurace pro scraping - integruje stávající nastavení"""
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    concurrent_requests: int = 5
    user_agents: List[str] = field(default_factory=lambda: [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ])

    # Rate limits per source - z config.py
    rate_limits: Dict[str, float] = field(default_factory=lambda: {
        'wikipedia': 0.5,
        'pubmed': 0.3,
        'openalex': 0.1,
        'semantic_scholar': 1.0,
        'google_scholar': 2.0
    })

@dataclass
class SourceConfig:
    """Konfigurace pro jednotlivé zdroje s validací"""
    name: str
    base_url: str
    api_key_env: Optional[str] = None
    rate_limit_delay: float = 1.0
    custom_headers: Dict[str, str] = field(default_factory=dict)
    parser_config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    max_concurrent: int = 1

@dataclass
class LocalAIConfig:
    """Konfigurace pro lokální AI modely optimalizované pro MacBook Air M1 - ultra-efficient pro peptide research"""
    # Ollama konfigurace s inteligentním model managementem
    ollama_host: str = "http://localhost:11434"
    primary_model: str = "llama3.1:8b"       # Pro komplexní peptide research (4GB RAM)
    fallback_model: str = "phi3:mini"        # Ultra-rychlý pro jednoduché dotazy (2.2GB RAM)
    code_model: str = "qwen2:7b"            # Pro analýzu dat a kódu (3.8GB RAM)
    embedding_model: str = "nomic-embed-text"  # Lokální embeddings (0.3GB RAM)

    # Nové ultra-light modely pro speed
    speed_model: str = "tinyllama:1.1b"     # Ultra-rychlý pro preview (0.6GB RAM)
    summary_model: str = "phi3:mini"        # Pro rychlé shrnutí

    # Inteligentní memory management
    max_context_length: int = 8192          # Větší kontext pro peptide research
    batch_size: int = 1                     # Jeden dotaz = nižší RAM usage
    temperature: float = 0.2                # Nižší pro vědeckou přesnost
    top_p: float = 0.85                     # Optimalizováno pro faktické odpovědi

    # Ultra energy efficiency
    auto_unload_timeout: int = 180          # 3 minuty - rychlejší unload
    use_mmap: bool = True                   # Memory mapping pro úsporu
    num_threads: int = 8                    # Více vláken pro M1
    num_predict: int = 1024                 # Delší odpovědi pro detailní research

    # Adaptivní model switching
    smart_model_switching: bool = True      # Auto výběr modelu podle dotazu
    query_complexity_threshold: int = 50    # Jednoduchý dotaz = rychlý model
    use_speed_model_for_preview: bool = True # TinyLlama pro rychlé preview

    # M1 specifické optimalizace
    use_gpu: bool = True                    # M1 Neural Engine
    metal_performance: bool = True          # Metal Performance Shaders
    low_memory_mode: bool = True            # Agresivní RAM optimalizace
    stream_responses: bool = True           # Stream pro lepší UX
    f16_precision: bool = True              # Half precision

    # Cache optimalizace pro research
    cache_responses: bool = True            # Agresivní caching
    cache_embeddings: bool = True           # Cache embeddings
    persistent_cache: bool = True           # Přetrvává přes restarty
    max_cache_size_mb: int = 512           # 512MB cache limit

    # Research-specific optimizations
    peptide_focused_prompts: bool = True    # Specializované prompty pro peptidy
    dosage_extraction: bool = True          # Auto extrakce dávkování
    side_effects_analysis: bool = True      # Analýza vedlejších účinků
    interaction_checking: bool = True       # Kontrola interakcí

    # Performance monitoring
    monitor_performance: bool = True        # Monitor CPU/RAM usage
    auto_model_downgrade: bool = True       # Auto downgrade při vysokém RAM usage
    performance_threshold_ram: float = 0.8  # 80% RAM = switch to smaller model

@dataclass
class AIConfig:
    """AI služby konfigurace s lokálním podporou"""
    # Lokální AI má prioritu
    use_local_ai: bool = True
    local_ai: LocalAIConfig = field(default_factory=LocalAIConfig)

    # Externí služby jako fallback (pouze pokud explicitně povoleno)
    primary_provider: str = "local"  # Změněno z "gemini"
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7
    rate_limit: int = 30

    # Privacy settings
    allow_external_apis: bool = False  # Defaultně zakázané
    log_queries: bool = False         # Žádné logování dotazů

@dataclass
class CacheConfig:
    """Cache konfigurace"""
    enabled: bool = True
    ttl_seconds: int = 3600
    max_size: int = 1000
    persist_to_disk: bool = True
    cache_dir: Optional[Path] = None

@dataclass
class CostConfig:
    """Cost tracking configuration"""
    enabled: bool = True
    daily_limit: float = 2.0
    monthly_target: float = 50.0
    token_price_per_1k: float = 0.00025

@dataclass
class LoggingConfig:
    """Logging konfigurace"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[Path] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

@dataclass
class APIConfig:
    """API server konfigurace"""
    host: str = "localhost"
    port: int = 8000
    debug: bool = False
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_requests: int = 60
    rate_limit_window: int = 60

@dataclass
class SecurityConfig:
    """Bezpečnostní konfigurace"""
    api_key_required: bool = False
    api_key: Optional[str] = None
    allowed_hosts: List[str] = field(default_factory=lambda: ["*"])
    max_request_size: int = 10 * 1024 * 1024  # 10MB

@dataclass
class UnifiedConfig:
    """Hlavní unified konfigurace"""
    environment: Environment = Environment.DEVELOPMENT
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    sources: Dict[str, SourceConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Post-init setup"""
        self._setup_default_sources()
        self._setup_cache_directory()
        self._setup_logging()

    def _setup_default_sources(self):
        """Nastavení defaultních zdrojů s optimalizací pro peptidy a biohacking"""
        default_sources = {
            # Core academic sources
            'pubmed': SourceConfig(
                name='pubmed',
                base_url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/',
                rate_limit_delay=0.3,
                enabled=True,
                parser_config={'search_fields': ['title', 'abstract', 'mesh_terms']}
            ),
            'openalex': SourceConfig(
                name='openalex',
                base_url='https://api.openalex.org/',
                rate_limit_delay=0.1,
                enabled=True
            ),
            'semantic_scholar': SourceConfig(
                name='semantic_scholar',
                base_url='https://api.semanticscholar.org/',
                rate_limit_delay=1.0,
                enabled=True
            ),

            # Peptide & Biohacking specific sources
            'peptide_guide': SourceConfig(
                name='peptide_guide',
                base_url='https://peptideguide.com/',
                rate_limit_delay=2.0,
                enabled=True,
                parser_config={'focus': 'peptides', 'extract_dosages': True}
            ),
            'examine_com': SourceConfig(
                name='examine_com',
                base_url='https://examine.com/',
                rate_limit_delay=1.5,
                enabled=True,
                parser_config={'focus': 'supplements', 'evidence_based': True}
            ),
            'selfhacked': SourceConfig(
                name='selfhacked',
                base_url='https://selfhacked.com/',
                rate_limit_delay=2.0,
                enabled=True,
                parser_config={'focus': 'biohacking', 'health_optimization': True}
            ),
            'ben_greenfield': SourceConfig(
                name='ben_greenfield',
                base_url='https://bengreenfieldbody.com/',
                rate_limit_delay=3.0,
                enabled=True,
                parser_config={'focus': 'performance', 'biohacking': True}
            ),

            # Medical & Research databases
            'clinicaltrials': SourceConfig(
                name='clinicaltrials',
                base_url='https://clinicaltrials.gov/api/',
                rate_limit_delay=1.0,
                enabled=True,
                parser_config={'status': 'active', 'peptide_focus': True}
            ),
            'cochrane': SourceConfig(
                name='cochrane',
                base_url='https://www.cochranelibrary.com/api/',
                rate_limit_delay=2.0,
                enabled=True,
                parser_config={'systematic_reviews': True}
            ),

            # Supplement & Nootropic databases
            'nootropics_expert': SourceConfig(
                name='nootropics_expert',
                base_url='https://nootropicsexpert.com/',
                rate_limit_delay=2.0,
                enabled=True,
                parser_config={'cognitive_enhancement': True}
            ),
            'longecity': SourceConfig(
                name='longecity',
                base_url='https://www.longecity.org/',
                rate_limit_delay=3.0,
                enabled=True,
                parser_config={'longevity': True, 'community_research': True}
            ),

            # Core scientific sources
            'arxiv': SourceConfig(
                name='arxiv',
                base_url='http://export.arxiv.org/api/',
                rate_limit_delay=3.0,
                enabled=True
            ),
            'crossref': SourceConfig(
                name='crossref',
                base_url='https://api.crossref.org/',
                rate_limit_delay=1.0,
                enabled=True
            ),
            'wikipedia': SourceConfig(
                name='wikipedia',
                base_url='https://en.wikipedia.org/api/rest_v1/',
                rate_limit_delay=0.5,
                enabled=True
            ),

            # Specialized health databases
            'health_line': SourceConfig(
                name='health_line',
                base_url='https://www.healthline.com/',
                rate_limit_delay=2.0,
                enabled=True,
                parser_config={'medical_review': True}
            ),
            'mayo_clinic': SourceConfig(
                name='mayo_clinic',
                base_url='https://www.mayoclinic.org/',
                rate_limit_delay=2.0,
                enabled=True,
                parser_config={'authoritative_medical': True}
            )
        }

        # Merge s existujícími zdroji
        for name, config in default_sources.items():
            if name not in self.sources:
                self.sources[name] = config

    def _setup_cache_directory(self):
        """Nastavení cache adresáře"""
        if self.cache.cache_dir is None:
            self.cache.cache_dir = Path.cwd() / "cache"
        self.cache.cache_dir.mkdir(exist_ok=True)

    def _setup_logging(self):
        """Nastavení loggingu"""
        import logging

        level = getattr(logging, self.logging.level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format=self.logging.format
        )

    def get_source_config(self, source_name: str) -> Optional[SourceConfig]:
        """Získá konfiguraci pro zdroj"""
        return self.sources.get(source_name)

    def validate(self) -> List[str]:
        """Validace konfigurace, vrací seznam chyb"""
        errors = []

        # Validace portů
        if not (1 <= self.api.port <= 65535):
            errors.append(f"Invalid API port: {self.api.port}")

        # Validace cache
        if self.cache.ttl_seconds <= 0:
            errors.append("Cache TTL must be positive")

        # Validace AI konfigurace
        if self.ai.use_local_ai and not self.ai.local_ai.ollama_host:
            errors.append("Local AI enabled but no Ollama host specified")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Export do dictionary pro serialization"""
        from dataclasses import asdict
        return asdict(self)

# Global config instance s thread safety
_config_lock = threading.Lock()
_global_config: Optional[UnifiedConfig] = None

def create_config(environment: Environment = Environment.DEVELOPMENT) -> UnifiedConfig:
    """Vytvoří novou konfiguraci"""
    config = UnifiedConfig(environment=environment)

    # Load from environment variables
    _load_from_env(config)

    return config

def get_config() -> UnifiedConfig:
    """Získá globální konfiguraci (thread-safe singleton)"""
    global _global_config

    if _global_config is None:
        with _config_lock:
            if _global_config is None:
                _global_config = create_config()

    return _global_config

def _load_from_env(config: UnifiedConfig):
    """Načte konfiguraci z environment variables"""
    import os

    # Database konfigurace
    if os.getenv('DATABASE_URL'):
        config.database.url = os.getenv('DATABASE_URL')

    # API konfigurace
    if os.getenv('API_PORT'):
        try:
            config.api.port = int(os.getenv('API_PORT'))
        except ValueError:
            pass

    if os.getenv('API_HOST'):
        config.api.host = os.getenv('API_HOST')

    # AI konfigurace
    if os.getenv('OLLAMA_HOST'):
        config.ai.local_ai.ollama_host = os.getenv('OLLAMA_HOST')

    if os.getenv('GEMINI_API_KEY'):
        config.ai.gemini_api_key = os.getenv('GEMINI_API_KEY')

    if os.getenv('OPENAI_API_KEY'):
        config.ai.openai_api_key = os.getenv('OPENAI_API_KEY')

    # Environment
    if os.getenv('ENVIRONMENT'):
        env_value = os.getenv('ENVIRONMENT').lower()
        if env_value in ['development', 'testing', 'production']:
            config.environment = Environment(env_value)

    # Logging
    if os.getenv('LOG_LEVEL'):
        config.logging.level = os.getenv('LOG_LEVEL').upper()

    # Cache konfigurace
    if os.getenv('CACHE_DIR'):
        config.cache.cache_dir = Path(os.getenv('CACHE_DIR'))

    # Security
    if os.getenv('API_KEY'):
        config.security.api_key = os.getenv('API_KEY')
        config.security.api_key_required = True

def reset_config():
    """Reset globální konfigurace (pro testing)"""
    global _global_config
    with _config_lock:
        _global_config = None

# Export hlavních objektů
__all__ = [
    'UnifiedConfig', 'get_config', 'create_config', 'reset_config',
    'Environment', 'ResearchStrategy',
    'DatabaseConfig', 'ScrapingConfig', 'LocalAIConfig', 'AIConfig',
    'CacheConfig', 'CostConfig', 'LoggingConfig', 'APIConfig', 'SecurityConfig', 'SourceConfig'
]
