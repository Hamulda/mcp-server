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
        self._setup_cache_dir()

    def _setup_default_sources(self):
        """Setup default source configurations"""
        default_sources = {
            'wikipedia': SourceConfig(
                name='wikipedia',
                base_url='https://en.wikipedia.org/api/rest_v1',
                rate_limit_delay=0.5,
                custom_headers={'User-Agent': 'Academic Research Tool 2.1.0'},
                enabled=True,
                max_concurrent=3
            ),
            'pubmed': SourceConfig(
                name='pubmed',
                base_url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils',
                rate_limit_delay=0.3,
                custom_headers={'User-Agent': 'Academic Research Tool 2.1.0'},
                enabled=True,
                max_concurrent=2
            ),
            'openalex': SourceConfig(
                name='openalex',
                base_url='https://api.openalex.org',
                rate_limit_delay=0.1,
                custom_headers={'User-Agent': 'Academic Research Tool 2.1.0'},
                enabled=True,
                max_concurrent=5
            )
        }

        # Merge with existing sources
        for name, config in default_sources.items():
            if name not in self.sources:
                self.sources[name] = config

    def _setup_cache_dir(self):
        """Setup cache directory"""
        if self.cache.cache_dir is None:
            project_root = Path(__file__).parent
            self.cache.cache_dir = project_root / "cache" / "unified"
            self.cache.cache_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> bool:
        """Validate configuration"""
        try:
            # Basic validation
            if not self.sources:
                return False

            # Check cache directory
            if self.cache.enabled and self.cache.cache_dir:
                self.cache.cache_dir.mkdir(parents=True, exist_ok=True)

            return True
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Export do dictionary pro serialization"""
        from dataclasses import asdict
        return asdict(self)

# Global config instance with thread safety
_config_instance: Optional[UnifiedConfig] = None
_config_lock = threading.Lock()

def get_config() -> UnifiedConfig:
    """Thread-safe singleton factory for unified configuration"""
    global _config_instance

    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = UnifiedConfig()

                # Load environment overrides
                if os.getenv('ENVIRONMENT'):
                    try:
                        _config_instance.environment = Environment(os.getenv('ENVIRONMENT'))
                    except ValueError:
                        pass

                # API configuration from environment
                if os.getenv('API_HOST'):
                    _config_instance.api.host = os.getenv('API_HOST')
                if os.getenv('API_PORT'):
                    try:
                        _config_instance.api.port = int(os.getenv('API_PORT'))
                    except ValueError:
                        pass

                # Validate configuration
                if not _config_instance.validate():
                    raise RuntimeError("Configuration validation failed")

    return _config_instance

def reset_config():
    """Reset global configuration instance (for testing)"""
    global _config_instance
    with _config_lock:
        _config_instance = None

# Convenience functions
def get_source_config(source_name: str) -> Optional[SourceConfig]:
    """Get configuration for specific source"""
    config = get_config()
    return config.sources.get(source_name)

def is_source_enabled(source_name: str) -> bool:
    """Check if source is enabled"""
    source_config = get_source_config(source_name)
    return source_config.enabled if source_config else False

# Export hlavních objektů
__all__ = [
    'UnifiedConfig', 'get_config', 'create_config', 'reset_config',
    'Environment', 'ResearchStrategy',
    'DatabaseConfig', 'ScrapingConfig', 'LocalAIConfig', 'AIConfig',
    'CacheConfig', 'CostConfig', 'LoggingConfig', 'APIConfig', 'SecurityConfig', 'SourceConfig'
]
