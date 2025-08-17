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
class AIConfig:
    """AI služby konfigurace"""
    primary_provider: str = "gemini"
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    max_tokens: int = 2000  # Z config_personal.py
    temperature: float = 0.7
    rate_limit: int = 30  # Z config_personal.py

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
    """API server konfigurace s optimalizacemi"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_enabled: bool = True
    worker_count: int = 1
    max_requests_per_worker: int = 1000
    keep_alive: int = 2
    timeout: int = 30
    # Nové optimalizace
    gzip_compression: bool = True
    rate_limit_enabled: bool = True
    prometheus_enabled: bool = True
    request_id_header: str = "X-Request-ID"

@dataclass
class PerformanceConfig:
    """Performance tuning konfigurace"""
    # Async optimalizace
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    connection_pool_size: int = 20
    max_keepalive_connections: int = 100

    # Memory optimalizace
    memory_limit_mb: int = 512
    gc_threshold: int = 1000

    # Disk I/O optimalizace
    use_async_io: bool = True
    buffer_size: int = 8192

    # Network optimalizace
    tcp_nodelay: bool = True
    socket_keepalive: bool = True
    compression_enabled: bool = True

@dataclass
class UnifiedConfig:
    """Sjednocená konfigurace s pokročilými optimalizacemi"""
    environment: Environment
    database: DatabaseConfig
    scraping: ScrapingConfig
    sources: Dict[str, SourceConfig]
    ai: AIConfig
    cache: CacheConfig
    cost: CostConfig
    logging: LoggingConfig
    api: APIConfig
    performance: PerformanceConfig

    def __post_init__(self):
        """Post-initialization optimalizace pro různá prostředí"""
        if self.environment == Environment.PRODUCTION:
            # Produkční optimalizace
            self.api.debug = False
            self.api.worker_count = max(2, os.cpu_count())
            self.performance.max_concurrent_requests = 200
            self.cache.ttl_seconds = 7200  # 2 hodiny
            self.cache.max_size = 5000
            self.logging.level = "INFO"

        elif self.environment == Environment.DEVELOPMENT:
            # Development optimalizace
            self.api.debug = True
            self.api.worker_count = 1
            self.performance.max_concurrent_requests = 50
            self.cache.ttl_seconds = 600  # 10 minut
            self.cache.max_size = 100
            self.logging.level = "DEBUG"

        elif self.environment == Environment.TESTING:
            # Testing optimalizace
            self.api.debug = False
            self.api.worker_count = 1
            self.performance.max_concurrent_requests = 10
            self.cache.enabled = False  # Disable cache in tests
            self.logging.level = "WARNING"

    def get_source_config(self, source_name: str) -> Optional[SourceConfig]:
        """Získá konfiguraci pro konkrétní zdroj"""
        return self.sources.get(source_name)

    def get_enabled_sources(self) -> List[str]:
        """Vrátí seznam povolených zdrojů"""
        return [name for name, config in self.sources.items() if config.enabled]

    def get_sources_by_priority(self) -> List[str]:
        """Vrátí zdroje seřazené podle rychlosti/priority"""
        # Priorita podle rychlosti: Wikipedia > OpenAlex > PubMed
        priority_order = ['wikipedia', 'openalex', 'pubmed']
        enabled_sources = self.get_enabled_sources()

        # Seřaď podle priority, pak zbytek alfabeticky
        sorted_sources = []
        for source in priority_order:
            if source in enabled_sources:
                sorted_sources.append(source)
                enabled_sources.remove(source)

        sorted_sources.extend(sorted(enabled_sources))
        return sorted_sources

    def validate(self) -> List[str]:
        """Validuje konfiguraci a vrátí seznam chyb/varování"""
        errors = []
        
        # Validace API konfigurace
        if self.api.port < 1 or self.api.port > 65535:
            errors.append(f"Invalid API port: {self.api.port}")

        # Validace cache konfigurace
        if self.cache.max_size < 1:
            errors.append("Cache max_size must be positive")

        # Validace zdrojů
        if not self.sources:
            errors.append("No sources configured")

        enabled_sources = self.get_enabled_sources()
        if not enabled_sources:
            errors.append("No sources enabled")

        # Validace performance nastavení
        if self.performance.max_concurrent_requests < 1:
            errors.append("max_concurrent_requests must be positive")

        # Varování pro produkci
        if self.environment == Environment.PRODUCTION:
            if self.api.debug:
                errors.append("DEBUG mode should be disabled in production")
            if not self.api.rate_limit_enabled:
                errors.append("Rate limiting should be enabled in production")

        return errors
    
    def get_database_url(self) -> str:
        """Vrátí database URL s fallback na SQLite"""
        if self.database.url:
            return self.database.url
        
        # Fallback na SQLite
        db_path = self.data_dir / "research.db"
        return f"sqlite:///{db_path}"
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Bezpečně vrátí API klíč pro danou službu"""
        source_config = self.get_source_config(service)
        if source_config and source_config.api_key_env:
            return os.getenv(source_config.api_key_env)
        return None
    
    def get_effective_rate_limit(self, source: str) -> float:
        """Vrátí efektivní rate limit pro zdroj"""
        source_config = self.get_source_config(source)
        if source_config:
            return source_config.rate_limit_delay
        return self.scraping.retry_delay  # Default fallback
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializuje konfiguraci do dictionary (bez citlivých dat)"""
        return {
            'environment': self.environment.value,
            'database': {
                'type': self.database.type,
                'connection_pool_size': self.database.connection_pool_size,
                'timeout': self.database.timeout
            },
            'scraping': {
                'request_timeout': self.scraping.request_timeout,
                'max_retries': self.scraping.max_retries,
                'retry_delay': self.scraping.retry_delay,
                'concurrent_requests': self.scraping.concurrent_requests
            },
            'cache': {
                'enabled': self.cache.enabled,
                'ttl_seconds': self.cache.ttl_seconds,
                'max_size': self.cache.max_size
            },
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'debug': self.api.debug
            },
            'sources': {
                name: {
                    'name': config.name,
                    'enabled': config.enabled,
                    'rate_limit_delay': config.rate_limit_delay
                }
                for name, config in self.sources.items()
            }
        }

# Global configuration instance
_config_instance: Optional[UnifiedConfig] = None
_config_lock = threading.Lock()

def get_config() -> UnifiedConfig:
    """Thread-safe singleton pro získání konfigurace"""
    global _config_instance
    
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = _create_default_config()

    return _config_instance

def _create_default_config() -> UnifiedConfig:
    """Vytvoří výchozí konfiguraci s rozumnými defaulty"""

    # Detect environment from ENV variable
    env_name = os.getenv('ENVIRONMENT', 'development').lower()
    try:
        environment = Environment(env_name)
    except ValueError:
        environment = Environment.DEVELOPMENT

    # Default sources configuration
    default_sources = {
        'wikipedia': SourceConfig(
            name='wikipedia',
            base_url='https://en.wikipedia.org/w/api.php',
            rate_limit_delay=0.5,
            enabled=True
        ),
        'pubmed': SourceConfig(
            name='pubmed',
            base_url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/',
            rate_limit_delay=0.3,
            enabled=True
        ),
        'openalex': SourceConfig(
            name='openalex',
            base_url='https://api.openalex.org/',
            rate_limit_delay=0.1,
            enabled=True
        )
    }

    return UnifiedConfig(
        environment=environment,
        database=DatabaseConfig(),
        scraping=ScrapingConfig(),
        sources=default_sources,
        ai=AIConfig(),
        cache=CacheConfig(),
        cost=CostConfig(),
        logging=LoggingConfig(),
        api=APIConfig(),
        performance=PerformanceConfig()
    )

def create_config(environment: Environment = None) -> UnifiedConfig:
    """Factory function pro vytvoření nové konfigurace"""
    return UnifiedConfig(environment)

def reset_config():
    """Reset globální konfigurace (především pro testování)"""
    global _config_instance
    with _config_lock:
        _config_instance = None

# Environment variable helper functions
def load_env_file(env_file: str = ".env"):
    """Načte .env soubor pokud existuje"""
    env_path = Path(env_file)
    if env_path.exists():
        try:
            # Fallback implementace pro načtení .env souboru
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Pouze nastavit pokud už není v ENV
                        if key not in os.environ:
                            os.environ[key] = value.strip('"\'')
        except Exception as e:
            print(f"Warning: Could not load {env_file}: {e}")

# Initialize environment variables on import
load_env_file()

# Configuration validation helper
def validate_config_on_startup():
    """Validuje konfiguraci při startu aplikace"""
    try:
        config = get_config()
        errors = config.validate()
        
        if errors:
            print("⚠️  Configuration validation warnings:")
            for error in errors:
                print(f"   • {error}")
        else:
            print("✅ Configuration validation passed")
            
        return len(errors) == 0
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

# Export all necessary components
__all__ = [
    'Environment',
    'DatabaseConfig', 'ScrapingConfig', 'SourceConfig', 'AIConfig', 
    'CacheConfig', 'CostConfig', 'LoggingConfig', 'APIConfig',
    'UnifiedConfig', 'get_config', 'create_config', 'reset_config',
    'load_env_file', 'validate_config_on_startup'
]
