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
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_enabled: bool = True
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 100
    request_timeout: int = 300

class UnifiedConfig:
    """Sjednocená konfigurace s ENV support a validací"""
    
    def __init__(self, environment: Environment = None):
        self.environment = environment or self._detect_environment()
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        
        # Initialize configurations
        self.database = self._init_database_config()
        self.scraping = self._init_scraping_config()
        self.cache = self._init_cache_config()
        self.logging = self._init_logging_config()
        self.api = self._init_api_config()
        self.sources = self._init_sources_config()
        
        # Create directories
        self._ensure_directories()
    
    def _detect_environment(self) -> Environment:
        """Automatická detekce prostředí"""
        env_name = os.getenv('ENVIRONMENT', 'development').lower()
        if os.getenv('WEBSITE_SITE_NAME'):  # Azure App Service
            return Environment.PRODUCTION
        elif env_name == 'testing':
            return Environment.TESTING
        else:
            return Environment.DEVELOPMENT
    
    def _init_database_config(self) -> DatabaseConfig:
        """Inicializace databázové konfigurace z ENV"""
        if self.environment == Environment.PRODUCTION:
            # Azure Cosmos DB nebo jiná produkční DB
            return DatabaseConfig(
                type=os.getenv('DB_TYPE', 'cosmos'),
                url=os.getenv('DATABASE_URL'),
                connection_pool_size=int(os.getenv('DB_POOL_SIZE', '20')),
                timeout=int(os.getenv('DB_TIMEOUT', '60'))
            )
        else:
            # Lokální SQLite
            return DatabaseConfig(
                type='sqlite',
                url=str(self.data_dir / "research.db"),
                connection_pool_size=5,
                timeout=30
            )
    
    def _init_scraping_config(self) -> ScrapingConfig:
        """Inicializace scraping konfigurace z ENV"""
        return ScrapingConfig(
            request_timeout=int(os.getenv('SCRAPING_TIMEOUT', '30')),
            max_retries=int(os.getenv('SCRAPING_MAX_RETRIES', '3')),
            retry_delay=float(os.getenv('SCRAPING_RETRY_DELAY', '1.0')),
            concurrent_requests=int(os.getenv('SCRAPING_CONCURRENT', '5'))
        )
    
    def _init_cache_config(self) -> CacheConfig:
        """Inicializace cache konfigurace"""
        cache_dir = self.data_dir / "cache" if self.environment != Environment.PRODUCTION else None
        return CacheConfig(
            enabled=os.getenv('CACHE_ENABLED', 'true').lower() == 'true',
            ttl_seconds=int(os.getenv('CACHE_TTL', '3600')),
            max_size=int(os.getenv('CACHE_MAX_SIZE', '1000')),
            persist_to_disk=self.environment != Environment.PRODUCTION,
            cache_dir=cache_dir
        )
    
    def _init_logging_config(self) -> LoggingConfig:
        """Inicializace logging konfigurace"""
        level = "DEBUG" if self.environment == Environment.DEVELOPMENT else "INFO"
        file_path = self.data_dir / "app.log" if self.environment != Environment.PRODUCTION else None
        
        return LoggingConfig(
            level=os.getenv('LOG_LEVEL', level),
            file_path=file_path
        )
    
    def _init_api_config(self) -> APIConfig:
        """Inicializace API konfigurace"""
        return APIConfig(
            host=os.getenv('API_HOST', '0.0.0.0'),
            port=int(os.getenv('PORT', '8000')),  # Azure uses PORT env var
            debug=self.environment == Environment.DEVELOPMENT,
            cors_enabled=os.getenv('CORS_ENABLED', 'true').lower() == 'true',
            rate_limit_enabled=self.environment == Environment.PRODUCTION,
            rate_limit_per_minute=int(os.getenv('RATE_LIMIT_PER_MINUTE', '100'))
        )
    
    def _init_sources_config(self) -> Dict[str, SourceConfig]:
        """Inicializace konfigurace zdrojů"""
        return {
            'wikipedia': SourceConfig(
                name='Wikipedia',
                base_url='https://en.wikipedia.org',
                rate_limit_delay=0.5,
                parser_config={
                    'content_selector': '#mw-content-text',
                    'title_selector': 'h1.firstHeading',
                    'summary_selector': '.mw-parser-output > p:first-of-type'
                }
            ),
            'openalex': SourceConfig(
                name='OpenAlex',
                base_url='https://api.openalex.org',
                rate_limit_delay=0.1,
                custom_headers={'User-Agent': 'Research-Tool (mailto:research@example.com)'},
                parser_config={
                    'works_endpoint': '/works',
                    'authors_endpoint': '/authors'
                }
            ),
            'semantic_scholar': SourceConfig(
                name='Semantic Scholar',
                base_url='https://api.semanticscholar.org',
                api_key_env='SEMANTIC_SCHOLAR_API_KEY',
                rate_limit_delay=1.0,
                parser_config={
                    'search_endpoint': '/graph/v1/paper/search'
                }
            ),
            'pubmed': SourceConfig(
                name='PubMed',
                base_url='https://eutils.ncbi.nlm.nih.gov',
                rate_limit_delay=0.3,
                parser_config={
                    'search_endpoint': '/entrez/eutils/esearch.fcgi',
                    'fetch_endpoint': '/entrez/eutils/efetch.fcgi'
                }
            )
        }
    
    def _ensure_directories(self):
        """Vytvoří potřebné adresáře"""
        if self.environment != Environment.PRODUCTION:
            for dir_path in [self.data_dir, self.cache.cache_dir]:
                if dir_path:
                    dir_path.mkdir(exist_ok=True)
    
    def get_source_config(self, source_name: str) -> Optional[SourceConfig]:
        """Vrátí konfiguraci pro daný zdroj"""
        return self.sources.get(source_name)
    
    def is_source_enabled(self, source_name: str) -> bool:
        """Zkontroluje, zda je zdroj povolený"""
        config = self.get_source_config(source_name)
        return config.enabled if config else False
    
    def validate(self) -> List[str]:
        """Validuje konfiguraci a vrátí seznam chyb/varování"""
        errors = []
        
        # Validate database config
        if self.database.type == 'cosmos' and not self.database.url:
            errors.append("Cosmos DB URL is required for production")
        
        # Validate API keys for external services
        for source_name, source_config in self.sources.items():
            if source_config.api_key_env:
                api_key = os.getenv(source_config.api_key_env)
                if not api_key and source_config.enabled:
                    errors.append(f"Missing API key for {source_name}: {source_config.api_key_env}")
        
        # Validate directories in non-production
        if self.environment != Environment.PRODUCTION:
            if not self.data_dir.exists():
                try:
                    self.data_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create data directory {self.data_dir}: {e}")
        
        # Validate performance settings
        if self.scraping.concurrent_requests > 10:
            errors.append("Warning: High concurrent requests may cause rate limiting")
        
        if self.scraping.request_timeout < 5:
            errors.append("Warning: Very low request timeout may cause failures")
        
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
                _config_instance = UnifiedConfig()
    
    return _config_instance

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
    'CacheConfig', 'LoggingConfig', 'APIConfig',
    'UnifiedConfig', 'get_config', 'create_config', 'reset_config',
    'load_env_file', 'validate_config_on_startup'
]
