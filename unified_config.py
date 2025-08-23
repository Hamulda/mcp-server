#!/usr/bin/env python3
"""
Unified Configuration System - Centralizovaná konfigurace pro celý projekt
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

class UnifiedConfig:
    """Centralizovaný konfigurační systém"""

    def __init__(self):
        self.config_path = Path(__file__).parent / "config.json"
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Načte konfiguraci ze souboru nebo vytvoří výchozí"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Chyba při načítání konfigurace: {e}")

        # Výchozí konfigurace
        default_config = {
            "database": {
                "cache_db_path": "cache/smart_mcp_cache.db",
                "chroma_db_path": "chroma_data",
                "max_connections": 20
            },
            "api": {
                "rate_limit": 60,
                "timeout": 30,
                "max_retries": 3
            },
            "scraping": {
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "max_concurrent": 5,
                "delay_between_requests": 1.0
            },
            "ai": {
                "model": "llama3.1:8b",
                "temperature": 0.7,
                "max_tokens": 2048
            },
            "monitoring": {
                "enable_prometheus": True,
                "prometheus_port": 8000,
                "log_level": "INFO"
            },
            "security": {
                "enable_ssl_verification": True,
                "max_input_length": 10000,
                "allowed_domains": ["wikipedia.org", "pubmed.ncbi.nlm.nih.gov"]
            }
        }

        self._save_config(default_config)
        return default_config

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Uloží konfiguraci do souboru"""
        try:
            os.makedirs(self.config_path.parent, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Chyba při ukládání konfigurace: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Získá hodnotu z konfigurace"""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Nastaví hodnotu v konfiguraci"""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        self._save_config(self._config)

    def update(self, updates: Dict[str, Any]) -> None:
        """Aktualizuje více hodnot najednou"""
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        deep_update(self._config, updates)
        self._save_config(self._config)

# Globální instance
_config_instance = None

def get_config() -> UnifiedConfig:
    """Získá globální instanci konfigurace"""
    global _config_instance
    if _config_instance is None:
        _config_instance = UnifiedConfig()
    return _config_instance
