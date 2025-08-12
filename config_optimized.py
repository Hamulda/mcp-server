"""
Cost-Optimized Configuration - Minimální náklady, maximální efektivita
Zaměřeno na free API sources a ultra-low cost operations
"""

import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class CostOptimizedConfig:
    """Ultra-optimalizovaná konfigurace pro minimální náklady"""
    
    # Free sources only - no API keys needed
    FREE_SOURCES = {
        'wikipedia': {
            'enabled': True,
            'rate_limit': 0.5,  # 2 requests per second
            'max_concurrent': 3,
            'cost_per_request': 0.0  # FREE
        },
        'pubmed': {
            'enabled': True,
            'rate_limit': 0.3,  # NCBI guidelines
            'max_concurrent': 2,
            'cost_per_request': 0.0  # FREE
        }
    }
    
    # Cost optimization settings
    COST_OPTIMIZATION = {
        'max_requests_per_query': 10,  # Limit total requests
        'cache_ttl_hours': 24,         # Aggressive caching
        'prefer_free_sources': True,   # Always prefer free APIs
        'batch_processing': True,      # Batch requests for efficiency
        'compression_enabled': True    # Compress responses
    }
    
    # Performance vs cost balance
    PERFORMANCE_PROFILE = 'cost_optimized'  # vs 'balanced' or 'performance'
    
    # Monitoring for cost control
    COST_LIMITS = {
        'max_daily_requests': 1000,    # Stay under free tier limits
        'max_monthly_cost': 5.0,       # $5/month target vs $20 Perplexity
        'alert_threshold': 0.8         # Alert at 80% of limits
    }

# Global config instance
config = CostOptimizedConfig()

def get_optimized_config():
    """Get cost-optimized configuration"""
    return config
