Local AI Adapter - Ultra-optimized for MacBook Air M1 with peptide research focus
Intelligent model switching, memory management, and performance monitoring
"""

import asyncio
import logging
import time
import psutil
import hashlib
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
import json
import aiohttp
from pathlib import Path

# Import specialized prompts
try:
    from peptide_prompts import get_prompt, detect_query_type
    PEPTIDE_PROMPTS_AVAILABLE = True
except ImportError:
    PEPTIDE_PROMPTS_AVAILABLE = False

try:
    from unified_config import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceStats:
    """Performance statistics for model monitoring"""
    model_name: str
    avg_tokens_per_second: float
    avg_memory_usage_mb: float
    total_queries: int
    error_rate: float
    last_used: float

class IntelligentModelManager:
    """Ultra-intelligent model manager for M1 optimization"""

    def __init__(self):
        self.config = get_config() if CONFIG_AVAILABLE else None
        self.performance_stats: Dict[str, ModelPerformanceStats] = {}
        self.current_model: Optional[str] = None
        self.model_loaded_time: float = 0
        self.cache_dir = Path("cache/ai_responses")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def get_optimal_model(self, query: str, priority: str = "balanced") -> str:
        """Inteligentní výběr modelu podle dotazu a systémových zdrojů"""
        if not self.config:
            return "llama3.1:8b"

        # Kontrola dostupných zdrojů
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=0.1)

        # Auto-detect query complexity
        query_length = len(query.split())
        is_complex = query_length > 50 or any(word in query.lower() for word in
                                             ['mechanism', 'pharmacokinetics', 'clinical', 'research'])

        # Model selection logic
        if priority == "speed" or memory_usage > 85:
            return self.config.ai.local_ai.speed_model  # tinyllama:1.1b
        elif priority == "balanced" and not is_complex:
            return self.config.ai.local_ai.fallback_model  # phi3:mini
        elif is_complex or priority == "quality":
            return self.config.ai.local_ai.primary_model  # llama3.1:8b
        else:
            return self.config.ai.local_ai.fallback_model

    async def ensure_model_loaded(self, model_name: str) -> bool:
        """Zajistí že model je načten a připraven"""
        try:
            async with aiohttp.ClientSession() as session:
                # Check if model is loaded
                check_url = f"{self.config.ai.local_ai.ollama_host}/api/ps"
                async with session.get(check_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        loaded_models = [model.get('name', '') for model in data.get('models', [])]

                        if model_name not in loaded_models:
                            # Load model
                            logger.info(f"Loading model {model_name}...")
                            load_url = f"{self.config.ai.local_ai.ollama_host}/api/generate"
                            payload = {
                                "model": model_name,
                                "prompt": "warmup",
                                "stream": False
                            }
                            async with session.post(load_url, json=payload) as load_response:
                                if load_response.status == 200:
                                    logger.info(f"✅ Model {model_name} loaded successfully")
                                    self.current_model = model_name
                                    self.model_loaded_time = time.time()
                                    return True
                        else:
                            self.current_model = model_name
                            return True
            return False
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

class M1OptimizedOllamaClient:
    """Ultra-optimized Ollama client pro MacBook Air M1"""

    def __init__(self):
        self.config = get_config() if CONFIG_AVAILABLE else None
        self.model_manager = IntelligentModelManager()
        self.response_cache: Dict[str, Any] = {}
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _get_cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key for response"""
        content = f"{model}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    async def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available"""
        cache_file = self.model_manager.cache_dir / f"{cache_key}.json"
        try:
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check if cache is still valid (24 hours)
                    if time.time() - data.get('timestamp', 0) < 86400:
                        return data.get('response')
        except Exception:
            pass
        return None

    async def _cache_response(self, cache_key: str, response: str):
        """Cache response to disk"""
        cache_file = self.model_manager.cache_dir / f"{cache_key}.json"
        try:
            data = {
                'response': response,
                'timestamp': time.time()
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    async def generate_optimized(self,
                               query: str,
                               priority: str = "balanced",
                               use_specialized_prompt: bool = True) -> str:
        """Ultra-optimized generation with intelligent model selection"""

        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        # Get specialized prompt if available
        if use_specialized_prompt and PEPTIDE_PROMPTS_AVAILABLE:
            category, prompt_type = detect_query_type(query)
            final_prompt = get_prompt(category, prompt_type, query)
        else:
            final_prompt = f"Provide detailed, evidence-based information about: {query}"

        # Select optimal model
        model = await self.model_manager.get_optimal_model(query, priority)

        # Check cache first
        cache_key = self._get_cache_key(final_prompt, model)
        cached_response = await self._get_cached_response(cache_key)
        if cached_response:
            logger.info(f"✅ Cache hit for query (model: {model})")
            return cached_response

        # Ensure model is loaded
        if not await self.model_manager.ensure_model_loaded(model):
            # Fallback to smaller model
            model = self.config.ai.local_ai.fallback_model if self.config else "phi3:mini"
            await self.model_manager.ensure_model_loaded(model)

        # Generate response
        start_time = time.time()
        try:
            url = f"{self.config.ai.local_ai.ollama_host}/api/generate" if self.config else "http://localhost:11434/api/generate"

            payload = {
                "model": model,
                "prompt": final_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.85,
                    "num_predict": 1024,
                    "num_ctx": 8192 if "llama3.1" in model else 4096,
                    "repeat_penalty": 1.1,
                    "num_threads": 8
                }
            }

            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get('response', '').strip()

                    # Cache successful response
                    await self._cache_response(cache_key, result)

                    # Log performance
                    duration = time.time() - start_time
                    tokens = len(result.split())
                    tokens_per_second = tokens / duration if duration > 0 else 0

                    logger.info(f"✅ Generated {tokens} tokens in {duration:.2f}s "
                              f"({tokens_per_second:.1f} tok/s) using {model}")

                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama API error {response.status}: {error_text}")
                    return f"Error generating response: {error_text}"

        except Exception as e:
            logger.error(f"Generation failed with {model}: {e}")
            return f"Error: {str(e)}"

# Convenience functions for quick access
async def quick_ai_query(query: str, priority: str = "balanced") -> str:
    """Quick AI query with automatic optimization"""
    async with M1OptimizedOllamaClient() as client:
        return await client.generate_optimized(query, priority=priority)

async def peptide_research_query(peptide_name: str) -> str:
    """Specialized peptide research query"""
    async with M1OptimizedOllamaClient() as client:
        return await client.generate_optimized(
            f"Research peptide: {peptide_name}",
            priority="quality",
            use_specialized_prompt=True
        )

async def quick_dosage_check(compound: str) -> str:
    """Quick dosage information"""
    async with M1OptimizedOllamaClient() as client:
        return await client.generate_optimized(
            f"dosage protocol for {compound}",
            priority="speed",
            use_specialized_prompt=True
        )
