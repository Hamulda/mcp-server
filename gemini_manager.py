"""
Ultraúsporný Gemini manager - cíl: být levnější než Perplexity
"""
import google.generativeai as genai
import asyncio
from typing import List, Dict, Any
import logging
import json
import re
from high_performance_cache import cache_result
from config_personal import *
from cost_optimizer import cost_optimizer

class UltraCheapGeminiManager:
    """Maximálně úsporný Gemini manager"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self.model = self._setup_gemini()
        self.daily_api_calls = 0
        self.max_daily_calls = 50  # Limit API volání za den

    def _setup_gemini(self):
        """Rychlé nastavení Gemini"""
        try:
            genai.configure(api_key=self.api_key)
            return genai.GenerativeModel('gemini-pro')
        except Exception as e:
            self.logger.error(f"Gemini setup failed: {e}")
            return None

    @cache_result(ttl_hours=72)  # 3denní cache pro maximální úsporu
    async def analyze_batch_cheaply(self, texts: List[str], analysis_type: str) -> List[Dict[str, Any]]:
        """Ultraúsporná batch analýza"""
        if not texts or not self.model:
            return []

        # Kontrola denního limitu API volání
        if self.daily_api_calls >= self.max_daily_calls:
            self.logger.warning("🚫 Denní limit API volání dosažen - používám cache")
            return self._fallback_analysis(texts)

        # Agresivní zkrácení textů pro úsporu tokenů
        short_texts = [text[:500] for text in texts[:BATCH_SIZE//2]]  # Půlka batch size

        # Odhad nákladů
        estimated_cost = len(' '.join(short_texts)) * 0.000001  # Velmi konzervativní odhad

        # Kontrola rozpočtu
        if not cost_optimizer.should_use_api(estimated_cost):
            self.logger.info("💰 Šetřím - používám fallback analýzu")
            return self._fallback_analysis(texts)

        # Minimalistický prompt pro úsporu tokenů
        prompt = f"Krátce JSON: {chr(10).join(short_texts)}"

        try:
            self.daily_api_calls += 1
            response = await self._cheap_gemini_call(prompt)
            return self._fast_parse(response, len(texts))
        except Exception as e:
            self.logger.error(f"Cheap analysis failed: {e}")
            return self._fallback_analysis(texts)

    async def _cheap_gemini_call(self, prompt: str) -> str:
        """Nejlevnější možné volání Gemini"""
        loop = asyncio.get_event_loop()

        def sync_call():
            return self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,  # Nulová pro konzistentní cache
                    top_p=0.1,     # Nízká pro rychlost
                    max_output_tokens=500,  # Minimum pro úsporu
                )
            ).text

        return await loop.run_in_executor(None, sync_call)

    def _fallback_analysis(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Fallback analýza bez API nákladů"""
        from text_processing_utils import fast_text_processor

        results = []
        for text in texts:
            keywords = fast_text_processor.extract_keywords(text, 5)
            sentiment = fast_text_processor.analyze_sentiment_basic(text)

            results.append({
                "summary": text[:200] + "...",
                "keywords": [kw[0] for kw in keywords],
                "sentiment": sentiment["positive"] > 0.5,
                "source": "local_analysis"
            })

        return results

    def _fast_parse(self, response: str, expected_count: int) -> List[Dict[str, Any]]:
        """Rychlé parsování"""
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        # Fallback - rozdel response
        return [{"content": response}] if expected_count == 1 else [{"content": f"Part {i+1}"} for i in range(expected_count)]

    def get_daily_cost_summary(self) -> Dict[str, float]:
        """Denní přehled nákladů"""
        estimated_daily_cost = self.daily_api_calls * 0.01  # Odhad $0.01 za call
        return {
            "api_calls_today": self.daily_api_calls,
            "estimated_cost": estimated_daily_cost,
            "remaining_budget": max(0, DAILY_COST_LIMIT - estimated_daily_cost),
            "calls_remaining": max(0, self.max_daily_calls - self.daily_api_calls)
        }
