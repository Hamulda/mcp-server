"""
Optimalizovaný TextProcessor pro maximální rychlost
Bez zbytečných kontrol, agresivní caching
"""
import re
import tiktoken
from typing import List, Dict, Any, Tuple
from collections import Counter
import logging
from high_performance_cache import cache_result, high_perf_cache

class FastTextProcessor:
    """Ultrarychlý text processor pro soukromé použití"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.encoder = None
        self._setup_tokenizer()

        # Předkompilované regex pro rychlost
        self.clean_regex = re.compile(r'[^\w\s]|[\d]+', re.IGNORECASE)
        self.sentence_regex = re.compile(r'[.!?]+')

        # Agresivní cache klíčových slov
        self.medical_keywords = {
            "nootropika", "peptidy", "medikace", "psychické", "poruchy",
            "peptide", "supplement", "medication", "clinical", "trial",
            "efficacy", "dosage", "treatment", "therapy", "biomarker"
        }

        self.research_keywords = {
            "research", "analysis", "trend", "key", "finding",
            "study", "výzkum", "analýza", "závěr", "výsledek"
        }

    def _setup_tokenizer(self):
        """Rychlé nastavení tokenizeru"""
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoder = None

    @cache_result(ttl_hours=48)  # Dlouhý cache pro clean text
    def clean_text(self, text: str) -> str:
        """Ultrarychlé čištění textu"""
        if not text:
            return ""

        # Jeden regex pass místo několika
        cleaned = self.clean_regex.sub(' ', text)
        return ' '.join(cleaned.split())  # Normalizace whitespace

    @cache_result(ttl_hours=24)
    def count_tokens(self, text: str) -> int:
        """Rychlé počítání tokenů s agresivním cachingem"""
        if not text:
            return 0

        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            return len(text.split())  # Fallback

    def prioritize_by_keywords(self, text: str, keywords: List[str] = None, max_sentences: int = 10) -> str:
        """Rychlá prioritizace bez zbytečných kontrol"""
        if not keywords:
            keywords = self.research_keywords

        sentences = self.sentence_regex.split(text)
        if len(sentences) <= max_sentences:
            return text

        # Rychlé skórování - jen počítání výskytů
        scored = []
        keywords_set = set(kw.lower() for kw in keywords)

        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for kw in keywords_set if kw in sentence_lower)
            if score > 0:  # Jen relevantní věty
                scored.append((sentence, score))

        # Seřaď a vezmi top N
        scored.sort(key=lambda x: x[1], reverse=True)
        return ' '.join(s[0] for s in scored[:max_sentences])

    @cache_result(ttl_hours=12)
    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[Tuple[str, int]]:
        """Rychlá extrakce klíčových slov"""
        if not text:
            return []

        # Rychlé word splitting bez složitého preprocessing
        words = [w.lower() for w in text.split() if len(w) > 2]

        # Základní stop words (jen nejčastější)
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'je', 'na', 'se', 'to'}

        # Filtruj a počítej
        filtered_words = [w for w in words if w not in stop_words]
        word_freq = Counter(filtered_words)

        return word_freq.most_common(max_keywords)

    def optimize_for_tokens(self, text: str, max_tokens: int = 3000) -> str:
        """Agresivní optimalizace pro tokeny"""
        current_tokens = self.count_tokens(text)

        if current_tokens <= max_tokens:
            return text

        # Rychlé zkrácení - vezmi jen první část
        ratio = max_tokens / current_tokens
        target_length = int(len(text) * ratio * 0.9)  # Safety margin

        return text[:target_length]

    def distill_medical_text(self, text: str, max_sentences: int = 10) -> str:
        """Rychlá lékařská destilace"""
        return self.prioritize_by_keywords(text, list(self.medical_keywords), max_sentences)

    def analyze_sentiment_basic(self, text: str) -> Dict[str, float]:
        """Základní rychlá analýza sentimentu"""
        if not text:
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

        # Minimální seznamy pro rychlost
        pos_words = {'good', 'great', 'excellent', 'effective', 'successful'}
        neg_words = {'bad', 'terrible', 'poor', 'failed', 'ineffective'}

        words_set = set(text.lower().split())

        pos_count = len(words_set & pos_words)
        neg_count = len(words_set & neg_words)
        total = pos_count + neg_count

        if total == 0:
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

        pos_score = pos_count / total
        neg_score = neg_count / total

        return {
            "positive": pos_score,
            "neutral": max(0, 1 - pos_score - neg_score),
            "negative": neg_score
        }

# Globální instance pro rychlý přístup
fast_text_processor = FastTextProcessor()
