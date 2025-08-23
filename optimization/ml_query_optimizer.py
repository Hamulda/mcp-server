"""
ML-Based Query Optimization System - Phase 2
Implementuje pokročilé ML optimalizace pro research queries
- Query similarity clustering
- Automatic query reformulation
- Research topic classification
- Predictive caching
"""

import asyncio
import numpy as np
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from collections import defaultdict, deque
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Typy research queries"""
    ACADEMIC_SEARCH = "academic_search"
    CITATION_LOOKUP = "citation_lookup"
    TOPIC_EXPLORATION = "topic_exploration"
    METHODOLOGY_RESEARCH = "methodology_research"
    TREND_ANALYSIS = "trend_analysis"
    COMPARATIVE_STUDY = "comparative_study"

@dataclass
class QueryPattern:
    """Pattern pro podobné queries"""
    cluster_id: int
    representative_query: str
    keywords: List[str]
    query_type: QueryType
    avg_response_time: float
    frequency: int
    success_rate: float
    last_seen: float

@dataclass
class QueryOptimizationMetrics:
    """Metriky ML optimalizace"""
    total_queries: int = 0
    optimized_queries: int = 0
    cache_hits_from_ml: int = 0
    avg_optimization_time: float = 0.0
    classification_accuracy: float = 0.0
    clustering_efficiency: float = 0.0

class ResearchTopicClassifier:
    """Klasifikátor research topics pomocí ML"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self.topic_keywords = {
            QueryType.ACADEMIC_SEARCH: [
                'research', 'study', 'analysis', 'paper', 'journal', 'academic',
                'scholar', 'publication', 'thesis', 'dissertation'
            ],
            QueryType.CITATION_LOOKUP: [
                'cite', 'citation', 'reference', 'bibliography', 'doi', 'pmid',
                'author', 'year', 'volume', 'issue'
            ],
            QueryType.TOPIC_EXPLORATION: [
                'overview', 'survey', 'review', 'introduction', 'basics',
                'fundamentals', 'what is', 'definition'
            ],
            QueryType.METHODOLOGY_RESEARCH: [
                'method', 'methodology', 'approach', 'technique', 'procedure',
                'protocol', 'framework', 'algorithm', 'implementation'
            ],
            QueryType.TREND_ANALYSIS: [
                'trend', 'evolution', 'development', 'progress', 'advancement',
                'future', 'emerging', 'recent', 'latest', 'current'
            ],
            QueryType.COMPARATIVE_STUDY: [
                'compare', 'comparison', 'versus', 'vs', 'difference',
                'similarity', 'contrast', 'evaluation', 'assessment'
            ]
        }
        self.stemmer = PorterStemmer()

    def preprocess_query(self, query: str) -> str:
        """Preprocessing query pro ML"""
        # Tokenizace a odstranění stop words
        tokens = word_tokenize(query.lower())
        stop_words = set(stopwords.words('english'))

        # Stemming a filtrace
        processed_tokens = []
        for token in tokens:
            if token.isalpha() and token not in stop_words:
                stemmed = self.stemmer.stem(token)
                processed_tokens.append(stemmed)

        return ' '.join(processed_tokens)

    def classify_query_type(self, query: str) -> Tuple[QueryType, float]:
        """Klasifikuje typ research query"""
        preprocessed = self.preprocess_query(query)

        best_type = QueryType.ACADEMIC_SEARCH
        best_score = 0.0

        for query_type, keywords in self.topic_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in preprocessed:
                    score += 1.0

            # Normalizace podle délky query
            normalized_score = score / max(len(preprocessed.split()), 1)

            if normalized_score > best_score:
                best_score = normalized_score
                best_type = query_type

        # Minimum confidence threshold
        confidence = min(best_score, 1.0)

        return best_type, confidence

class QuerySimilarityClusterer:
    """Clustering podobných queries pro optimalizaci"""

    def __init__(self, max_clusters: int = 50):
        self.max_clusters = max_clusters
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.kmeans = None
        self.query_patterns: Dict[int, QueryPattern] = {}
        self.query_history: deque = deque(maxlen=1000)

    def add_query(self, query: str, response_time: float, success: bool):
        """Přidává query do historie pro clustering"""
        self.query_history.append({
            'query': query,
            'response_time': response_time,
            'success': success,
            'timestamp': time.time()
        })

    def update_clusters(self):
        """Aktualizuje clustery na základě nových dat"""
        if len(self.query_history) < 10:
            return

        # Extrakce queries
        queries = [item['query'] for item in self.query_history]

        try:
            # Vektorizace
            X = self.vectorizer.fit_transform(queries)

            # Determine optimal number of clusters
            n_clusters = min(self.max_clusters, len(queries) // 5)
            if n_clusters < 2:
                return

            # K-means clustering
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = self.kmeans.fit_predict(X)

            # Update patterns
            self._update_patterns(queries, cluster_labels)

        except Exception as e:
            logger.error(f"Clustering error: {e}")

    def _update_patterns(self, queries: List[str], labels: List[int]):
        """Aktualizuje query patterns na základě clusterů"""
        cluster_data = defaultdict(list)

        # Group queries by cluster
        for query, label, item in zip(queries, labels, self.query_history):
            cluster_data[label].append({
                'query': query,
                'response_time': item['response_time'],
                'success': item['success']
            })

        # Update patterns
        for cluster_id, items in cluster_data.items():
            if len(items) < 2:
                continue

            # Calculate statistics
            avg_response_time = np.mean([item['response_time'] for item in items])
            success_rate = np.mean([item['success'] for item in items])

            # Find representative query (most frequent or shortest)
            query_counts = defaultdict(int)
            for item in items:
                query_counts[item['query']] += 1

            representative = min(query_counts.keys(), key=len)

            # Extract keywords (most common words in cluster)
            all_words = []
            for item in items:
                all_words.extend(item['query'].lower().split())

            word_freq = defaultdict(int)
            for word in all_words:
                if len(word) > 3:  # Filter short words
                    word_freq[word] += 1

            keywords = sorted(word_freq.keys(), key=word_freq.get, reverse=True)[:5]

            self.query_patterns[cluster_id] = QueryPattern(
                cluster_id=cluster_id,
                representative_query=representative,
                keywords=keywords,
                query_type=QueryType.ACADEMIC_SEARCH,  # Default, can be improved
                avg_response_time=avg_response_time,
                frequency=len(items),
                success_rate=success_rate,
                last_seen=time.time()
            )

    def find_similar_queries(self, query: str, threshold: float = 0.7) -> List[QueryPattern]:
        """Najde podobné queries pro optimalizaci"""
        if not self.kmeans or not self.query_patterns:
            return []

        try:
            # Vektorizace nového query
            query_vector = self.vectorizer.transform([query])

            # Najdi nejbližší cluster
            cluster_label = self.kmeans.predict(query_vector)[0]

            # Získej pattern pro cluster
            if cluster_label in self.query_patterns:
                pattern = self.query_patterns[cluster_label]

                # Calculate similarity to representative query
                rep_vector = self.vectorizer.transform([pattern.representative_query])
                similarity = cosine_similarity(query_vector, rep_vector)[0][0]

                if similarity >= threshold:
                    return [pattern]

        except Exception as e:
            logger.error(f"Similarity search error: {e}")

        return []

class QueryReformulator:
    """Automatická reformulace queries pro lepší výsledky"""

    def __init__(self):
        self.academic_enhancers = [
            'research', 'study', 'analysis', 'scholarly', 'peer-reviewed'
        ]
        self.specificity_enhancers = [
            'methodology', 'framework', 'approach', 'technique'
        ]

    def reformulate_query(self, query: str, query_type: QueryType) -> List[str]:
        """Generuje reformulované verze query"""
        reformulations = []

        # Original query
        reformulations.append(query)

        # Add academic context
        if query_type == QueryType.ACADEMIC_SEARCH:
            for enhancer in self.academic_enhancers[:2]:
                reformed = f"{query} {enhancer}"
                reformulations.append(reformed)

        # Add specificity
        if query_type == QueryType.METHODOLOGY_RESEARCH:
            for enhancer in self.specificity_enhancers[:2]:
                reformed = f"{enhancer} {query}"
                reformulations.append(reformed)

        # Add temporal context for trends
        if query_type == QueryType.TREND_ANALYSIS:
            temporal_terms = ['recent', 'latest', '2024', 'current']
            for term in temporal_terms[:2]:
                reformed = f"{term} {query}"
                reformulations.append(reformed)

        return reformulations[:5]  # Limit to 5 reformulations

class MLQueryOptimizer:
    """Hlavní ML optimizer pro research queries"""

    def __init__(self):
        self.classifier = ResearchTopicClassifier()
        self.clusterer = QuerySimilarityClusterer()
        self.reformulator = QueryReformulator()
        self.metrics = QueryOptimizationMetrics()
        self.prediction_cache: Dict[str, Any] = {}
        self._last_cluster_update = time.time()

    async def optimize_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Optimalizuje query pomocí ML"""
        start_time = time.time()
        self.metrics.total_queries += 1

        try:
            # 1. Klasifikace typu query
            query_type, confidence = self.classifier.classify_query_type(query)

            # 2. Hledání podobných queries
            similar_patterns = self.clusterer.find_similar_queries(query)

            # 3. Reformulace queries
            reformulations = self.reformulator.reformulate_query(query, query_type)

            # 4. Predikce optimální strategie
            optimization_strategy = self._predict_optimization_strategy(
                query, query_type, similar_patterns, context
            )

            # 5. Update metrics
            optimization_time = time.time() - start_time
            self._update_optimization_metrics(optimization_time)

            return {
                'original_query': query,
                'query_type': query_type.value,
                'classification_confidence': confidence,
                'reformulations': reformulations,
                'similar_patterns': [p.__dict__ for p in similar_patterns],
                'optimization_strategy': optimization_strategy,
                'processing_time_ms': optimization_time * 1000
            }

        except Exception as e:
            logger.error(f"Query optimization error: {e}")
            return {
                'original_query': query,
                'error': str(e),
                'fallback_strategy': 'direct_search'
            }

    def _predict_optimization_strategy(
        self,
        query: str,
        query_type: QueryType,
        similar_patterns: List[QueryPattern],
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Predikuje optimální strategii pro query"""

        strategy = {
            'use_cache': False,
            'parallel_search': False,
            'reformulate': False,
            'source_prioritization': [],
            'estimated_response_time': 5.0
        }

        # Pokud máme podobné patterns s dobrou success rate
        if similar_patterns:
            best_pattern = max(similar_patterns, key=lambda p: p.success_rate)

            if best_pattern.success_rate > 0.8:
                strategy['use_cache'] = True
                strategy['estimated_response_time'] = best_pattern.avg_response_time

        # Paralelní search pro komplexní queries
        if len(query.split()) > 5 or query_type == QueryType.COMPARATIVE_STUDY:
            strategy['parallel_search'] = True

        # Reformulace pro low-confidence nebo nové typy
        if (not similar_patterns or
            (similar_patterns and min(p.success_rate for p in similar_patterns) < 0.6)):
            strategy['reformulate'] = True

        # Source prioritization podle typu
        if query_type == QueryType.ACADEMIC_SEARCH:
            strategy['source_prioritization'] = ['pubmed', 'arxiv', 'scholar']
        elif query_type == QueryType.CITATION_LOOKUP:
            strategy['source_prioritization'] = ['crossref', 'scholar', 'pubmed']
        elif query_type == QueryType.TREND_ANALYSIS:
            strategy['source_prioritization'] = ['arxiv', 'scholar', 'news']

        return strategy

    def record_query_result(self, query: str, response_time: float, success: bool):
        """Zaznamenává výsledky pro ML learning"""
        self.clusterer.add_query(query, response_time, success)

        # Update clusters periodically
        if time.time() - self._last_cluster_update > 300:  # Every 5 minutes
            self.clusterer.update_clusters()
            self._last_cluster_update = time.time()

    def _update_optimization_metrics(self, optimization_time: float):
        """Aktualizuje metriky optimalizace"""
        self.metrics.optimized_queries += 1

        # Exponential moving average
        alpha = 0.1
        self.metrics.avg_optimization_time = (
            (1 - alpha) * self.metrics.avg_optimization_time +
            alpha * optimization_time
        )

    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Získá statistiky ML optimalizace"""
        return {
            'total_queries': self.metrics.total_queries,
            'optimized_queries': self.metrics.optimized_queries,
            'optimization_rate': (
                self.metrics.optimized_queries / max(self.metrics.total_queries, 1)
            ),
            'avg_optimization_time_ms': self.metrics.avg_optimization_time * 1000,
            'active_patterns': len(self.clusterer.query_patterns),
            'cache_efficiency': (
                self.metrics.cache_hits_from_ml / max(self.metrics.optimized_queries, 1)
            )
        }

    async def export_learned_patterns(self) -> Dict[str, Any]:
        """Exportuje naučené patterns pro analysis"""
        return {
            'query_patterns': {
                str(k): v.__dict__ for k, v in self.clusterer.query_patterns.items()
            },
            'topic_classification_accuracy': self.metrics.classification_accuracy,
            'clustering_efficiency': self.metrics.clustering_efficiency
        }

# Factory funkce
async def create_ml_query_optimizer() -> MLQueryOptimizer:
    """Factory pro vytvoření ML query optimizeru"""
    optimizer = MLQueryOptimizer()

    # Download NLTK data if needed
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        logger.warning(f"NLTK download failed: {e}")

    logger.info("ML Query Optimizer initialized")
    return optimizer
