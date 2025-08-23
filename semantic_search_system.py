"""
Semantic Search Capabilities - Phase 3
Implementuje vector embeddings pro research papers
- Citation network analysis
- Research trend prediction
- Semantic similarity search
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
from sentence_transformers import SentenceTransformer
import faiss
import networkx as nx
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EmbeddingModel(Enum):
    """Dostupné embedding modely"""
    SCIENTIFIC_PAPERS = "allenai-specter"
    GENERAL_PURPOSE = "all-MiniLM-L6-v2"
    MULTILINGUAL = "paraphrase-multilingual-MiniLM-L12-v2"

@dataclass
class ResearchPaper:
    """Reprezentace research paperu pro semantic search"""
    id: str
    title: str
    abstract: str
    authors: List[str]
    year: int
    keywords: List[str]
    doi: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    semantic_score: float = 0.0

@dataclass
class CitationNetwork:
    """Síť citací pro analýzu"""
    papers: Dict[str, ResearchPaper]
    citation_graph: nx.DiGraph
    influence_scores: Dict[str, float]
    trending_topics: List[str]
    last_updated: float

class SemanticSearchEngine:
    """Pokročilý semantic search engine pro research"""

    def __init__(self, model_name: EmbeddingModel = EmbeddingModel.SCIENTIFIC_PAPERS):
        self.model_name = model_name
        self.embedding_model = None
        self.index = None
        self.papers_db: Dict[str, ResearchPaper] = {}
        self.dimension = 768  # Default dimension
        self.citation_network = None
        self._initialize_model()

    def _initialize_model(self):
        """Inicializuje embedding model"""
        try:
            if self.model_name == EmbeddingModel.SCIENTIFIC_PAPERS:
                # Specialized model pro scientific papers
                model_path = "allenai-specter"
            elif self.model_name == EmbeddingModel.GENERAL_PURPOSE:
                model_path = "all-MiniLM-L6-v2"
            else:
                model_path = "paraphrase-multilingual-MiniLM-L12-v2"

            self.embedding_model = SentenceTransformer(model_path)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()

            # Initialize FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for similarity

            logger.info(f"Semantic search engine initialized with {model_path}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            # Fallback to simple embeddings
            self._initialize_fallback_model()

    def _initialize_fallback_model(self):
        """Fallback embedding model"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.embedding_model = TfidfVectorizer(max_features=768, stop_words='english')
        self.dimension = 768
        self.index = faiss.IndexFlatIP(self.dimension)
        logger.info("Using fallback TF-IDF embeddings")

    async def add_paper(self, paper: ResearchPaper) -> bool:
        """Přidá paper do semantic search indexu"""
        try:
            # Generate embedding
            text_for_embedding = f"{paper.title} {paper.abstract}"

            if hasattr(self.embedding_model, 'encode'):
                # SentenceTransformer
                embedding = self.embedding_model.encode(text_for_embedding)
            else:
                # Fallback TF-IDF
                embedding = self._generate_tfidf_embedding(text_for_embedding)

            paper.embedding = embedding

            # Add to FAISS index
            self.index.add(embedding.reshape(1, -1).astype('float32'))

            # Store in database
            self.papers_db[paper.id] = paper

            return True

        except Exception as e:
            logger.error(f"Failed to add paper {paper.id}: {e}")
            return False

    def _generate_tfidf_embedding(self, text: str) -> np.ndarray:
        """Generuje TF-IDF embedding jako fallback"""
        # Simple fallback - would need proper implementation
        words = text.lower().split()
        embedding = np.zeros(self.dimension)

        for i, word in enumerate(words[:self.dimension]):
            embedding[i] = hash(word) % 1000 / 1000.0

        return embedding

    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[ResearchPaper, float]]:
        """Provede semantic search"""

        if not self.papers_db:
            return []

        try:
            # Generate query embedding
            if hasattr(self.embedding_model, 'encode'):
                query_embedding = self.embedding_model.encode(query)
            else:
                query_embedding = self._generate_tfidf_embedding(query)

            # Search in FAISS index
            scores, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'),
                min(top_k, len(self.papers_db))
            )

            # Prepare results
            results = []
            paper_ids = list(self.papers_db.keys())

            for score, idx in zip(scores[0], indices[0]):
                if idx < len(paper_ids) and score >= similarity_threshold:
                    paper_id = paper_ids[idx]
                    paper = self.papers_db[paper_id]
                    results.append((paper, float(score)))

            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    async def find_similar_papers(
        self,
        paper_id: str,
        top_k: int = 5
    ) -> List[Tuple[ResearchPaper, float]]:
        """Najde podobné papers"""

        if paper_id not in self.papers_db:
            return []

        reference_paper = self.papers_db[paper_id]
        if reference_paper.embedding is None:
            return []

        try:
            # Search using paper's embedding
            scores, indices = self.index.search(
                reference_paper.embedding.reshape(1, -1).astype('float32'),
                min(top_k + 1, len(self.papers_db))  # +1 to exclude self
            )

            results = []
            paper_ids = list(self.papers_db.keys())

            for score, idx in zip(scores[0], indices[0]):
                if idx < len(paper_ids):
                    similar_paper_id = paper_ids[idx]
                    if similar_paper_id != paper_id:  # Exclude self
                        similar_paper = self.papers_db[similar_paper_id]
                        results.append((similar_paper, float(score)))

            return results[:top_k]

        except Exception as e:
            logger.error(f"Similar papers search failed: {e}")
            return []

class CitationNetworkAnalyzer:
    """Analyzér citačních sítí pro trend prediction"""

    def __init__(self):
        self.citation_graph = nx.DiGraph()
        self.papers: Dict[str, ResearchPaper] = {}
        self.influence_cache = {}
        self._last_analysis_time = 0

    def add_paper_with_citations(self, paper: ResearchPaper):
        """Přidá paper s citacemi do network grafu"""
        self.papers[paper.id] = paper
        self.citation_graph.add_node(paper.id, **{
            'title': paper.title,
            'year': paper.year,
            'authors': paper.authors
        })

        # Add citation edges
        for cited_paper_id in paper.citations:
            if cited_paper_id in self.papers:
                self.citation_graph.add_edge(paper.id, cited_paper_id)

    def calculate_influence_scores(self) -> Dict[str, float]:
        """Vypočítá influence scores pomocí PageRank"""
        try:
            if len(self.citation_graph.nodes()) == 0:
                return {}

            # PageRank pro influence scoring
            pagerank_scores = nx.pagerank(self.citation_graph, weight='weight')

            # Combine s citation count
            influence_scores = {}
            for paper_id, pagerank_score in pagerank_scores.items():
                citation_count = self.citation_graph.in_degree(paper_id)

                # Weighted combination
                influence_scores[paper_id] = (
                    0.7 * pagerank_score +
                    0.3 * (citation_count / max(len(self.papers), 1))
                )

            self.influence_cache = influence_scores
            self._last_analysis_time = time.time()

            return influence_scores

        except Exception as e:
            logger.error(f"Influence calculation failed: {e}")
            return {}

    def detect_trending_topics(self, time_window_months: int = 12) -> List[Tuple[str, float]]:
        """Detekuje trending topics na základě citací"""
        current_time = time.time()
        cutoff_time = current_time - (time_window_months * 30 * 24 * 3600)

        # Get recent papers
        recent_papers = []
        for paper in self.papers.values():
            paper_timestamp = datetime(paper.year, 1, 1).timestamp()
            if paper_timestamp >= cutoff_time:
                recent_papers.append(paper)

        # Analyze keyword trends
        keyword_trends = defaultdict(list)

        for paper in recent_papers:
            paper_influence = self.influence_cache.get(paper.id, 0.0)

            for keyword in paper.keywords:
                keyword_trends[keyword].append(paper_influence)

        # Calculate trending scores
        trending_scores = []
        for keyword, influences in keyword_trends.items():
            if len(influences) >= 3:  # Minimum occurrence
                trend_score = np.mean(influences) * len(influences)
                trending_scores.append((keyword, trend_score))

        # Sort by trend score
        trending_scores.sort(key=lambda x: x[1], reverse=True)

        return trending_scores[:20]  # Top 20 trends

    def find_research_gaps(self) -> List[Dict[str, Any]]:
        """Identifikuje research gaps v citation network"""
        gaps = []

        # Find papers with high citations but few follow-ups
        for paper_id, paper in self.papers.items():
            in_degree = self.citation_graph.in_degree(paper_id)
            out_degree = self.citation_graph.out_degree(paper_id)

            # High impact but low follow-up ratio
            if in_degree > 5 and out_degree / max(in_degree, 1) < 0.3:
                gaps.append({
                    'paper_id': paper_id,
                    'title': paper.title,
                    'gap_score': in_degree / max(out_degree, 1),
                    'reason': 'high_impact_low_followup'
                })

        return sorted(gaps, key=lambda x: x['gap_score'], reverse=True)[:10]

class ResearchTrendPredictor:
    """Predikce research trendů pomocí ML"""

    def __init__(self):
        self.topic_evolution = defaultdict(list)
        self.prediction_model = None
        self._initialize_predictor()

    def _initialize_predictor(self):
        """Inicializuje prediction model"""
        # Simple trend prediction using linear regression
        from sklearn.linear_model import LinearRegression
        self.prediction_model = LinearRegression()

    def add_temporal_data(self, topic: str, year: int, popularity_score: float):
        """Přidá temporální data pro topic"""
        self.topic_evolution[topic].append((year, popularity_score))

    def predict_future_trends(self, years_ahead: int = 2) -> Dict[str, float]:
        """Predikuje budoucí trendy"""
        predictions = {}
        current_year = datetime.now().year

        for topic, data_points in self.topic_evolution.items():
            if len(data_points) >= 3:  # Minimum data pro prediction
                # Prepare data
                years = np.array([point[0] for point in data_points]).reshape(-1, 1)
                scores = np.array([point[1] for point in data_points])

                try:
                    # Fit model
                    self.prediction_model.fit(years, scores)

                    # Predict future
                    future_year = current_year + years_ahead
                    predicted_score = self.prediction_model.predict([[future_year]])[0]

                    predictions[topic] = max(0.0, predicted_score)  # No negative predictions

                except Exception as e:
                    logger.warning(f"Prediction failed for topic {topic}: {e}")

        return predictions

class UnifiedSemanticSystem:
    """Unified systém pro semantic search a trend analysis"""

    def __init__(self):
        self.search_engine = SemanticSearchEngine()
        self.citation_analyzer = CitationNetworkAnalyzer()
        self.trend_predictor = ResearchTrendPredictor()
        self.paper_count = 0

    async def add_research_paper(self, paper_data: Dict[str, Any]) -> str:
        """Přidá research paper do všech subsystémů"""
        paper_id = paper_data.get('id', f"paper_{self.paper_count}")
        self.paper_count += 1

        # Create ResearchPaper object
        paper = ResearchPaper(
            id=paper_id,
            title=paper_data.get('title', ''),
            abstract=paper_data.get('abstract', ''),
            authors=paper_data.get('authors', []),
            year=paper_data.get('year', datetime.now().year),
            keywords=paper_data.get('keywords', []),
            doi=paper_data.get('doi'),
            citations=paper_data.get('citations', [])
        )

        # Add to all systems
        await self.search_engine.add_paper(paper)
        self.citation_analyzer.add_paper_with_citations(paper)

        # Update trend data
        for keyword in paper.keywords:
            self.trend_predictor.add_temporal_data(keyword, paper.year, 1.0)

        return paper_id

    async def comprehensive_search(
        self,
        query: str,
        include_trends: bool = True,
        include_gaps: bool = True
    ) -> Dict[str, Any]:
        """Komprehensivní search kombinující všechny funkce"""

        # Semantic search
        semantic_results = await self.search_engine.semantic_search(query, top_k=10)

        # Analyze results
        results = {
            'query': query,
            'semantic_results': [],
            'total_found': len(semantic_results),
            'search_time': time.time()
        }

        for paper, score in semantic_results:
            results['semantic_results'].append({
                'paper_id': paper.id,
                'title': paper.title,
                'abstract': paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract,
                'authors': paper.authors,
                'year': paper.year,
                'semantic_score': score,
                'doi': paper.doi
            })

        # Add trending topics if requested
        if include_trends:
            trending_topics = self.citation_analyzer.detect_trending_topics()
            results['trending_topics'] = [
                {'topic': topic, 'trend_score': score}
                for topic, score in trending_topics[:10]
            ]

            # Future predictions
            future_trends = self.trend_predictor.predict_future_trends()
            results['future_predictions'] = [
                {'topic': topic, 'predicted_popularity': score}
                for topic, score in sorted(future_trends.items(), key=lambda x: x[1], reverse=True)[:5]
            ]

        # Add research gaps if requested
        if include_gaps:
            research_gaps = self.citation_analyzer.find_research_gaps()
            results['research_gaps'] = research_gaps[:5]

        return results

    async def get_paper_recommendations(self, paper_id: str) -> Dict[str, Any]:
        """Získá doporučení na základě paperu"""
        if paper_id not in self.search_engine.papers_db:
            return {'error': 'Paper not found'}

        paper = self.search_engine.papers_db[paper_id]

        # Find similar papers
        similar_papers = await self.search_engine.find_similar_papers(paper_id, top_k=5)

        # Calculate influence score
        influence_scores = self.citation_analyzer.calculate_influence_scores()
        paper_influence = influence_scores.get(paper_id, 0.0)

        return {
            'paper_id': paper_id,
            'title': paper.title,
            'influence_score': paper_influence,
            'similar_papers': [
                {
                    'paper_id': similar.id,
                    'title': similar.title,
                    'similarity_score': score,
                    'authors': similar.authors
                }
                for similar, score in similar_papers
            ]
        }

    async def get_system_stats(self) -> Dict[str, Any]:
        """Získá statistiky semantic systému"""
        return {
            'total_papers': len(self.search_engine.papers_db),
            'citation_network_size': len(self.citation_analyzer.citation_graph.nodes()),
            'total_citations': len(self.citation_analyzer.citation_graph.edges()),
            'indexed_embeddings': self.search_engine.index.ntotal if self.search_engine.index else 0,
            'tracking_topics': len(self.trend_predictor.topic_evolution),
            'model_type': self.search_engine.model_name.value
        }

# Factory funkce
async def create_semantic_system() -> UnifiedSemanticSystem:
    """Factory pro vytvoření semantic systému"""
    system = UnifiedSemanticSystem()
    logger.info("Semantic Search System initialized")
    return system
