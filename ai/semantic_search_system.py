"""
Semantic Search Capabilities - Enhanced Version
Implementuje vector embeddings pro research papers s pokročilými funkcemi
- Citation network analysis
- Research trend prediction
- Semantic similarity search
- Research collaboration tools
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
import chromadb
from chromadb.config import Settings

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
    research_field: Optional[str] = None
    impact_score: float = 0.0

@dataclass
class CitationNetwork:
    """Síť citací pro analýzu"""
    papers: Dict[str, ResearchPaper]
    citation_graph: nx.DiGraph

@dataclass
class ResearchTrend:
    """Research trend data"""
    topic: str
    keywords: List[str]
    paper_count: int
    growth_rate: float
    confidence: float
    related_fields: List[str]

class EnhancedSemanticSearchEngine:
    """Pokročilý semantic search engine pro academic research"""

    def __init__(self, model_name: EmbeddingModel = EmbeddingModel.SCIENTIFIC_PAPERS):
        self.model_name = model_name
        self.embedding_model = None
        self.vector_db = None
        self.citation_network = CitationNetwork({}, nx.DiGraph())
        self.research_trends = {}
        self.collaboration_graph = nx.Graph()

        # ChromaDB pro persistent storage
        try:
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_data",
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name="research_papers",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.warning(f"ChromaDB nedostupná: {e}")
            self.chroma_client = None
            self.collection = None

        # FAISS index pro rychlé vyhledávání
        self.faiss_index = None
        self.paper_id_to_index = {}

        # Cache pro embeddings
        self.embedding_cache = {}

    async def initialize(self):
        """Inicializace modelu a databází"""
        try:
            model_map = {
                EmbeddingModel.SCIENTIFIC_PAPERS: "allenai-specter",
                EmbeddingModel.GENERAL_PURPOSE: "all-MiniLM-L6-v2",
                EmbeddingModel.MULTILINGUAL: "paraphrase-multilingual-MiniLM-L12-v2"
            }

            self.embedding_model = SentenceTransformer(model_map[self.model_name])
            logger.info(f"✅ Semantic search engine inicializován s {self.model_name.value}")

        except Exception as e:
            logger.error(f"❌ Chyba při inicializaci embedding modelu: {e}")
            # Fallback na jednodušší model
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("✅ Fallback na general purpose model")
            except Exception as fallback_e:
                logger.error(f"❌ Fallback také selhal: {fallback_e}")

    async def add_paper(self, paper: ResearchPaper) -> bool:
        """Přidá paper do semantic search indexu"""
        try:
            # Generuj text pro embedding
            text_for_embedding = f"{paper.title} [SEP] {paper.abstract}"
            if paper.keywords:
                text_for_embedding += f" [SEP] {' '.join(paper.keywords)}"

            # Generuj embedding
            embedding = await self._get_embedding(text_for_embedding)
            paper.embedding = embedding

            # Uložení do ChromaDB
            if self.collection:
                metadata = {
                    "title": paper.title,
                    "authors": ",".join(paper.authors),
                    "year": paper.year,
                    "doi": paper.doi or "",
                    "keywords": ",".join(paper.keywords),
                    "research_field": paper.research_field or ""
                }

                self.collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[text_for_embedding],
                    metadatas=[metadata],
                    ids=[paper.id]
                )

            # Aktualizuj FAISS index
            await self._update_faiss_index(paper)

            # Aktualizuj citation network
            self.citation_network.papers[paper.id] = paper
            self._update_citation_graph(paper)

            return True

        except Exception as e:
            logger.error(f"Chyba při přidávání paperu {paper.id}: {e}")
            return False

    async def semantic_search(self, query: str, top_k: int = 10,
                            filters: Optional[Dict[str, Any]] = None) -> List[ResearchPaper]:
        """Semantic search napříč research papers"""
        try:
            # Generuj embedding pro query
            query_embedding = await self._get_embedding(query)

            results = []

            # ChromaDB search
            if self.collection:
                chroma_results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=min(top_k * 2, 100),  # Více výsledků pro filtrování
                    where=self._build_chroma_filters(filters) if filters else None
                )

                for i, paper_id in enumerate(chroma_results['ids'][0]):
                    if paper_id in self.citation_network.papers:
                        paper = self.citation_network.papers[paper_id]
                        paper.semantic_score = 1.0 - chroma_results['distances'][0][i]
                        results.append(paper)

            # FAISS search jako backup
            elif self.faiss_index and len(self.paper_id_to_index) > 0:
                distances, indices = self.faiss_index.search(
                    query_embedding.reshape(1, -1), min(top_k * 2, len(self.paper_id_to_index))
                )

                for dist, idx in zip(distances[0], indices[0]):
                    paper_id = list(self.paper_id_to_index.keys())[idx]
                    if paper_id in self.citation_network.papers:
                        paper = self.citation_network.papers[paper_id]
                        paper.semantic_score = 1.0 / (1.0 + dist)
                        results.append(paper)

            # Aplikuj filtry
            if filters:
                results = self._apply_filters(results, filters)

            # Seřaď podle semantic score a impact score
            results.sort(key=lambda x: (x.semantic_score * 0.7 + x.impact_score * 0.3), reverse=True)

            return results[:top_k]

        except Exception as e:
            logger.error(f"Chyba při semantic search: {e}")
            return []

    async def find_similar_papers(self, paper_id: str, top_k: int = 5) -> List[ResearchPaper]:
        """Najde podobné papers na základě embeddings"""
        if paper_id not in self.citation_network.papers:
            return []

        paper = self.citation_network.papers[paper_id]
        if paper.embedding is None:
            return []

        return await self.semantic_search(
            f"{paper.title} {paper.abstract}",
            top_k=top_k + 1  # +1 protože original paper bude v rezultech
        )[1:]  # Odstranit original paper

    async def analyze_citation_network(self, paper_ids: List[str]) -> Dict[str, Any]:
        """Analyzuje citation network pro zadané papers"""
        try:
            subgraph = self.citation_network.citation_graph.subgraph(paper_ids)

            analysis = {
                'network_metrics': {
                    'nodes': subgraph.number_of_nodes(),
                    'edges': subgraph.number_of_edges(),
                    'density': nx.density(subgraph),
                    'average_clustering': nx.average_clustering(subgraph)
                },
                'influential_papers': [],
                'citation_patterns': {},
                'research_clusters': []
            }

            # Najdi nejvlivnější papers
            if subgraph.number_of_nodes() > 0:
                centrality = nx.pagerank(subgraph)
                top_papers = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]

                for paper_id, score in top_papers:
                    if paper_id in self.citation_network.papers:
                        paper = self.citation_network.papers[paper_id]
                        analysis['influential_papers'].append({
                            'id': paper_id,
                            'title': paper.title,
                            'influence_score': score,
                            'citations_count': subgraph.in_degree(paper_id)
                        })

            # Detekuj research clusters
            if subgraph.number_of_nodes() > 3:
                try:
                    communities = nx.community.greedy_modularity_communities(
                        subgraph.to_undirected()
                    )

                    for i, community in enumerate(communities):
                        cluster_papers = []
                        for paper_id in community:
                            if paper_id in self.citation_network.papers:
                                paper = self.citation_network.papers[paper_id]
                                cluster_papers.append({
                                    'id': paper_id,
                                    'title': paper.title,
                                    'field': paper.research_field
                                })

                        if cluster_papers:
                            analysis['research_clusters'].append({
                                'cluster_id': i,
                                'size': len(cluster_papers),
                                'papers': cluster_papers
                            })

                except Exception as e:
                    logger.warning(f"Community detection selhala: {e}")

            return analysis

        except Exception as e:
            logger.error(f"Chyba při analýze citation network: {e}")
            return {}

    async def detect_research_trends(self, timeframe_years: int = 5) -> List[ResearchTrend]:
        """Detekuje research trendy na základě papers z posledních let"""
        try:
            current_year = datetime.now().year
            recent_papers = [
                paper for paper in self.citation_network.papers.values()
                if paper.year >= (current_year - timeframe_years)
            ]

            if not recent_papers:
                return []

            # Analýza klíčových slov a témat
            keyword_counts = defaultdict(int)
            field_counts = defaultdict(list)
            yearly_counts = defaultdict(lambda: defaultdict(int))

            for paper in recent_papers:
                for keyword in paper.keywords:
                    keyword_counts[keyword] += 1
                    yearly_counts[keyword][paper.year] += 1

                if paper.research_field:
                    field_counts[paper.research_field].append(paper)

            trends = []

            # Najdi rostoucí trendy
            for keyword, total_count in keyword_counts.items():
                if total_count < 3:  # Ignoruj vzácná klíčová slova
                    continue

                yearly_data = yearly_counts[keyword]
                years = sorted(yearly_data.keys())

                if len(years) >= 3:
                    # Spočítej growth rate
                    early_avg = sum(yearly_data[year] for year in years[:len(years)//2]) / (len(years)//2)
                    late_avg = sum(yearly_data[year] for year in years[len(years)//2:]) / (len(years) - len(years)//2)

                    if early_avg > 0:
                        growth_rate = (late_avg - early_avg) / early_avg

                        if growth_rate > 0.2:  # Alespoň 20% růst
                            # Najdi související oblasti
                            related_fields = []
                            for paper in recent_papers:
                                if keyword in paper.keywords and paper.research_field:
                                    if paper.research_field not in related_fields:
                                        related_fields.append(paper.research_field)

                            trend = ResearchTrend(
                                topic=keyword,
                                keywords=[keyword],
                                paper_count=total_count,
                                growth_rate=growth_rate,
                                confidence=min(total_count / 10.0, 1.0),
                                related_fields=related_fields[:5]
                            )
                            trends.append(trend)

            # Seřaď podle growth rate a confidence
            trends.sort(key=lambda x: x.growth_rate * x.confidence, reverse=True)

            return trends[:20]  # Top 20 trendů

        except Exception as e:
            logger.error(f"Chyba při detekci trendů: {e}")
            return []

    async def recommend_collaborations(self, researcher_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Doporučí potenciální spolupracovníky na základě research podobnosti"""
        try:
            # Najdi papers od researche
            researcher_papers = [
                paper for paper in self.citation_network.papers.values()
                if researcher_id in [author.lower() for author in paper.authors]
            ]

            if not researcher_papers:
                return []

            # Analýza research zájmů
            researcher_keywords = set()
            researcher_fields = set()

            for paper in researcher_papers:
                researcher_keywords.update(paper.keywords)
                if paper.research_field:
                    researcher_fields.add(paper.research_field)

            # Najdi podobné researche
            potential_collaborators = defaultdict(float)

            for paper in self.citation_network.papers.values():
                for author in paper.authors:
                    if author.lower() == researcher_id.lower():
                        continue

                    # Spočítej podobnost na základě klíčových slov
                    paper_keywords = set(paper.keywords)
                    keyword_overlap = len(researcher_keywords & paper_keywords)

                    if keyword_overlap > 0:
                        similarity = keyword_overlap / len(researcher_keywords | paper_keywords)
                        potential_collaborators[author] += similarity

                    # Bonus za stejnou research oblast
                    if paper.research_field in researcher_fields:
                        potential_collaborators[author] += 0.2

            # Seřaď a formatuj výsledky
            recommendations = []
            for author, score in sorted(potential_collaborators.items(),
                                      key=lambda x: x[1], reverse=True)[:top_k]:

                # Najdi papers od tohoto autora
                author_papers = [
                    paper for paper in self.citation_network.papers.values()
                    if author in paper.authors
                ]

                common_keywords = researcher_keywords & set(
                    kw for paper in author_papers for kw in paper.keywords
                )

                recommendations.append({
                    'researcher': author,
                    'similarity_score': score,
                    'common_keywords': list(common_keywords),
                    'paper_count': len(author_papers),
                    'recent_papers': [
                        {'title': p.title, 'year': p.year}
                        for p in author_papers[-3:]  # Poslední 3 papers
                    ]
                })

            return recommendations

        except Exception as e:
            logger.error(f"Chyba při doporučování spolupráce: {e}")
            return []

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Získá embedding pro text s cachováním"""
        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        if self.embedding_model is None:
            await self.initialize()

        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            self.embedding_cache[text_hash] = embedding

            # Omezí velikost cache
            if len(self.embedding_cache) > 10000:
                # Smaž nejstarší embeddings
                oldest_keys = list(self.embedding_cache.keys())[:1000]
                for key in oldest_keys:
                    del self.embedding_cache[key]

            return embedding

        except Exception as e:
            logger.error(f"Chyba při generování embedding: {e}")
            # Fallback na random embedding
            return np.random.random(384).astype(np.float32)

    async def _update_faiss_index(self, paper: ResearchPaper):
        """Aktualizuje FAISS index s novým paperem"""
        try:
            if paper.embedding is None:
                return

            embedding_dim = paper.embedding.shape[0]

            if self.faiss_index is None:
                self.faiss_index = faiss.IndexFlatIP(embedding_dim)

            # Normalizuj embedding pro cosine similarity
            normalized_embedding = paper.embedding / np.linalg.norm(paper.embedding)

            self.faiss_index.add(normalized_embedding.reshape(1, -1))
            self.paper_id_to_index[paper.id] = len(self.paper_id_to_index)

        except Exception as e:
            logger.error(f"Chyba při aktualizaci FAISS indexu: {e}")

    def _update_citation_graph(self, paper: ResearchPaper):
        """Aktualizuje citation graph"""
        try:
            self.citation_network.citation_graph.add_node(paper.id)

            for cited_paper_id in paper.citations:
                if cited_paper_id in self.citation_network.papers:
                    self.citation_network.citation_graph.add_edge(paper.id, cited_paper_id)

        except Exception as e:
            logger.error(f"Chyba při aktualizaci citation graph: {e}")

    def _build_chroma_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Vytvoří ChromaDB filtry"""
        chroma_filters = {}

        if 'year_min' in filters:
            chroma_filters['year'] = {'$gte': filters['year_min']}
        if 'year_max' in filters:
            if 'year' in chroma_filters:
                chroma_filters['year']['$lte'] = filters['year_max']
            else:
                chroma_filters['year'] = {'$lte': filters['year_max']}

        if 'research_field' in filters:
            chroma_filters['research_field'] = {'$eq': filters['research_field']}

        return chroma_filters

    def _apply_filters(self, papers: List[ResearchPaper], filters: Dict[str, Any]) -> List[ResearchPaper]:
        """Aplikuje filtry na seznam papers"""
        filtered = papers

        if 'year_min' in filters:
            filtered = [p for p in filtered if p.year >= filters['year_min']]
        if 'year_max' in filters:
            filtered = [p for p in filtered if p.year <= filters['year_max']]
        if 'research_field' in filters:
            filtered = [p for p in filtered if p.research_field == filters['research_field']]
        if 'authors' in filters:
            author_filter = filters['authors'].lower()
            filtered = [p for p in filtered if any(author_filter in author.lower() for author in p.authors)]

        return filtered

# Global instance
enhanced_semantic_search = EnhancedSemanticSearchEngine()
