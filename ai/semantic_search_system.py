"""
Semantic Search System - Pokročilé sémantické vyhledávání
Využívá sentence-transformers a chromadb pro inteligentní vyhledávání
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib
from datetime import datetime

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SEMANTIC_DEPS_AVAILABLE = True
except ImportError as e:
    SEMANTIC_DEPS_AVAILABLE = False
    print(f"⚠️ Semantic search dependencies not available: {e}")

from unified_config import get_config

logger = logging.getLogger(__name__)

class SemanticSearchSystem:
    """
    Pokročilý sémantický vyhledávač pro biohacking výzkum
    Používá lokální embeddings pro rychlé a přesné vyhledávání
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.chroma_client = None
        self.collection = None
        self.collection_name = "biohack_research"

        # Konfigurace
        config = get_config()
        self.data_dir = Path("./chroma_data")
        self.data_dir.mkdir(exist_ok=True)

    async def __aenter__(self):
        """Inicializace komponent"""
        if not SEMANTIC_DEPS_AVAILABLE:
            logger.warning("⚠️ Semantic search unavailable - missing dependencies")
            return self

        try:
            # Načtení sentence transformer modelu
            logger.info(f"📚 Loading semantic model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

            # Inicializace ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.data_dir)
            )

            # Získání nebo vytvoření kolekce
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"✅ Using existing collection: {self.collection_name}")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Biohacking research semantic index"}
                )
                logger.info(f"✅ Created new collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"❌ Semantic search initialization failed: {e}")
            self.model = None
            self.collection = None

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup"""
        # ChromaDB se samo uklidí
        pass

    def _generate_doc_id(self, content: str, source: str = "") -> str:
        """Generuje konzistentní ID pro dokumenty"""
        content_hash = hashlib.md5(
            f"{content}{source}".encode()
        ).hexdigest()
        return f"doc_{content_hash[:16]}"

    async def add_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        source: str = "unknown"
    ) -> bool:
        """Přidá dokument do sémantického indexu"""
        if not self.model or not self.collection:
            return False

        try:
            # Generování embeddings
            embedding = await asyncio.get_event_loop().run_in_executor(
                None, self.model.encode, content
            )

            doc_id = self._generate_doc_id(content, source)

            # Přidání metadat
            full_metadata = {
                **metadata,
                "source": source,
                "indexed_at": datetime.now().isoformat(),
                "content_length": len(content)
            }

            # Uložení do ChromaDB
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[content],
                metadatas=[full_metadata],
                ids=[doc_id]
            )

            logger.debug(f"📚 Added document: {doc_id} from {source}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to add document: {e}")
            return False

    async def search(
        self,
        query: str,
        n_results: int = 10,
        source_filter: Optional[str] = None,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Sémantické vyhledávání s pokročilým filtrováním

        Args:
            query: Vyhledávací dotaz
            n_results: Max počet výsledků
            source_filter: Filtr podle zdroje
            min_similarity: Minimální podobnost (0-1)
        """
        if not self.model or not self.collection:
            logger.warning("⚠️ Semantic search not available")
            return []

        try:
            # Generování query embeddings
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                None, self.model.encode, query
            )

            # Příprava where podmínek
            where_clause = None
            if source_filter:
                where_clause = {"source": source_filter}

            # Vyhledávání v ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )

            # Zpracování výsledků
            formatted_results = []

            for i in range(len(results["documents"][0])):
                distance = results["distances"][0][i]
                similarity = 1 - distance  # ChromaDB používá distance, my chceme similarity

                if similarity >= min_similarity:
                    formatted_results.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity": similarity,
                        "distance": distance
                    })

            logger.info(f"🔍 Semantic search: {len(formatted_results)} results for '{query[:50]}...'")
            return formatted_results

        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return []

    async def search_peptides(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Specializované vyhledávání pro peptidy"""
        return await self.search(
            query=query,
            n_results=n_results,
            source_filter="peptide_research",
            min_similarity=0.4
        )

    async def search_safety_info(
        self,
        compound: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Vyhledávání bezpečnostních informací"""
        safety_query = f"{compound} safety side effects contraindications warnings"

        return await self.search(
            query=safety_query,
            n_results=n_results,
            source_filter="safety_database",
            min_similarity=0.5
        )

    async def add_research_batch(
        self,
        research_results: List[Dict[str, Any]],
        source: str = "research_batch"
    ) -> int:
        """Přidá batch research výsledků do indexu"""
        added_count = 0

        for result in research_results:
            content = result.get("content", "")
            metadata = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "authors": result.get("authors", []),
                "publication_date": result.get("date", ""),
                "research_type": result.get("type", "general")
            }

            if content and await self.add_document(content, metadata, source):
                added_count += 1

        logger.info(f"📚 Added {added_count}/{len(research_results)} documents to semantic index")
        return added_count

    async def get_similar_queries(
        self,
        query: str,
        n_results: int = 5
    ) -> List[str]:
        """Najde podobné dotazy pro doporučení"""
        results = await self.search(
            query=query,
            n_results=n_results * 2,  # Více výsledků pro filtrování
            min_similarity=0.6
        )

        # Extrakce unikátních dotazů z metadat
        similar_queries = []
        seen_queries = set()

        for result in results:
            metadata = result.get("metadata", {})
            original_query = metadata.get("original_query", "")

            if original_query and original_query not in seen_queries:
                similar_queries.append(original_query)
                seen_queries.add(original_query)

                if len(similar_queries) >= n_results:
                    break

        return similar_queries

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Statistiky sémantického indexu"""
        if not self.collection:
            return {"error": "Collection not available"}

        try:
            count = self.collection.count()

            return {
                "total_documents": count,
                "model_name": self.model_name,
                "collection_name": self.collection_name,
                "data_directory": str(self.data_dir),
                "semantic_search_available": SEMANTIC_DEPS_AVAILABLE
            }

        except Exception as e:
            return {"error": f"Failed to get stats: {e}"}

# Global instance
_semantic_search = None

async def get_semantic_search() -> SemanticSearchSystem:
    """Singleton pro sémantické vyhledávání"""
    global _semantic_search

    if _semantic_search is None:
        _semantic_search = SemanticSearchSystem()
        await _semantic_search.__aenter__()

    return _semantic_search

async def cleanup_semantic_search():
    """Cleanup pro aplikaci shutdown"""
    global _semantic_search
    if _semantic_search:
        await _semantic_search.__aexit__(None, None, None)
        _semantic_search = None
