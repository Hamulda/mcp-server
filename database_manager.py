"""
Databázový manager pro ukládání a správu research výsledků
"""
import sqlite3
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import aiosqlite

from config import DATABASE_PATH

class DatabaseManager:
    """Manager pro SQLite databázi s research výsledky"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DATABASE_PATH
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Inicializace databáze a tabulek"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await self._create_tables(db)
                await db.commit()
            self.logger.info("Databáze úspěšně inicializována")
        except Exception as e:
            self.logger.error(f"Chyba při inicializaci databáze: {e}")
            raise
    
    async def _create_tables(self, db):
        """Vytvoření tabulek"""
        
        # Hlavní tabulka pro research queries
        await db.execute("""
            CREATE TABLE IF NOT EXISTS research_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                query_params TEXT,  -- JSON s parametry
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                total_sources INTEGER DEFAULT 0,
                confidence_score REAL DEFAULT 0.0
            )
        """)
        
        # Tabulka pro zdroje/články
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER,
                source_type TEXT,  -- web, academic, news
                source_name TEXT,  -- arxiv, pubmed, etc.
                title TEXT,
                content TEXT,
                abstract TEXT,
                authors TEXT,  -- JSON array
                publication_date TEXT,
                url TEXT,
                metadata TEXT,  -- JSON s dalšími daty
                scraped_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES research_queries (id)
            )
        """)
        
        # Tabulka pro analýzy
        await db.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER,
                analysis_type TEXT,  -- sentiment, keywords, summary
                analysis_data TEXT,  -- JSON s výsledky
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES research_queries (id)
            )
        """)
        
        # Tabulka pro reporty
        await db.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER,
                report_type TEXT,  -- summary, detailed, analysis
                report_content TEXT,
                file_path TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES research_queries (id)
            )
        """)
        
        # Indexy pro rychlejší vyhledávání
        await db.execute("CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON research_queries(timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_sources_query_id ON sources(query_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(source_type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_analyses_query_id ON analyses(query_id)")
    
    async def save_research_query(self, query_text: str, query_params: Dict[str, Any]) -> int:
        """Uložení research dotazu"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "INSERT INTO research_queries (query_text, query_params) VALUES (?, ?)",
                    (query_text, json.dumps(query_params))
                )
                query_id = cursor.lastrowid
                await db.commit()
                return query_id
        except Exception as e:
            self.logger.error(f"Chyba při ukládání dotazu: {e}")
            raise
    
    async def save_sources(self, query_id: int, sources: List[Dict[str, Any]]):
        """Uložení zdrojů pro daný dotaz"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                for source in sources:
                    await db.execute("""
                        INSERT INTO sources (
                            query_id, source_type, source_name, title, content, 
                            abstract, authors, publication_date, url, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        query_id,
                        source.get('type', 'web'),
                        source.get('source', ''),
                        source.get('title', ''),
                        source.get('content', ''),
                        source.get('abstract', ''),
                        json.dumps(source.get('authors', [])),
                        source.get('publication_date', ''),
                        source.get('url', ''),
                        json.dumps({k: v for k, v in source.items() 
                                  if k not in ['type', 'source', 'title', 'content', 'abstract', 'authors', 'publication_date', 'url']})
                    ))
                await db.commit()
        except Exception as e:
            self.logger.error(f"Chyba při ukládání zdrojů: {e}")
            raise
    
    async def save_analysis(self, query_id: int, analysis_type: str, analysis_data: Dict[str, Any]):
        """Uložení výsledků analýzy"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO analyses (query_id, analysis_type, analysis_data) VALUES (?, ?, ?)",
                    (query_id, analysis_type, json.dumps(analysis_data))
                )
                await db.commit()
        except Exception as e:
            self.logger.error(f"Chyba při ukládání analýzy: {e}")
            raise
    
    async def save_report(self, query_id: int, report_type: str, report_content: str, file_path: str = None):
        """Uložení reportu"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO reports (query_id, report_type, report_content, file_path) VALUES (?, ?, ?, ?)",
                    (query_id, report_type, report_content, file_path)
                )
                await db.commit()
        except Exception as e:
            self.logger.error(f"Chyba při ukládání reportu: {e}")
            raise
    
    async def update_query_status(self, query_id: int, status: str, total_sources: int = None, confidence_score: float = None):
        """Aktualizace stavu dotazu"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                updates = ["status = ?"]
                params = [status]
                
                if total_sources is not None:
                    updates.append("total_sources = ?")
                    params.append(total_sources)
                
                if confidence_score is not None:
                    updates.append("confidence_score = ?")
                    params.append(confidence_score)
                
                params.append(query_id)
                
                await db.execute(
                    f"UPDATE research_queries SET {', '.join(updates)} WHERE id = ?",
                    params
                )
                await db.commit()
        except Exception as e:
            self.logger.error(f"Chyba při aktualizaci stavu dotazu: {e}")
            raise
    
    async def get_research_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Získání historie research dotazů"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                cursor = await db.execute("""
                    SELECT id, query_text, timestamp, status, total_sources, confidence_score
                    FROM research_queries 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Chyba při získávání historie: {e}")
            return []
    
    async def get_query_details(self, query_id: int) -> Optional[Dict[str, Any]]:
        """Získání detailů konkrétního dotazu"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Základní info o dotazu
                cursor = await db.execute(
                    "SELECT * FROM research_queries WHERE id = ?", (query_id,)
                )
                query_row = await cursor.fetchone()
                
                if not query_row:
                    return None
                
                query_data = dict(query_row)
                
                # Zdroje
                cursor = await db.execute(
                    "SELECT * FROM sources WHERE query_id = ?", (query_id,)
                )
                sources = [dict(row) for row in await cursor.fetchall()]
                
                # Analýzy
                cursor = await db.execute(
                    "SELECT * FROM analyses WHERE query_id = ?", (query_id,)
                )
                analyses = [dict(row) for row in await cursor.fetchall()]
                
                # Reporty
                cursor = await db.execute(
                    "SELECT * FROM reports WHERE query_id = ?", (query_id,)
                )
                reports = [dict(row) for row in await cursor.fetchall()]
                
                return {
                    'query': query_data,
                    'sources': sources,
                    'analyses': analyses,
                    'reports': reports
                }
                
        except Exception as e:
            self.logger.error(f"Chyba při získávání detailů dotazu: {e}")
            return None
    
    async def search_sources(self, search_term: str, source_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Vyhledávání ve zdrojích"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                query = """
                    SELECT s.*, rq.query_text 
                    FROM sources s
                    JOIN research_queries rq ON s.query_id = rq.id
                    WHERE (s.title LIKE ? OR s.content LIKE ? OR s.abstract LIKE ?)
                """
                params = [f"%{search_term}%"] * 3
                
                if source_type:
                    query += " AND s.source_type = ?"
                    params.append(source_type)
                
                query += " ORDER BY s.scraped_at DESC LIMIT ?"
                params.append(limit)
                
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Chyba při vyhledávání zdrojů: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Získání statistik databáze"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                stats = {}
                
                # Celkový počet dotazů
                cursor = await db.execute("SELECT COUNT(*) FROM research_queries")
                stats['total_queries'] = (await cursor.fetchone())[0]
                
                # Celkový počet zdrojů
                cursor = await db.execute("SELECT COUNT(*) FROM sources")
                stats['total_sources'] = (await cursor.fetchone())[0]
                
                # Statistiky podle typu zdroje
                cursor = await db.execute("""
                    SELECT source_type, COUNT(*) 
                    FROM sources 
                    GROUP BY source_type
                """)
                stats['sources_by_type'] = dict(await cursor.fetchall())
                
                # Nejnovější dotazy
                cursor = await db.execute("""
                    SELECT COUNT(*) 
                    FROM research_queries 
                    WHERE timestamp > datetime('now', '-7 days')
                """)
                stats['queries_last_week'] = (await cursor.fetchone())[0]
                
                # Průměrné skóre spolehlivosti
                cursor = await db.execute("""
                    SELECT AVG(confidence_score) 
                    FROM research_queries 
                    WHERE confidence_score > 0
                """)
                avg_confidence = await cursor.fetchone()
                stats['avg_confidence'] = avg_confidence[0] if avg_confidence[0] else 0
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Chyba při získávání statistik: {e}")
            return {}
    
    async def cleanup_old_data(self, days_old: int = 90):
        """Vyčištění starých dat"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Najít staré dotazy
                cursor = await db.execute("""
                    SELECT id FROM research_queries 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(days_old))
                
                old_query_ids = [row[0] for row in await cursor.fetchall()]
                
                if old_query_ids:
                    # Smazat související data
                    placeholders = ','.join('?' * len(old_query_ids))
                    
                    await db.execute(f"DELETE FROM reports WHERE query_id IN ({placeholders})", old_query_ids)
                    await db.execute(f"DELETE FROM analyses WHERE query_id IN ({placeholders})", old_query_ids)
                    await db.execute(f"DELETE FROM sources WHERE query_id IN ({placeholders})", old_query_ids)
                    await db.execute(f"DELETE FROM research_queries WHERE id IN ({placeholders})", old_query_ids)
                    
                    await db.commit()
                    
                    self.logger.info(f"Vymazáno {len(old_query_ids)} starých dotazů")
                
        except Exception as e:
            self.logger.error(f"Chyba při vyčišťování dat: {e}")
            raise
    
    async def export_data(self, query_id: int = None, format: str = 'json') -> Dict[str, Any]:
        """Export dat z databáze"""
        try:
            if query_id:
                # Export konkrétního dotazu
                data = await self.get_query_details(query_id)
            else:
                # Export všech dat
                queries = await self.get_research_history(limit=1000)
                data = {'queries': queries}
            
            if format == 'json':
                return data
            else:
                # Zde by mohly být další formáty (CSV, Excel, atd.)
                return data
                
        except Exception as e:
            self.logger.error(f"Chyba při exportu dat: {e}")
            return {}
    
    async def save_medical_results(self, query_id: int, keywords: List[str], sentiment: float, summary: str):
        """Uložení lékařských výsledků do databáze"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    UPDATE research_queries
                    SET key_findings = ?, confidence_score = ?, summary = ?
                    WHERE id = ?
                    """,
                    (json.dumps(keywords), sentiment, summary, query_id)
                )
                await db.commit()
            self.logger.info(f"Lékařské výsledky úspěšně uloženy pro query ID {query_id}")
        except Exception as e:
            self.logger.error(f"Chyba při ukládání lékařských výsledků: {e}")
            raise
