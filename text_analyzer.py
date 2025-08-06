"""
Analyzátory pro zpracování textu - nyní používá centrální TextProcessor
"""
import asyncio
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime
from text_processing_utils import text_processor

class TextAnalyzer:
    """Analyzátor textu používající centrální TextProcessor"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def extract_keywords(self, text: str, max_keywords: int = 20) -> List[Tuple[str, int]]:
        """Extrakce klíčových slov pomocí centrálního procesoru"""
        return text_processor.extract_keywords(text, max_keywords)

    async def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analýza sentimentu pomocí centrálního procesoru"""
        return text_processor.analyze_sentiment_basic(text)

    async def summarize_text(self, text: str, max_sentences: int = 3) -> str:
        """Sumarizace textu"""
        return text_processor.prioritize_by_keywords(text, max_sentences=max_sentences)

    def extract_health_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """Extrakce zdravotních klíčových slov"""
        keywords = text_processor.extract_keywords(text, max_keywords)
        health_related = [kw[0] for kw in keywords if any(med_kw in kw[0].lower() for med_kw in text_processor.medical_keywords)]
        return health_related

    def analyze_health_sentiment(self, text: str) -> Dict[str, float]:
        """Analýza sentimentu pro zdravotní témata"""
        return text_processor.analyze_sentiment_basic(text)

    def analyze_medical_text(self, text: str, medical_keywords: List[str] = None) -> Dict[str, Any]:
        """Analýza textu s prioritizací lékařských klíčových slov"""
        if medical_keywords is None:
            medical_keywords = text_processor.medical_keywords

        prioritized_text = text_processor.prioritize_by_keywords(text, medical_keywords, 5)
        sentiment = text_processor.analyze_sentiment_basic(prioritized_text)
        keywords = text_processor.extract_keywords(prioritized_text, 10)

        return {
            "prioritized_text": prioritized_text,
            "sentiment": sentiment,
            "keywords": keywords
        }

class DataAnalyzer:
    """Analýza dat s použitím centrálního TextProcessor"""

    def __init__(self):
        self.text_analyzer = TextAnalyzer()
        self.logger = logging.getLogger(__name__)

    async def analyze_research_data(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Komplexní analýza research dat"""
        try:
            analysis_results = {
                'overview': await self._create_overview(data),
                'sentiment_analysis': await self._analyze_overall_sentiment(data),
                'key_themes': await self._extract_key_themes(data),
                'source_analysis': await self._analyze_sources(data),
                'quality_metrics': await self._calculate_quality_metrics(data)
            }
            return analysis_results
        except Exception as e:
            self.logger.error(f"Chyba při analýze research dat: {e}")
            return {}

    async def _create_overview(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Vytvoření přehledu dat"""
        total_sources = sum(len(articles) for articles in data.values())
        source_counts = {source: len(articles) for source, articles in data.items()}

        # Počítání typů zdrojů
        academic_count = 0
        news_count = 0
        web_count = 0
        
        for source, articles in data.items():
            for article in articles:
                article_type = article.get('type', 'web')
                if article_type == 'academic_paper':
                    academic_count += 1
                elif 'news' in article.get('source', '').lower():
                    news_count += 1
                else:
                    web_count += 1

        return {
            'total_sources': total_sources,
            'source_breakdown': source_counts,
            'content_types': {
                'academic': academic_count,
                'news': news_count,
                'web': web_count
            }
        }

    async def _analyze_overall_sentiment(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analýza celkového sentimentu"""
        all_sentiments = []

        for source, articles in data.items():
            for article in articles:
                text_content = f"{article.get('title', '')} {article.get('content', '')} {article.get('abstract', '')}"
                sentiment = await self.text_analyzer.analyze_sentiment(text_content)
                all_sentiments.append(sentiment)

        if not all_sentiments:
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

        # Průměrné hodnoty
        avg_sentiment = {
            'positive': sum(s['positive'] for s in all_sentiments) / len(all_sentiments),
            'neutral': sum(s['neutral'] for s in all_sentiments) / len(all_sentiments),
            'negative': sum(s['negative'] for s in all_sentiments) / len(all_sentiments)
        }

        return avg_sentiment

    async def _extract_key_themes(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Extrakce klíčových témat"""
        all_text = []

        for source, articles in data.items():
            for article in articles:
                text_content = f"{article.get('title', '')} {article.get('content', '')} {article.get('abstract', '')}"
                all_text.append(text_content)

        combined_text = ' '.join(all_text)
        keywords = await self.text_analyzer.extract_keywords(combined_text, max_keywords=30)

        # Seskupení klíčových slov do témat
        themes = self._group_keywords_into_themes(keywords)

        return {
            'top_keywords': keywords[:15],
            'themes': themes
        }

    def _group_keywords_into_themes(self, keywords: List[Tuple[str, float]]) -> Dict[str, List[str]]:
        """Seskupení klíčových slov do témat"""
        # Předem definované kategorie
        tech_words = {'ai', 'artificial', 'intelligence', 'machine', 'learning', 'algorithm', 'data', 'computer', 'software', 'technology'}
        business_words = {'business', 'company', 'market', 'industry', 'economic', 'financial', 'investment', 'revenue', 'profit'}
        research_words = {'research', 'study', 'analysis', 'method', 'experiment', 'result', 'finding', 'conclusion'}

        themes = {
            'Technology': [],
            'Business': [],
            'Research': [],
            'Other': []
        }
        
        for keyword, score in keywords:
            word_lower = keyword.lower()

            if any(tech_word in word_lower for tech_word in tech_words):
                themes['Technology'].append(keyword)
            elif any(business_word in word_lower for business_word in business_words):
                themes['Business'].append(keyword)
            elif any(research_word in word_lower for research_word in research_words):
                themes['Research'].append(keyword)
            else:
                themes['Other'].append(keyword)

        return themes

    async def _analyze_sources(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analýza zdrojů"""
        source_quality = {}

        for source_name, articles in data.items():
            if not articles:
                continue

            # Metriky kvality
            avg_length = sum(len(article.get('content', '') + article.get('abstract', '')) for article in articles) / len(articles)

            # Počet autorů (pro akademické články)
            author_counts = []
            for article in articles:
                authors = article.get('authors', [])
                if authors:
                    author_counts.append(len(authors))

            avg_authors = sum(author_counts) / len(author_counts) if author_counts else 0

            # Citace (pokud dostupné)
            citation_counts = []
            for article in articles:
                citations = article.get('citation_count', 0)
                if citations:
                    citation_counts.append(citations)

            avg_citations = sum(citation_counts) / len(citation_counts) if citation_counts else 0

            source_quality[source_name] = {
                'article_count': len(articles),
                'avg_content_length': avg_length,
                'avg_authors': avg_authors,
                'avg_citations': avg_citations
            }

        return source_quality


    async def _calculate_quality_metrics(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Výpočet metrik kvality"""
        total_articles = sum(len(articles) for articles in data.values())

        if total_articles == 0:
            return {'overall_quality': 0.0}

        # Metriky
        articles_with_abstracts = 0
        articles_with_authors = 0
        articles_with_dates = 0
        total_content_length = 0

        for source, articles in data.items():
            for article in articles:
                if article.get('abstract') or article.get('content'):
                    content_len = len(article.get('abstract', '') + article.get('content', ''))
                    if content_len > 100:  # Minimální délka pro kvalitní obsah
                        articles_with_abstracts += 1
                    total_content_length += content_len

                if article.get('authors'):
                    articles_with_authors += 1

                date_fields = ['published_date', 'publication_date', 'date', 'published_at']
                if any(article.get(field) for field in date_fields):
                    articles_with_dates += 1

        # Výpočet skóre kvality
        completeness_score = (
            (articles_with_abstracts / total_articles) * 0.4 +
            (articles_with_authors / total_articles) * 0.3 +
            (articles_with_dates / total_articles) * 0.3
        )

        avg_content_length = total_content_length / total_articles if total_articles > 0 else 0
        length_score = min(avg_content_length / 1000, 1.0)  # Normalizace na 1000 znaků

        overall_quality = (completeness_score * 0.7 + length_score * 0.3)

        return {
            'overall_quality': overall_quality,
            'completeness_score': completeness_score,
            'avg_content_length': avg_content_length,
            'articles_with_metadata': {
                'abstracts': articles_with_abstracts,
                'authors': articles_with_authors,
                'dates': articles_with_dates
            }
        }
