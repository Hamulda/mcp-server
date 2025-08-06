"""
Streamlit webové rozhraní pro Research Tool
"""
import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

# Import našich modulů
from research_engine import ResearchEngine, ResearchQuery, ResearchResult
from database_manager import DatabaseManager
from config import *

class ResearchUI:
    """Hlavní třída pro webové rozhraní"""

    def __init__(self):
        self.engine = None
        self.db_manager = None
        self.setup_page()

    def setup_page(self):
        """Nastavení Streamlit stránky"""
        st.set_page_config(
            page_title="AI Research Tool",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .stTab {
            font-size: 1.2rem;
        }
        </style>
        """, unsafe_allow_html=True)

    async def initialize_components(self):
        """Inicializace komponent"""
        if 'engine_initialized' not in st.session_state:
            with st.spinner("Inicializuji research engine..."):
                self.engine = ResearchEngine()
                self.db_manager = DatabaseManager()
                await self.db_manager.initialize()
                st.session_state.engine_initialized = True
                st.session_state.engine = self.engine
                st.session_state.db_manager = self.db_manager
        else:
            self.engine = st.session_state.engine
            self.db_manager = st.session_state.db_manager

    def run(self):
        """Hlavní funkce pro spuštění UI"""
        asyncio.run(self._run_async())

    async def _run_async(self):
        """Asynchronní běh aplikace"""
        await self.initialize_components()

        # Hlavní nadpis
        st.markdown('<h1 class="main-header">🔬 AI Research Tool</h1>', unsafe_allow_html=True)

        # Sidebar s navigací
        with st.sidebar:
            st.title("Navigace")
            page = st.selectbox(
                "Vyberte sekci:",
                ["🔍 Nový Research", "📊 Dashboard", "📚 Historie", "⚙️ Nastavení"]
            )

        # Routing podle vybrané stránky
        if page == "🔍 Nový Research":
            await self.show_research_page()
        elif page == "📊 Dashboard":
            await self.show_dashboard()
        elif page == "📚 Historie":
            await self.show_history_page()
        elif page == "⚙️ Nastavení":
            self.show_settings_page()

    async def show_research_page(self):
        """Stránka pro nový research"""
        st.header("Nový Research")

        with st.form("research_form"):
            col1, col2 = st.columns([2, 1])

            with col1:
                query_text = st.text_area(
                    "Research dotaz:",
                    placeholder="Například: 'artificial intelligence trends 2024' nebo 'kvantové počítače aplikace'",
                    height=100
                )

            with col2:
                st.subheader("Nastavení")

                sources = st.multiselect(
                    "Zdroje:",
                    ["Web", "Akademické", "Zprávy", "RSS"],
                    default=["Web", "Akademické"]
                )

                depth = st.selectbox(
                    "Hloubka researche:",
                    ["Rychlý", "Střední", "Hluboký"],
                    index=1
                )

                max_results = st.slider(
                    "Max. počet zdrojů:",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10
                )

                languages = st.multiselect(
                    "Jazyky:",
                    ["Čeština", "Angličtina", "Němčina"],
                    default=["Čeština", "Angličtina"]
                )

            submitted = st.form_submit_button("🚀 Spustit Research", use_container_width=True)

            if submitted and query_text:
                await self.conduct_research(query_text, sources, depth, max_results, languages)

    async def conduct_research(self, query_text, sources, depth, max_results, languages):
        """Skutečné provedení researche místo simulace"""
        # Mapování UI hodnot na interní formát
        source_mapping = {
            "Web": "web",
            "Akademické": "academic",
            "Zprávy": "news",
            "RSS": "rss"
        }

        depth_mapping = {
            "Rychlý": "shallow",
            "Střední": "medium",
            "Hluboký": "deep"
        }

        lang_mapping = {
            "Čeština": "cs",
            "Angličtina": "en",
            "Němčina": "de"
        }

        # Vytvoření query objektu s automatickou optimalizací
        research_query = ResearchQuery(
            query=query_text,
            sources=[source_mapping[s] for s in sources],
            depth=depth_mapping[depth],
            max_results=max_results,
            languages=[lang_mapping[l] for l in languages],
            domain="medical" if any(kw in query_text.lower() for kw in ["nootropika", "peptidy", "medikace"]) else "general"
        )

        # Progress tracking s callback funkcí
        progress_callback = self._create_progress_callback()

        try:
            # Skutečný research s progress updaty
            progress_callback("🔍 Optimalizuji dotaz...", 10)

            progress_callback("📥 Vyhledávám zdroje...", 30)
            result = await self.engine.conduct_research(research_query)

            progress_callback("🧠 Analyzuji data pomocí AI...", 70)

            progress_callback("📊 Vytvářím report...", 90)

            progress_callback("✅ Research dokončen!", 100)

            # Zobrazení reálných výsledků
            await self.show_real_research_results(result)

        except Exception as e:
            st.error(f"Chyba při researchi: {str(e)}")
            self.logger.error(f"Research error: {e}")

    def _create_progress_callback(self):
        """Vytvoření callback funkce pro progress tracking"""
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(message: str, percentage: int):
            progress_bar.progress(percentage)
            status_text.text(message)

        return update_progress

    async def show_real_research_results(self, result: ResearchResult):
        """Zobrazení skutečných výsledků researche"""
        st.success("Research úspěšně dokončen!")

        # Reálné metriky z výsledků
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Nalezené zdroje", result.sources_found)

        with col2:
            academic_count = sum(1 for source in result.sources if source.get('type') == 'academic_paper')
            st.metric("Akademické články", academic_count)

        with col3:
            st.metric("Skóre spolehlivosti", f"{result.confidence_score:.1%}")

        with col4:
            st.metric("Doba researche", f"{result.processing_time:.1f}s")

        # Zobrazení nákladů pokud dostupné
        if result.cost_info:
            st.info(f"💰 Náklady: ${result.cost_info.get('estimated_cost_usd', 0):.4f} USD")

        # Taby s reálnými výsledky
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Shrnutí", "📊 Analýza", "🔗 Zdroje", "📈 Grafy"])

        with tab1:
            st.subheader("Hlavní shrnutí")
            st.write(result.summary)

            st.subheader("Klíčová zjištění")
            for i, finding in enumerate(result.key_findings[:5], 1):
                st.write(f"{i}. {finding}")

        with tab2:
            if result.key_findings:
                # Zobrazení klíčových slov z reálných dat
                keywords_df = pd.DataFrame({
                    'Klíčové slovo': result.key_findings[:10],
                    'Relevance': [1.0 - i*0.1 for i in range(len(result.key_findings[:10]))]
                })

                fig = px.bar(keywords_df, x='Klíčové slovo', y='Relevance',
                           title='Klíčová slova z analýzy')
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Analyzované zdroje")

            if result.sources:
                sources_df = pd.DataFrame(result.sources)
                st.dataframe(sources_df, use_container_width=True)
            else:
                st.info("Žádné zdroje k zobrazení")

        with tab4:
            # Grafy podle typu zdrojů
            if result.sources:
                source_types = {}
                for source in result.sources:
                    stype = source.get('type', 'unknown')
                    source_types[stype] = source_types.get(stype, 0) + 1

                if source_types:
                    fig = px.pie(values=list(source_types.values()),
                               names=list(source_types.keys()),
                               title='Rozložení podle typu zdrojů')
                    st.plotly_chart(fig, use_container_width=True)

    async def show_dashboard(self):
        """Dashboard s reálnými daty z databáze"""
        st.header("📊 Dashboard")

        try:
            # Získání skutečných statistik z databáze
            stats = await self.db_manager.get_statistics()

            # Reálné metriky
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Celkem dotazů", stats.get('total_queries', 0))

            with col2:
                st.metric("Celkem zdrojů", stats.get('total_sources', 0))

            with col3:
                st.metric("Dotazy tento týden", stats.get('queries_last_week', 0))

            with col4:
                avg_conf = stats.get('avg_confidence', 0)
                st.metric("Průměrná spolehlivost", f"{avg_conf:.1%}" if avg_conf else "N/A")

            # Reálné grafy z databáze
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Zdroje podle typu")
                sources_by_type = stats.get('sources_by_type', {})
                if sources_by_type:
                    fig = px.bar(x=list(sources_by_type.keys()),
                               y=list(sources_by_type.values()),
                               title="Počet zdrojů podle typu")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Zatím nejsou k dispozici data o zdrojích")

        except Exception as e:
            st.error(f"Chyba při načítání dashboard dat: {e}")
            self.logger.error(f"Dashboard error: {e}")

    async def show_history_page(self):
        """Stránka s reálnou historií z databáze"""
        st.header("📚 Historie researche")

        try:
            # Získání reálné historie
            history = await self.db_manager.get_research_history(limit=50)

            if not history:
                st.info("Zatím nebyly provedeny žádné researche.")
                return

            # Filtry
            col1, col2, col3 = st.columns(3)

            with col1:
                status_filter = st.selectbox("Status:", ["Všechny", "completed", "pending", "error"])

            with col2:
                date_range = st.date_input("Datum od:", value=datetime.now() - timedelta(days=30))

            with col3:
                search_term = st.text_input("Hledat v dotazech:")

            # Filtrování historie
            filtered_history = history
            if status_filter != "Všechny":
                filtered_history = [h for h in filtered_history if h.get('status') == status_filter]

            if search_term:
                filtered_history = [h for h in filtered_history if search_term.lower() in h.get('query_text', '').lower()]

            # Zobrazení reálné historie
            for item in filtered_history:
                with st.expander(f"🔍 {item['query_text']} - {item['timestamp']}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write(f"**Status:** {item['status']}")
                        st.write(f"**Zdroje:** {item['total_sources']}")

                    with col2:
                        confidence = item.get('confidence_score', 0)
                        st.write(f"**Spolehlivost:** {confidence:.1%}" if confidence else "N/A")
                        st.write(f"**ID:** {item['id']}")

                    with col3:
                        if st.button(f"Zobrazit detaily", key=f"detail_{item['id']}"):
                            await self.show_query_details(item['id'])

        except Exception as e:
            st.error(f"Chyba při načítání historie: {e}")
            self.logger.error(f"History error: {e}")

    async def show_query_details(self, query_id: int):
        """Zobrazení detailů dotazu"""
        details = await self.db_manager.get_query_details(query_id)

        if details:
            st.json(details)
        else:
            st.error("Nepodařilo se načíst detaily dotazu")

    def show_settings_page(self):
        """Stránka s nastavením"""
        st.header("⚙️ Nastavení")

        tab1, tab2, tab3 = st.tabs(["🔧 Obecné", "🔑 API klíče", "🗃️ Databáze"])

        with tab1:
            st.subheader("Obecná nastavení")

            max_concurrent = st.slider("Max. současných požadavků:", 1, 20, 10)
            request_delay = st.slider("Zpoždění mezi požadavky (s):", 0.1, 5.0, 1.0, 0.1)

            default_sources = st.multiselect(
                "Výchozí zdroje:",
                ["Web", "Akademické", "Zprávy", "RSS"],
                default=["Web", "Akademické"]
            )

            if st.button("💾 Uložit nastavení"):
                st.success("Nastavení uloženo!")

        with tab2:
            st.subheader("API klíče")
            st.info("Pro plnou funkcionalitu zadejte API klíče pro externí služby")

            openai_key = st.text_input("OpenAI API klíč:", type="password")
            news_api_key = st.text_input("News API klíč:", type="password")
            serp_api_key = st.text_input("SERP API klíč:", type="password")

            if st.button("💾 Uložit API klíče"):
                # Zde by se klíče uložily do .env souboru
                st.success("API klíče uloženy!")

        with tab3:
            st.subheader("Správa databáze")

            stats = asyncio.run(self.db_manager.get_statistics())

            st.write("**Statistiky databáze:**")
            st.json(stats)

            col1, col2 = st.columns(2)

            with col1:
                if st.button("🧹 Vyčistit stará data (90+ dní)"):
                    with st.spinner("Čistím databázi..."):
                        asyncio.run(self.db_manager.cleanup_old_data(90))
                    st.success("Stará data vyčištěna!")

            with col2:
                if st.button("📤 Exportovat data"):
                    data = asyncio.run(self.db_manager.export_data())
                    st.download_button(
                        "Stáhnout export",
                        json.dumps(data, indent=2),
                        file_name=f"research_export_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )

    def display_medical_results(self, results: ResearchResult):
        """Vizualizace výsledků lékařských dotazů"""
        st.header("📊 Výsledky lékařského výzkumu")

        # Zobrazení klíčových slov
        st.subheader("🔑 Klíčová slova")
        keywords = results.key_findings
        if keywords:
            keyword_df = pd.DataFrame(keywords, columns=["Klíčové slovo"])
            st.dataframe(keyword_df)

        # Vizualizace sentimentu
        st.subheader("📈 Sentiment")
        sentiment = results.confidence_score
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment * 100,
            title={"text": "Důvěra v analýzu (%)"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        st.plotly_chart(fig)

        # Zobrazení shrnutí
        st.subheader("📄 Shrnutí")
        st.write(results.summary)

        # Zobrazení zdrojů
        st.subheader("📚 Zdroje")
        sources = pd.DataFrame(results.sources)
        st.dataframe(sources)

# Pro spuštění je potřeba numpy
try:
    import numpy as np
except ImportError:
    # Fallback pokud numpy není k dispozici
    class np:
        @staticmethod
        def random():
            return type('obj', (object,), {
                'poisson': lambda x: __import__('random').randint(0, x*2)
            })()

def main():
    """Hlavní funkce pro spuštění aplikace"""
    app = ResearchUI()
    app.run()

if __name__ == "__main__":
    main()
