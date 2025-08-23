"""
Academic Workflow Optimization - Phase 3
Implementuje pokročilé workflow pro academic research
- Research project templates
- Collaborative annotation tools
- Export integration (LaTeX, Reference managers)
- Academic citation management
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import re
import bibtex
from pathlib import Path

logger = logging.getLogger(__name__)

class ProjectType(Enum):
    """Typy research projektů"""
    LITERATURE_REVIEW = "literature_review"
    EXPERIMENTAL_STUDY = "experimental_study"
    THEORETICAL_ANALYSIS = "theoretical_analysis"
    META_ANALYSIS = "meta_analysis"
    CASE_STUDY = "case_study"
    SURVEY_RESEARCH = "survey_research"

class CitationStyle(Enum):
    """Styly citací"""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    VANCOUVER = "vancouver"
    HARVARD = "harvard"

class ExportFormat(Enum):
    """Export formáty"""
    BIBTEX = "bibtex"
    ENDNOTE = "endnote"
    MENDELEY = "mendeley"
    ZOTERO = "zotero"
    LATEX = "latex"
    WORD = "word"
    PDF = "pdf"

@dataclass
class ResearchProject:
    """Reprezentace research projektu"""
    id: str
    title: str
    project_type: ProjectType
    description: str
    created_date: datetime
    last_modified: datetime
    owner_id: str
    collaborators: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    status: str = "active"
    papers: List[str] = field(default_factory=list)  # Paper IDs
    annotations: Dict[str, Any] = field(default_factory=dict)
    bibliography: List[Dict] = field(default_factory=list)
    notes: List[Dict] = field(default_factory=list)
    milestones: List[Dict] = field(default_factory=list)

@dataclass
class Annotation:
    """Anotace k research paperu"""
    id: str
    paper_id: str
    user_id: str
    content: str
    annotation_type: str  # highlight, note, question, critique
    position: Optional[Dict] = None  # page, line, etc.
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    is_public: bool = False

@dataclass
class CitationEntry:
    """Citační záznam"""
    id: str
    title: str
    authors: List[str]
    year: int
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    entry_type: str = "article"  # article, book, inproceedings, etc.

class ProjectTemplateManager:
    """Manager pro research project templates"""

    def __init__(self):
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[ProjectType, Dict]:
        """Inicializuje template pro různé typy projektů"""
        return {
            ProjectType.LITERATURE_REVIEW: {
                "milestones": [
                    {"name": "Define research question", "days": 3},
                    {"name": "Search strategy development", "days": 5},
                    {"name": "Initial paper collection", "days": 10},
                    {"name": "Screening and selection", "days": 14},
                    {"name": "Data extraction", "days": 21},
                    {"name": "Analysis and synthesis", "days": 28},
                    {"name": "Writing first draft", "days": 35},
                    {"name": "Review and revision", "days": 42}
                ],
                "sections": [
                    "Abstract", "Introduction", "Methods", "Results",
                    "Discussion", "Conclusion", "References"
                ],
                "recommended_databases": [
                    "PubMed", "Web of Science", "Scopus", "Google Scholar"
                ]
            },
            ProjectType.EXPERIMENTAL_STUDY: {
                "milestones": [
                    {"name": "Literature review", "days": 7},
                    {"name": "Hypothesis formation", "days": 10},
                    {"name": "Methodology design", "days": 14},
                    {"name": "Ethics approval", "days": 21},
                    {"name": "Data collection", "days": 35},
                    {"name": "Statistical analysis", "days": 42},
                    {"name": "Results interpretation", "days": 49},
                    {"name": "Manuscript writing", "days": 56}
                ],
                "sections": [
                    "Abstract", "Introduction", "Methods", "Results",
                    "Discussion", "Limitations", "Conclusion", "References"
                ],
                "checklist": [
                    "Ethics approval obtained",
                    "Sample size calculated",
                    "Statistical plan defined",
                    "Data management plan",
                    "Quality control measures"
                ]
            },
            ProjectType.META_ANALYSIS: {
                "milestones": [
                    {"name": "Protocol development", "days": 7},
                    {"name": "PROSPERO registration", "days": 10},
                    {"name": "Systematic search", "days": 14},
                    {"name": "Study selection", "days": 21},
                    {"name": "Quality assessment", "days": 28},
                    {"name": "Data extraction", "days": 35},
                    {"name": "Statistical meta-analysis", "days": 42},
                    {"name": "PRISMA reporting", "days": 49}
                ],
                "guidelines": ["PRISMA", "MOOSE", "Cochrane Handbook"],
                "tools": ["RevMan", "R metafor", "Stata", "CMA"]
            }
        }

    def create_project_from_template(
        self,
        project_type: ProjectType,
        title: str,
        owner_id: str,
        custom_params: Optional[Dict] = None
    ) -> ResearchProject:
        """Vytvoří nový projekt z template"""

        project_id = f"proj_{int(time.time())}"
        template = self.templates.get(project_type, {})

        # Generate milestones with dates
        milestones = []
        start_date = datetime.now()

        for milestone in template.get("milestones", []):
            milestone_date = start_date + timedelta(days=milestone["days"])
            milestones.append({
                "name": milestone["name"],
                "target_date": milestone_date.isoformat(),
                "completed": False,
                "notes": ""
            })

        # Create project
        project = ResearchProject(
            id=project_id,
            title=title,
            project_type=project_type,
            description=f"Research project: {title}",
            created_date=datetime.now(),
            last_modified=datetime.now(),
            owner_id=owner_id,
            milestones=milestones
        )

        # Add template-specific notes
        if "checklist" in template:
            project.notes.append({
                "type": "checklist",
                "content": template["checklist"],
                "created": datetime.now().isoformat()
            })

        if "guidelines" in template:
            project.notes.append({
                "type": "guidelines",
                "content": template["guidelines"],
                "created": datetime.now().isoformat()
            })

        return project

class AnnotationManager:
    """Manager pro collaborative annotations"""

    def __init__(self):
        self.annotations: Dict[str, Annotation] = {}
        self.paper_annotations: Dict[str, List[str]] = {}  # paper_id -> annotation_ids

    async def add_annotation(
        self,
        paper_id: str,
        user_id: str,
        content: str,
        annotation_type: str = "note",
        position: Optional[Dict] = None,
        tags: List[str] = None,
        is_public: bool = False
    ) -> str:
        """Přidá novou anotaci"""

        annotation_id = f"ann_{int(time.time() * 1000)}"

        annotation = Annotation(
            id=annotation_id,
            paper_id=paper_id,
            user_id=user_id,
            content=content,
            annotation_type=annotation_type,
            position=position,
            tags=tags or [],
            is_public=is_public
        )

        self.annotations[annotation_id] = annotation

        if paper_id not in self.paper_annotations:
            self.paper_annotations[paper_id] = []
        self.paper_annotations[paper_id].append(annotation_id)

        return annotation_id

    async def get_paper_annotations(
        self,
        paper_id: str,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Získá anotace pro paper"""

        if paper_id not in self.paper_annotations:
            return []

        annotations = []
        for ann_id in self.paper_annotations[paper_id]:
            annotation = self.annotations[ann_id]

            # Filter based on visibility and user
            if annotation.is_public or (user_id and annotation.user_id == user_id):
                annotations.append({
                    "id": annotation.id,
                    "user_id": annotation.user_id,
                    "content": annotation.content,
                    "type": annotation.annotation_type,
                    "position": annotation.position,
                    "timestamp": annotation.timestamp.isoformat(),
                    "tags": annotation.tags,
                    "is_public": annotation.is_public
                })

        return sorted(annotations, key=lambda x: x["timestamp"])

    async def search_annotations(
        self,
        query: str,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Vyhledá anotace podle obsahu"""

        results = []
        query_lower = query.lower()

        for annotation in self.annotations.values():
            # Check visibility
            if not annotation.is_public and (not user_id or annotation.user_id != user_id):
                continue

            # Check content match
            if (query_lower in annotation.content.lower() or
                any(query_lower in tag.lower() for tag in annotation.tags)):

                results.append({
                    "id": annotation.id,
                    "paper_id": annotation.paper_id,
                    "user_id": annotation.user_id,
                    "content": annotation.content,
                    "type": annotation.annotation_type,
                    "timestamp": annotation.timestamp.isoformat(),
                    "tags": annotation.tags
                })

        return results

class CitationManager:
    """Manager pro citation management a export"""

    def __init__(self):
        self.citations: Dict[str, CitationEntry] = {}
        self.citation_styles = self._load_citation_styles()

    def _load_citation_styles(self) -> Dict[CitationStyle, Dict]:
        """Načte citation style templates"""
        return {
            CitationStyle.APA: {
                "article": "{authors} ({year}). {title}. *{journal}*, *{volume}*({issue}), {pages}. {doi}",
                "book": "{authors} ({year}). *{title}*. {publisher}.",
                "inproceedings": "{authors} ({year}). {title}. In *{booktitle}* (pp. {pages}). {publisher}."
            },
            CitationStyle.MLA: {
                "article": "{authors}. \"{title}.\" *{journal}*, vol. {volume}, no. {issue}, {year}, pp. {pages}.",
                "book": "{authors}. *{title}*. {publisher}, {year}.",
                "inproceedings": "{authors}. \"{title}.\" *{booktitle}*, {year}, pp. {pages}."
            },
            CitationStyle.IEEE: {
                "article": "{authors}, \"{title},\" *{journal}*, vol. {volume}, no. {issue}, pp. {pages}, {year}.",
                "book": "{authors}, *{title}*. {publisher}, {year}.",
                "inproceedings": "{authors}, \"{title},\" in *{booktitle}*, {year}, pp. {pages}."
            }
        }

    async def add_citation(self, citation_data: Dict[str, Any]) -> str:
        """Přidá citaci do manageru"""

        citation_id = citation_data.get('id', f"cite_{int(time.time())}")

        citation = CitationEntry(
            id=citation_id,
            title=citation_data.get('title', ''),
            authors=citation_data.get('authors', []),
            year=citation_data.get('year', datetime.now().year),
            journal=citation_data.get('journal'),
            volume=citation_data.get('volume'),
            issue=citation_data.get('issue'),
            pages=citation_data.get('pages'),
            doi=citation_data.get('doi'),
            url=citation_data.get('url'),
            abstract=citation_data.get('abstract'),
            keywords=citation_data.get('keywords', []),
            entry_type=citation_data.get('entry_type', 'article')
        )

        self.citations[citation_id] = citation
        return citation_id

    def format_citation(
        self,
        citation_id: str,
        style: CitationStyle = CitationStyle.APA
    ) -> str:
        """Formátuje citaci podle stylu"""

        if citation_id not in self.citations:
            return ""

        citation = self.citations[citation_id]
        style_templates = self.citation_styles.get(style, {})
        template = style_templates.get(citation.entry_type, style_templates.get('article', ''))

        # Format authors
        if len(citation.authors) == 1:
            authors_str = citation.authors[0]
        elif len(citation.authors) == 2:
            authors_str = f"{citation.authors[0]} & {citation.authors[1]}"
        else:
            authors_str = f"{citation.authors[0]} et al."

        # Format DOI
        doi_str = f"https://doi.org/{citation.doi}" if citation.doi else ""

        # Replace placeholders
        formatted = template.format(
            authors=authors_str,
            year=citation.year,
            title=citation.title,
            journal=citation.journal or "",
            volume=citation.volume or "",
            issue=citation.issue or "",
            pages=citation.pages or "",
            doi=doi_str,
            url=citation.url or ""
        )

        # Clean up empty fields
        formatted = re.sub(r'\s*,\s*,', ',', formatted)
        formatted = re.sub(r'\s*\(\s*\)', '', formatted)
        formatted = re.sub(r'\s+', ' ', formatted).strip()

        return formatted

    def export_bibliography(
        self,
        citation_ids: List[str],
        format_type: ExportFormat,
        style: CitationStyle = CitationStyle.APA
    ) -> str:
        """Exportuje bibliografii v požadovaném formátu"""

        if format_type == ExportFormat.BIBTEX:
            return self._export_bibtex(citation_ids)
        elif format_type == ExportFormat.LATEX:
            return self._export_latex_bibliography(citation_ids, style)
        elif format_type == ExportFormat.WORD:
            return self._export_word_format(citation_ids, style)
        else:
            # Default formatted bibliography
            return self._export_formatted_bibliography(citation_ids, style)

    def _export_bibtex(self, citation_ids: List[str]) -> str:
        """Export do BibTeX formátu"""
        bibtex_entries = []

        for citation_id in citation_ids:
            if citation_id not in self.citations:
                continue

            citation = self.citations[citation_id]

            # Generate BibTeX key
            if citation.authors:
                first_author = citation.authors[0].split()[-1]  # Last name
                bibtex_key = f"{first_author}{citation.year}"
            else:
                bibtex_key = f"unknown{citation.year}"

            # Build BibTeX entry
            entry = f"@{citation.entry_type}{{{bibtex_key},\n"
            entry += f"  title = {{{citation.title}}},\n"

            if citation.authors:
                authors_str = " and ".join(citation.authors)
                entry += f"  author = {{{authors_str}}},\n"

            entry += f"  year = {{{citation.year}}},\n"

            if citation.journal:
                entry += f"  journal = {{{citation.journal}}},\n"
            if citation.volume:
                entry += f"  volume = {{{citation.volume}}},\n"
            if citation.issue:
                entry += f"  number = {{{citation.issue}}},\n"
            if citation.pages:
                entry += f"  pages = {{{citation.pages}}},\n"
            if citation.doi:
                entry += f"  doi = {{{citation.doi}}},\n"
            if citation.url:
                entry += f"  url = {{{citation.url}}},\n"

            entry += "}\n"
            bibtex_entries.append(entry)

        return "\n".join(bibtex_entries)

    def _export_latex_bibliography(self, citation_ids: List[str], style: CitationStyle) -> str:
        """Export pro LaTeX"""
        latex_content = "\\begin{thebibliography}{99}\n\n"

        for i, citation_id in enumerate(citation_ids, 1):
            if citation_id not in self.citations:
                continue

            formatted = self.format_citation(citation_id, style)
            latex_content += f"\\bibitem{{ref{i}}} {formatted}\n\n"

        latex_content += "\\end{thebibliography}"
        return latex_content

    def _export_word_format(self, citation_ids: List[str], style: CitationStyle) -> str:
        """Export pro Word"""
        bibliography = []

        for citation_id in citation_ids:
            if citation_id not in self.citations:
                continue

            formatted = self.format_citation(citation_id, style)
            bibliography.append(formatted)

        return "\n\n".join(bibliography)

    def _export_formatted_bibliography(self, citation_ids: List[str], style: CitationStyle) -> str:
        """Export formátované bibliografie"""
        return self._export_word_format(citation_ids, style)

class AcademicWorkflowOrchestrator:
    """Hlavní orchestrátor pro academic workflow"""

    def __init__(self):
        self.template_manager = ProjectTemplateManager()
        self.annotation_manager = AnnotationManager()
        self.citation_manager = CitationManager()
        self.projects: Dict[str, ResearchProject] = {}

    async def create_research_project(
        self,
        title: str,
        project_type: ProjectType,
        owner_id: str,
        description: str = "",
        collaborators: List[str] = None
    ) -> str:
        """Vytvoří nový research projekt"""

        project = self.template_manager.create_project_from_template(
            project_type, title, owner_id
        )

        if description:
            project.description = description
        if collaborators:
            project.collaborators = collaborators

        self.projects[project.id] = project

        return project.id

    async def add_paper_to_project(
        self,
        project_id: str,
        paper_data: Dict[str, Any]
    ) -> bool:
        """Přidá paper do projektu"""

        if project_id not in self.projects:
            return False

        # Add citation
        citation_id = await self.citation_manager.add_citation(paper_data)

        # Add to project
        project = self.projects[project_id]
        project.papers.append(citation_id)
        project.bibliography.append(paper_data)
        project.last_modified = datetime.now()

        return True

    async def annotate_paper(
        self,
        project_id: str,
        paper_id: str,
        user_id: str,
        content: str,
        annotation_type: str = "note",
        tags: List[str] = None
    ) -> Optional[str]:
        """Přidá anotaci k paperu v projektu"""

        if project_id not in self.projects:
            return None

        project = self.projects[project_id]

        # Check if user has access
        if user_id != project.owner_id and user_id not in project.collaborators:
            return None

        annotation_id = await self.annotation_manager.add_annotation(
            paper_id, user_id, content, annotation_type, tags=tags, is_public=True
        )

        return annotation_id

    async def generate_project_bibliography(
        self,
        project_id: str,
        format_type: ExportFormat = ExportFormat.APA,
        style: CitationStyle = CitationStyle.APA
    ) -> Optional[str]:
        """Generuje bibliografii pro projekt"""

        if project_id not in self.projects:
            return None

        project = self.projects[project_id]

        if format_type == ExportFormat.APA:
            # Use citation manager formatting
            bibliography = self.citation_manager.export_bibliography(
                project.papers, format_type, style
            )
        else:
            bibliography = self.citation_manager.export_bibliography(
                project.papers, format_type, style
            )

        return bibliography

    async def export_project_report(
        self,
        project_id: str,
        format_type: ExportFormat = ExportFormat.LATEX
    ) -> Optional[str]:
        """Exportuje kompletní project report"""

        if project_id not in self.projects:
            return None

        project = self.projects[project_id]

        if format_type == ExportFormat.LATEX:
            return self._generate_latex_report(project)
        elif format_type == ExportFormat.WORD:
            return self._generate_word_report(project)
        else:
            return self._generate_markdown_report(project)

    def _generate_latex_report(self, project: ResearchProject) -> str:
        """Generuje LaTeX report"""

        template = self.template_manager.templates.get(project.project_type, {})
        sections = template.get("sections", ["Introduction", "Methods", "Results", "Discussion"])

        latex_content = f"""\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{cite}}
\\usepackage{{url}}

\\title{{{project.title}}}
\\author{{Generated by Academic Workflow System}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{project.description}
\\end{{abstract}}

"""

        # Add sections
        for section in sections:
            latex_content += f"\\section{{{section}}}\n\n% TODO: Add content for {section}\n\n"

        # Add bibliography
        if project.papers:
            bibliography = self.citation_manager.export_bibliography(
                project.papers, ExportFormat.LATEX
            )
            latex_content += f"\n{bibliography}\n"

        latex_content += "\\end{document}"

        return latex_content

    def _generate_word_report(self, project: ResearchProject) -> str:
        """Generuje Word-compatible report"""

        template = self.template_manager.templates.get(project.project_type, {})
        sections = template.get("sections", ["Introduction", "Methods", "Results", "Discussion"])

        content = f"""# {project.title}

**Project Type:** {project.project_type.value}
**Created:** {project.created_date.strftime('%Y-%m-%d')}
**Owner:** {project.owner_id}

## Abstract

{project.description}

"""

        # Add sections
        for section in sections:
            content += f"## {section}\n\n[TODO: Add content for {section}]\n\n"

        # Add bibliography
        if project.papers:
            content += "## References\n\n"
            bibliography = self.citation_manager.export_bibliography(
                project.papers, ExportFormat.WORD
            )
            content += bibliography

        return content

    def _generate_markdown_report(self, project: ResearchProject) -> str:
        """Generuje Markdown report"""
        return self._generate_word_report(project)  # Same format

    async def get_project_stats(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Získá statistiky projektu"""

        if project_id not in self.projects:
            return None

        project = self.projects[project_id]

        # Count annotations
        total_annotations = 0
        for paper_id in project.papers:
            annotations = await self.annotation_manager.get_paper_annotations(paper_id)
            total_annotations += len(annotations)

        # Calculate progress
        completed_milestones = sum(1 for m in project.milestones if m.get('completed', False))
        total_milestones = len(project.milestones)
        progress = (completed_milestones / max(total_milestones, 1)) * 100

        return {
            "project_id": project.id,
            "title": project.title,
            "type": project.project_type.value,
            "papers_count": len(project.papers),
            "annotations_count": total_annotations,
            "collaborators_count": len(project.collaborators),
            "progress_percentage": progress,
            "completed_milestones": completed_milestones,
            "total_milestones": total_milestones,
            "created_date": project.created_date.isoformat(),
            "last_modified": project.last_modified.isoformat()
        }

# Factory funkce
async def create_academic_workflow() -> AcademicWorkflowOrchestrator:
    """Factory pro vytvoření academic workflow systému"""
    orchestrator = AcademicWorkflowOrchestrator()
    logger.info("Academic Workflow Orchestrator initialized")
    return orchestrator
