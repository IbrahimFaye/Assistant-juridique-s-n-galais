"""
Chargement des PDFs et extraction du texte.
Utilise pdfplumber (meilleur pour les tableaux) avec fallback sur pypdf.
"""
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber
from pypdf import PdfReader
from langchain_core.documents import Document

from src.utils.config import DATA_RAW_DIR
from src.utils.logger import get_logger
import re

logger = get_logger(__name__)


def load_pdf_with_pdfplumber(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extrait le texte d'un PDF page par page avec pdfplumber.
    Retourne une liste de dicts {page, text}.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i, "text": text})
    return pages


def load_pdf_with_pypdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Fallback : extraction avec pypdf (plus simple mais moins fiable sur les tableaux).
    """
    pages = []
    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i, "text": text})
    return pages


def load_single_pdf(pdf_path: Path) -> List[Document]:
    """
    Charge un PDF et retourne une liste de Documents LangChain,
    un par page (avec métadonnées).
    """
    logger.info(f"Chargement : {pdf_path.name}")
    
    # Essaie pdfplumber d'abord
    try:
        pages = load_pdf_with_pdfplumber(pdf_path)
    except Exception as e:
        logger.warning(f"pdfplumber a échoué ({e}), fallback sur pypdf")
        pages = load_pdf_with_pypdf(pdf_path)
    
    if not pages:
        logger.warning(f"Aucun texte extrait de {pdf_path.name}")
        return []
    
    documents = []
    for page_data in pages:
        cleaned = clean_text(page_data["text"])
        if not cleaned:
            continue
        doc = Document(
            page_content=page_data["text"],
            metadata={
                "source": pdf_path.name,
                "page": page_data["page"],
                "file_path": str(pdf_path),
            },
        )
        documents.append(doc)
    
    logger.info(f"  → {len(documents)} pages extraites")
    return documents


def load_all_pdfs(directory: Path = DATA_RAW_DIR) -> List[Document]:
    """
    Charge tous les PDFs d'un dossier.
    """
    pdf_files = sorted(directory.glob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"Aucun PDF trouvé dans {directory}")
        return []
    
    logger.info(f"Trouvé {len(pdf_files)} PDF(s) à charger")
    
    all_documents = []
    for pdf_path in pdf_files:
        documents = load_single_pdf(pdf_path)
        all_documents.extend(documents)
    
    logger.info(f"Total : {len(all_documents)} pages chargées")
    return all_documents


def clean_text(text: str) -> str:
    """
    Nettoie le texte extrait des PDFs (artefacts courants).
    """
    # Remplace (cid:XXX) par un espace
    text = re.sub(r"\(cid:\d+\)", " ", text)
    # Remplace les puces mal encodées
    text = text.replace("\u2022", "-")  # • → -
    # Supprime les espaces multiples
    text = re.sub(r" +", " ", text)
    # Supprime les sauts de ligne multiples (max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()