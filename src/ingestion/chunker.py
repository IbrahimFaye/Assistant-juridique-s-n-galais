"""
Découpage des Documents en chunks optimaux pour la recherche.
"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_splitter() -> RecursiveCharacterTextSplitter:
    """
    Crée le splitter configuré pour des documents administratifs français.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",       # paragraphe
            "\n",         # ligne
            ". ",         # phrase
            "! ",
            "? ",
            "; ",
            ", ",
            " ",          # mot
            "",           # caractère (dernier recours)
        ],
        keep_separator=True,
    )


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Découpe une liste de Documents en chunks.
    Préserve les métadonnées et ajoute chunk_index.
    """
    splitter = create_splitter()
    chunks = splitter.split_documents(documents)
    
    # Enrichir les métadonnées de chaque chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        # Un ID unique pour traçabilité
        chunk.metadata["chunk_id"] = (
            f"{chunk.metadata.get('source', 'unknown')}"
            f"__p{chunk.metadata.get('page', '?')}"
            f"__c{i}"
        )
    
    logger.info(f"{len(documents)} pages → {len(chunks)} chunks")
    logger.info(
        f"Taille moyenne d'un chunk : "
        f"{sum(len(c.page_content) for c in chunks) // len(chunks)} caractères"
    )
    
    return chunks