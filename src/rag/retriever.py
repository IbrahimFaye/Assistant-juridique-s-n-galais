"""
Recherche vectorielle : trouver les chunks pertinents pour une question.
"""
from typing import List
from langchain_core.documents import Document
from src.ingestion.embedder import load_vectorstore
from src.utils.config import RETRIEVER_TOP_K
from src.utils.logger import get_logger

logger = get_logger(__name__)

_vectorstore = None


def get_vectorstore():
    """Singleton : charge la base une seule fois."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()
    return _vectorstore


def search(query: str, k: int = RETRIEVER_TOP_K) -> List[Document]:
    """
    Recherche les k chunks les plus pertinents pour la query.
    """
    vectorstore = get_vectorstore()
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    # results = [(Document, score), ...]
    
    # Trie par score croissant (ChromaDB : plus petit = plus proche)
    results.sort(key=lambda x: x[1])
    
    logger.info(f"Recherche : '{query[:60]}...' → {len(results)} résultats")
    
    return [doc for doc, score in results]