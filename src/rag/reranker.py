"""
Re-ranking des documents avec un cross-encoder.
"""
from typing import List
from langchain_core.documents import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from src.utils.logger import get_logger

logger = get_logger(__name__)

#RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL = "BAAI/bge-reranker-base"

_reranker = None


def get_reranker() -> CrossEncoderReranker:
    """Singleton : charge le cross-encoder une seule fois."""
    global _reranker
    if _reranker is None:
        logger.info(f"Chargement du reranker : {RERANKER_MODEL}")
        cross_encoder = HuggingFaceCrossEncoder(
            model_name=RERANKER_MODEL,
            model_kwargs={
                "device": "cpu",
                "trust_remote_code": True,
                 },
        )
        _reranker = CrossEncoderReranker(
            model=cross_encoder,
            top_n=5,
        )
    return _reranker


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = 5,
) -> List[Document]:
    """Réordonne les documents par pertinence pour la query."""
    if not documents:
        return []

    logger.info(f"  Re-ranking : {len(documents)} → top {top_n}")

    reranker = get_reranker()
    reranker.top_n = top_n

    reranked = reranker.compress_documents(documents, query)

    logger.info(f"  → {len(reranked)} chunks après re-ranking")
    return reranked