"""
Chaîne RAG complète AVEC re-ranking.
"""
from typing import Dict, Any, List
from langchain_core.documents import Document

from src.rag.retriever import search
from src.rag.reranker import rerank_documents
from src.rag.prompt import build_prompt
from src.rag.llm import get_llm
from src.utils.config import RETRIEVER_TOP_K
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Nombre de chunks récupérés AVANT re-ranking (large)
RETRIEVER_FETCH_K = 8
# Nombre de chunks gardés APRÈS re-ranking (précis)
RERANK_TOP_N = RETRIEVER_TOP_K  # = 5


def format_context(documents: List[Document]) -> str:
    """Formate les chunks en contexte lisible par le LLM."""
    parts = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "inconnu")
        page = doc.metadata.get("page", "?")
        parts.append(
            f"--- Extrait {i} (Source : {source}, page {page}) ---\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def answer_question(question: str, k: int = RETRIEVER_TOP_K) -> Dict[str, Any]:
    """
    Pipeline RAG complet : retrieval large → re-ranking → LLM.
    """
    logger.info(f"❓ Question : {question}")

    # 1. Retrieval large (15 chunks)
    candidates = search(question, k=RETRIEVER_FETCH_K)

    if not candidates:
        return {
            "answer": "Je n'ai trouvé aucun document pertinent pour répondre.",
            "sources": [],
            "context_used": [],
        }

    logger.info(f"  → {len(candidates)} chunks candidats")

    # 2. Re-ranking (15 → 5)
    documents = rerank_documents(question, candidates, top_n=k)

    # 3. Prompt
    context = format_context(documents)
    prompt = build_prompt()

    # 4. LLM
    llm = get_llm()
    chain = prompt | llm

    logger.info("  → Appel au LLM...")
    response = chain.invoke({
        "context": context,
        "question": question,
    })

    answer = response.content
    logger.info(f"  → Réponse générée ({len(answer)} caractères)")

    # 5. Sources uniques
    sources_set = set()
    sources = []
    for doc in documents:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key not in sources_set:
            sources_set.add(key)
            sources.append({
                "source": doc.metadata.get("source", "inconnu"),
                "page": doc.metadata.get("page", "?"),
            })

    return {
        "answer": answer,
        "sources": sources,
        "context_used": [doc.page_content for doc in documents],
    }

async def answer_question_stream(question: str, k: int = RETRIEVER_TOP_K):
    """
    Version streaming : yield les chunks de réponse au fur et à mesure.
    """
    logger.info(f"❓ Question (streaming) : {question}")
    
    # Retrieval + reranking
    candidates = search(question, k=RETRIEVER_FETCH_K)
    documents = rerank_documents(question, candidates, top_n=k)
    
    context = format_context(documents)
    prompt = build_prompt()
    llm = get_llm()
    
    # Streaming
    chain = prompt | llm
    async for chunk in chain.astream({"context": context, "question": question}):
        if chunk.content:
            yield chunk.content