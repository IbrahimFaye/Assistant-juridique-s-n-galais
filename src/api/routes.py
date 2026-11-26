"""
Endpoints de l'API.
"""
import time
from fastapi import APIRouter, HTTPException

from src.api.schemas import AskRequest, AskResponse, Source, HealthResponse
from src.rag.chain import answer_question, answer_question_stream
from src.rag.retriever import get_vectorstore
from src.utils.config import LLM_PROVIDER
from src.utils.logger import get_logger
from fastapi.responses import StreamingResponse
import json

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """
    Vérifie que l'API et le vectorstore sont opérationnels.
    """
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
    except Exception as e:
        logger.error(f"Vectorstore KO : {e}")
        raise HTTPException(status_code=503, detail="Vectorstore indisponible")
    
    return HealthResponse(
        status="ok",
        vectorstore_size=count,
        llm_provider=LLM_PROVIDER,
    )


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Répond à une question.
    """
    logger.info(f"📨 Nouvelle question : {request.question[:80]}")
    start = time.time()
    
    try:
        result = answer_question(request.question, k=request.k)
    except Exception as e:
        logger.exception("Erreur lors du traitement")
        raise HTTPException(status_code=500, detail=str(e))
    
    latency_ms = (time.time() - start) * 1000
    logger.info(f"✅ Réponse en {latency_ms:.0f} ms")
    
    return AskResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        latency_ms=round(latency_ms, 1),
    )




@router.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """Endpoint streaming pour l'UI."""
    async def event_generator():
        async for token in answer_question_stream(request.question, k=request.k):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )