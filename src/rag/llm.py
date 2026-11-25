"""
Factory LLM : permet de switcher entre Ollama (local) et Mistral API (prod).
"""
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.utils.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    LLM_MODEL,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_llm: BaseChatModel = None


def _create_ollama_llm() -> BaseChatModel:
    logger.info(f"LLM : Ollama / {LLM_MODEL}")
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        num_ctx=4096,
        num_predict=400, 
        num_thread=4,
    )


def _create_mistral_llm() -> BaseChatModel:
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY manquante dans .env")
    
    from langchain_mistralai import ChatMistralAI
    
    logger.info(f"LLM : Mistral API / {MISTRAL_MODEL}")
    return ChatMistralAI(
        model=MISTRAL_MODEL,
        api_key=MISTRAL_API_KEY,
        temperature=0.2,
        max_tokens=800,
        max_retries=3,
        timeout=60,
    )

def _create_groq_llm() -> BaseChatModel:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY manquante dans .env")
    
    from langchain_groq import ChatGroq
    
    logger.info(f"LLM : Groq / {GROQ_MODEL}")
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=800,
        max_retries=2,
    )


def get_llm() -> BaseChatModel:
    global _llm
    if _llm is None:
        if LLM_PROVIDER == "mistral":
            _llm = _create_mistral_llm()
        elif LLM_PROVIDER == "groq":
            _llm = _create_groq_llm()
        else:
            _llm = _create_ollama_llm()
    return _llm