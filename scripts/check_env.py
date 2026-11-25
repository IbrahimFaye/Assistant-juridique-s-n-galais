"""Vérifie que les variables d'environnement sont bien chargées."""
from src.utils.config import (
    LLM_PROVIDER,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    OLLAMA_BASE_URL,
    LLM_MODEL,
)

print(f"LLM_PROVIDER  : {LLM_PROVIDER}")
print(f"MISTRAL_MODEL : {MISTRAL_MODEL}")
print(f"MISTRAL_API_KEY : {'✅ définie (' + str(len(MISTRAL_API_KEY)) + ' caractères)' if MISTRAL_API_KEY else '❌ VIDE'}")
print(f"LLM_MODEL     : {LLM_MODEL}")
print(f"OLLAMA_URL    : {OLLAMA_BASE_URL}")