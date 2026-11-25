"""
Schémas Pydantic pour l'API.
"""
from pydantic import BaseModel, Field
from typing import List


class AskRequest(BaseModel):
    """Requête d'une question."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="La question de l'étudiant",
    )
    k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Nombre de chunks à récupérer",
    )


class Source(BaseModel):
    """Une source documentaire."""
    source: str
    page: int | str


class AskResponse(BaseModel):
    """Réponse à une question."""
    answer: str
    sources: List[Source]
    latency_ms: float


class HealthResponse(BaseModel):
    """État du service."""
    status: str
    vectorstore_size: int
    llm_provider: str