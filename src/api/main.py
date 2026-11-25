"""
Point d'entrée FastAPI.
Lance avec : uvicorn src.api.main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Chatbot Bourses Sénégal",
    description="API RAG pour répondre aux questions des étudiants boursiers",
    version="0.1.0",
)

# CORS : autorise les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["RAG"])


@app.get("/")
async def root():
    return {
        "message": "Chatbot Bourses Sénégal API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 API démarrée")