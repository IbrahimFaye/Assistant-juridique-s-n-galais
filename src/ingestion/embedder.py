"""
Création des embeddings et stockage dans ChromaDB.
"""
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.utils.config import EMBEDDING_MODEL, VECTORSTORE_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "bourses_senegal"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Charge le modèle d'embeddings.
    """
    logger.info(f"Chargement du modèle d'embeddings : {EMBEDDING_MODEL}")
    
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},       # ou "cuda" si GPU
        encode_kwargs={"normalize_embeddings": True},  # cosinus
    )


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """
    Crée (ou remplace) la base vectorielle à partir des chunks.
    """
    logger.info(f"Création de la base vectorielle ({len(chunks)} chunks)...")
    logger.info("Cela peut prendre quelques minutes...")
    
    embeddings = get_embedding_model()
    
    # Supprime l'ancienne collection si elle existe
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )
    
    # Ajoute les chunks par lots (batch) pour voir la progression
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        logger.info(f"  Lot {i // batch_size + 1} ajouté ({len(batch)} chunks)")
    
    logger.info(f"✅ Base vectorielle prête dans {VECTORSTORE_DIR}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Charge une base vectorielle existante (pour le retrieval).
    """
    embeddings = get_embedding_model()
    
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )
    
    count = vectorstore._collection.count()
    logger.info(f"Base vectorielle chargée : {count} chunks")
    return vectorstore