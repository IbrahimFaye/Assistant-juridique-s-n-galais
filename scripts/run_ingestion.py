"""
Script principal d'ingestion.
Usage : python -m src.ingestion.run_ingestion
"""
import shutil
from src.utils.config import VECTORSTORE_DIR
from src.utils.logger import get_logger
from src.ingestion.loader import load_all_pdfs
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import build_vectorstore, load_vectorstore

logger = get_logger(__name__)


def main(force_rebuild: bool = True):
    logger.info("=" * 60)
    logger.info("DÉMARRAGE DE L'INGESTION")
    logger.info("=" * 60)
    
    # 1. Nettoyer l'ancien vectorstore
    if force_rebuild and VECTORSTORE_DIR.exists():
        logger.info(f"Suppression de l'ancien vectorstore...")
        shutil.rmtree(VECTORSTORE_DIR)
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Charger les PDFs
    documents = load_all_pdfs()
    if not documents:
        logger.error("Aucun document chargé. Vérifie data/raw/")
        return
    
    # 3. Chunker
    chunks = chunk_documents(documents)
    if not chunks:
        logger.error("Aucun chunk créé.")
        return
    
    # 4. Créer la base vectorielle
    vectorstore = build_vectorstore(chunks)
    
    # 5. Vérification finale
    count = vectorstore._collection.count()
    logger.info("=" * 60)
    logger.info(f"✅ INGESTION TERMINÉE : {count} chunks indexés")
    logger.info("=" * 60)


if __name__ == "__main__":
    main(force_rebuild=True)