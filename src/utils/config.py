"""
Configuration centralisée du projet.
Charge les variables depuis .env et expose des constantes.
"""
import os
# Désactive la télémétrie ChromaDB
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"
from pathlib import Path
from dotenv import load_dotenv

# Charge le fichier .env à la racine du projet
load_dotenv()

# === Provider LLM ===
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" ou "mistral"
# === Mistral ===
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
# === Groq ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# === Chemins ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VECTORSTORE_DIR = PROJECT_ROOT / os.getenv("VECTORSTORE_DIR", "vectorstore")

# === LLM ===
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

# === Embeddings ===
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)

# === RAG ===
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))

# === Création des dossiers si absents ===
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, VECTORSTORE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)