# ⚖️ Assistant Juridique Sénégal — Chatbot RAG

> Un assistant virtuel intelligent qui répond aux questions juridiques des citoyens sénégalais en se basant **uniquement** sur les textes officiels (Code du travail, Code de la famille, etc.).

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.114-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-orange.svg)](https://www.langchain.com/)

## 🎯 Présentation

**Assistant Juridique Sénégal** est un chatbot basé sur une architecture **RAG (Retrieval-Augmented Generation)** conçu pour répondre de manière fiable, factuelle et bienveillante aux questions juridiques des citoyens sénégalais.

Le système s'appuie exclusivement sur des **textes officiels** (Code du travail 2026, Code de la famille 1972/1999) pour garantir des réponses traçables et sans hallucination. Chaque réponse cite l'article et le texte de loi source.

### Public cible
- 👨‍💼 Salariés et employeurs (droit du travail)
- 👨‍👩‍👧‍👦 Citoyens (droit de la famille, succession)
- 🎓 Étudiants en droit
- 🏛️ ONG et associations d'aide juridique

---

## ✨ Fonctionnalités

- ✅ **Réponses sourcées** : chaque réponse cite l'article et la page du texte de loi
- ✅ **Anti-hallucination** : refus explicite si l'information n'est pas dans le corpus (taux de refus : 100% sur les questions hors périmètre)
- ✅ **Re-ranking** : Cross-Encoder `BAAI/bge-reranker-base` pour ne garder que les extraits les plus pertinents
- ✅ **Multi-LLM** : support Ollama (local), Mistral API, Groq API
- ✅ **Interface chat** : Streamlit avec historique, sources dépliables et latence
- ✅ **Évaluation intégrée** : script de benchmark in-scope / out-of-scope avec métriques (faithfulness, relevancy, precision, recall)
- ✅ **Multilingue** : embeddings `paraphrase-multilingual-mpnet-base-v2` (français, wolof, anglais)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UTILISATEUR (Citoyen)                     │
│                        🌐 Web / Mobile                       │
└─────────────────────────────┬───────────────────────────────┘
                              │ HTTPS
┌─────────────────────────────▼───────────────────────────────┐
│                  FRONTEND (Streamlit)                        │
│  • Chat UI  • Historique  • Affichage des sources            │
└─────────────────────────────┬───────────────────────────────┘
                              │ REST API (JSON)
┌─────────────────────────────▼───────────────────────────────┐
│                   BACKEND (FastAPI)                          │
│  • /api/v1/ask     → pose une question                       │
│  • /api/v1/health  → monitoring                              │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                     MOTEUR RAG                               │
│  1. Retrieval (12 chunks candidats)                          │
│  2. Re-ranking (12 → 5)                                      │
│  3. Prompt Builder (anti-hallucination + citation articles)  │
│  4. LLM Caller (Ollama / Mistral / Groq)                     │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│              VECTOR STORE (ChromaDB)                         │
│  Embeddings : paraphrase-multilingual-mpnet-base-v2          │
│  Corpus : Code du travail 2026 + Code de la famille          │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                INGESTION (offline)                           │
│  PDF → pdfplumber → clean → chunk (800/150) → embeddings     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Stack technique

| Composant | Choix | Justification |
|---|---|---|
| **Langage** | Python 3.11 | Écosystème IA mature |
| **Framework RAG** | LangChain | Grande communauté, docs abondantes |
| **LLM (génération)** | Groq `openai/gpt-oss-20b` (cloud) / Ollama `llama3.1:8b` (local) | Rapide, gratuit ou local |
| **LLM (juge éval)** | Groq `openai/gpt-oss-120b` | Fiable pour le benchmarking |
| **Embeddings** | `paraphrase-multilingual-mpnet-base-v2` | Multilingue, gratuit, local |
| **Vector DB** | ChromaDB | Simple, local, gratuit |
| **Re-ranker** | `BAAI/bge-reranker-base` | Compromis vitesse/précision sur CPU |
| **Backend** | FastAPI | Async, docs auto (Swagger) |
| **Frontend** | Streamlit | Prototypage rapide |
| **Extraction PDF** | pdfplumber + pypdf (fallback) | Optimisé pour les tableaux |
