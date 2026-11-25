"""
Évaluation du RAG — Approche A : séparation in-scope / out-of-scope.

In-scope     : questions dont la réponse est dans les documents.
               → On évalue faithfulness, relevancy, precision, recall.
Out-of-scope : questions hors périmètre (info absente).
               → On évalue uniquement le taux de refus correct.

Usage : python -m scripts.evaluate_rag
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.chain import answer_question
from src.utils.config import GROQ_API_KEY, EMBEDDING_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)


# === Modèle juge : classique (non-reasoning), rapide, fiable ===
JUDGE_MODEL = "qwen/qwen3.8-27b"


# === Jeu de test ===
TEST_SET = [
    # --- Questions IN-SCOPE (réponse attendue dans les documents) ---
    {
        "question": "Qui est éligible à la bourse ?",
        "reference": "Les bacheliers des séries S1, S3, S4, S5, T1, T2, F6 (demi-bourse), S2, S2A, G, STEG, L, les orphelins, les handicapés physiques et les malades chroniques.",
        "type": "in_scope",
    },
    {
        "question": "Quand obtient-on une bourse d'excellence ?",
        "reference": "Les bacheliers de l'année en cours inscrits, ayant obtenu la mention Très Bien, Bien, ou étant lauréat du concours général.",
        "type": "in_scope",
    },
    {
        "question": "Quand débutera le paiement des bourses de septembre 2026 ?",
        "reference": "Le 07 septembre 2026.",
        "type": "in_scope",
    },
    {
        "question": "Quels sont les critères pour une bourse sociale ?",
        "reference": "Être orphelin, enfant de parents indigents, handicapé physique ou atteint de maladie chronique.",
        "type": "in_scope",
    },
    {
        "question": "Comment renouveler sa bourse ?",
        "reference": "Le renouvellement dépend du cycle d'étude et de l'âge (moins de 30 ans), avec un redoublement autorisé.",
        "type": "in_scope",
    },
    {
        "question": "Quelles sont les séries de bac qui donnent droit à une demi-bourse ?",
        "reference": "Les séries S1, S3, S4, S5, T1, T2 et F6.",
        "type": "in_scope",
    },
    {
        "question": "Y a-t-il une bourse pour les étudiants handicapés ?",
        "reference": "Oui, une bourse sociale est octroyée aux handicapés physiques.",
        "type": "in_scope",
    },
    {
        "question": "Quel est le numéro de téléphone de la Direction des Bourses ?",
        "reference": "Le 33 960 11 11.",
        "type": "in_scope",
    },

    # --- Questions OUT-OF-SCOPE (info absente des documents) ---
    {
        "question": "Quel est le montant de la bourse pour un étudiant en médecine ?",
        "reference": "Je ne trouve pas cette information dans mes documents.",
        "type": "out_of_scope",
    },
    {
        "question": "Quelle est la capitale du Sénégal ?",
        "reference": "Je ne trouve pas cette information dans mes documents.",
        "type": "out_of_scope",
    },
]


# === Prompts pour le LLM juge ===
FAITHFULNESS_PROMPT = """Contexte :
{context}

Réponse :
{answer}

La réponse est-elle supportée par le contexte ? Réponds uniquement "OUI" ou "NON"."""


RELEVANCY_PROMPT = """Question : {question}
Réponse : {answer}

La réponse répond-elle à la question ? Réponds uniquement "OUI" ou "NON"."""


# === Marqueurs de refus poli ===
REFUSAL_MARKERS = [
    "je ne trouve pas",
    "je n'ai pas cette information",
    "je ne dispose pas",
    "je n'ai pas trouvé",
    "veuillez contacter",
    "je te recommande de contacter",
    "je ne peux pas répondre",
    "pas dans mes documents",
    "je n'ai pas d'information",
]


# === LLM juge ===
def get_judge_llm() -> ChatGroq:
    return ChatGroq(
        model=JUDGE_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=10,
        timeout=30,
        max_retries=2,
    )


# === Modèle d'embeddings pour les métriques algorithmiques ===
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# === Métriques algorithmiques ===
def compute_context_precision(documents, reference, embeddings_model) -> float:
    """
    Proportion de chunks dont la similarité avec la référence > 0.5.
    Approximation rapide de context_precision.
    """
    if not documents or not reference:
        return 0.0
    ref_emb = embeddings_model.embed_query(reference)
    relevant = 0
    for doc in documents:
        doc_emb = embeddings_model.embed_query(doc)
        sim = cosine_similarity([ref_emb], [doc_emb])[0][0]
        if sim > 0.5:
            relevant += 1
    return relevant / len(documents)


def compute_context_recall(documents, reference, embeddings_model) -> float:
    """
    Similarité maximale entre la référence et l'ensemble des chunks.
    Approximation rapide de context_recall.
    """
    if not documents or not reference:
        return 0.0
    ref_emb = embeddings_model.embed_query(reference)
    max_sim = 0.0
    for doc in documents:
        doc_emb = embeddings_model.embed_query(doc)
        sim = cosine_similarity([ref_emb], [doc_emb])[0][0]
        max_sim = max(max_sim, sim)
    return max_sim


# === Juge LLM ===
def judge_yes_no(llm, prompt_template: str, **kwargs) -> float:
    """
    Appelle le LLM juge et retourne 1.0 pour OUI, 0.0 pour NON.
    Parse robuste : cherche OUI/NON en gérant les cas ambigus.
    """
    try:
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        response = chain.invoke(kwargs)
        text = response.content.strip().upper()

        has_oui = "OUI" in text
        has_non = "NON" in text

        if has_oui and not has_non:
            return 1.0
        if has_non and not has_oui:
            return 0.0
        if has_oui and has_non:
            # Les deux présents : prend le dernier mentionné
            return 1.0 if text.rfind("OUI") > text.rfind("NON") else 0.0

        logger.warning(f"    [juge] Réponse inattendue : '{text}' → NON")
        return 0.0
    except Exception as e:
        logger.warning(f"    [juge] Erreur : {e} → NON")
        return 0.0


# === Détection de refus correct ===
def is_correct_refusal(answer: str) -> float:
    """Détecte si la réponse est un refus poli (bon comportement out-of-scope)."""
    answer_lower = answer.lower()
    for marker in REFUSAL_MARKERS:
        if marker in answer_lower:
            return 1.0
    return 0.0


# === Pipeline principal ===
def main():
    logger.info("=" * 60)
    logger.info("🔍 ÉVALUATION DU RAG — Approche A")
    logger.info("=" * 60)

    # 1. Charger les modèles
    logger.info("Chargement des embeddings pour les métriques algorithmiques...")
    embeddings_model = get_embedding_model()

    logger.info(f"Chargement du LLM juge : {JUDGE_MODEL}")
    judge = get_judge_llm()

    # 2. Boucler sur chaque question
    results = []

    for i, test in enumerate(TEST_SET, 1):
        logger.info(f"[{i}/{len(TEST_SET)}] ({test['type']}) {test['question'][:55]}...")

        # Exécuter le RAG
        t0 = time.time()
        rag_result = answer_question(test["question"], k=5)
        latency = time.time() - t0

        # Métriques algorithmiques (indépendantes du type)
        cp = compute_context_precision(
            rag_result["context_used"], test["reference"], embeddings_model
        )
        cr = compute_context_recall(
            rag_result["context_used"], test["reference"], embeddings_model
        )

        # Métriques LLM selon le type de question
        if test["type"] == "in_scope":
            context_str = "\n---\n".join(rag_result["context_used"])[:3000]

            faith = judge_yes_no(
                judge, FAITHFULNESS_PROMPT,
                context=context_str,
                answer=rag_result["answer"],
            )
            relev = judge_yes_no(
                judge, RELEVANCY_PROMPT,
                question=test["question"],
                answer=rag_result["answer"],
            )
        else:  # out_of_scope
            # Pour out-of-scope : on vérifie le refus correct
            faith = 1.0  # Pas de contexte à vérifier, on considère fidèle
            relev = is_correct_refusal(rag_result["answer"])

        # Enregistrer
        results.append({
            "type": test["type"],
            "question": test["question"],
            "answer": rag_result["answer"],
            "reference": test["reference"],
            "faithfulness": faith,
            "answer_relevancy": relev,
            "context_precision": round(cp, 3),
            "context_recall": round(cr, 3),
            "latency_s": round(latency, 2),
        })

        logger.info(
            f"  → faith={faith} relev={relev} "
            f"cp={cp:.2f} cr={cr:.2f} ({latency:.1f}s)"
        )

    # 3. Calcul des moyennes séparées
    in_scope = [r for r in results if r["type"] == "in_scope"]
    out_scope = [r for r in results if r["type"] == "out_of_scope"]

    def avg(items, key):
        if not items:
            return 0.0
        return round(sum(r[key] for r in items) / len(items), 4)

    summary = {
        "in_scope": {
            "count": len(in_scope),
            "faithfulness": avg(in_scope, "faithfulness"),
            "answer_relevancy": avg(in_scope, "answer_relevancy"),
            "context_precision": avg(in_scope, "context_precision"),
            "context_recall": avg(in_scope, "context_recall"),
        },
        "out_of_scope": {
            "count": len(out_scope),
            "correct_refusal_rate": avg(out_scope, "answer_relevancy"),
        },
        "avg_latency_s": round(
            sum(r["latency_s"] for r in results) / len(results), 2
        ),
    }

    # 4. Affichage
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS DE L'ÉVALUATION")
    print("=" * 60)
    print(f"\n=== IN-SCOPE ({summary['in_scope']['count']} questions) ===")
    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"  {k:25s} : {summary['in_scope'][k]}")
    print(f"\n=== OUT-OF-SCOPE ({summary['out_of_scope']['count']} questions) ===")
    print(f"  {'correct_refusal_rate':25s} : {summary['out_of_scope']['correct_refusal_rate']}")
    print(f"\n=== PERFORMANCE ===")
    print(f"  {'avg_latency_s':25s} : {summary['avg_latency_s']}")
    print()

    # 5. Sauvegarde
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eval_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "summary": summary,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"💾 Résultats sauvegardés : {output_file}")

    return summary


if __name__ == "__main__":
    main()