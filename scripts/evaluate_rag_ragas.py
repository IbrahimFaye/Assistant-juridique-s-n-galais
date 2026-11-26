"""
Évaluation du système RAG juridique avec RAGAS.
"""
import json
from datetime import datetime
from pathlib import Path

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.chain import answer_question
from src.utils.config import GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# JEU DE TEST JURIDIQUE — 15 questions
# ============================================================
# Règles :
# - In-scope : la réponse est dans le corpus (Code du travail + Code de la famille)
# - Out-of-scope : la réponse N'EST PAS dans le corpus → doit refuser
# - Les références doivent être factuelles et vérifiables
# ============================================================

TEST_SET = [
    # ---------- DROIT DU TRAVAIL (Code du travail 2026) ----------
    {
        "question": "Quelle est la durée légale du travail au Sénégal ?",
        "reference": "La durée légale du travail est fixée à 40 heures par semaine, soit 2 080 heures par an (article L.127 du Code du travail).",
    },
    {
        "question": "Quel est le préavis pour un licenciement d'un travailleur ?",
        "reference": "Le préavis est de 1 mois pour les employés et ouvriers, 3 mois pour les cadres, sauf dispositions conventionnelles plus favorables (article L.56 du Code du travail).",
    },
    {
        "question": "Quelle est la durée maximale d'un CDD au Sénégal ?",
        "reference": "Le contrat à durée déterminée (CDD) ne peut excéder 2 ans, renouvellement compris (article L.43 du Code du travail).",
    },
    {
        "question": "Combien de jours de congés payés par mois un salarié acquiert-il ?",
        "reference": "Le salarié acquiert 2 jours ouvrables de congés payés par mois de service effectif (article L.145 du Code du travail).",
    },
    {
        "question": "Quelles sont les causes de suspension du contrat de travail ?",
        "reference": "La suspension du contrat de travail peut résulter de la maladie, de la maternité, d'un accident du travail, du chômage technique, du congé parental, ou de l'exercice d'un mandat politique ou syndical (articles L.51 et suivants du Code du travail).",
    },
    {
        "question": "Quel est le montant de l'indemnité de licenciement ?",
        "reference": "L'indemnité de licenciement est calculée à raison de 1 mois de salaire par année d'ancienneté pour les 5 premières années, puis 1,5 mois par année au-delà (article L.58 du Code du travail).",
    },

    # ---------- DROIT DE LA FAMILLE (Code de la famille 1972/1999) ----------
    {
        "question": "Quel est l'âge minimum pour se marier au Sénégal ?",
        "reference": "L'âge minimum pour contracter mariage est de 18 ans pour l'homme et 16 ans pour la femme, sauf dispense du procureur (article 111 du Code de la famille).",
    },
    {
        "question": "Quels sont les cas de divorce au Sénégal ?",
        "reference": "Le divorce peut être prononcé pour : consentement mutuel, faute, ou séparation de fait depuis plus de 2 ans (articles 165 et suivants du Code de la famille).",
    },
    {
        "question": "Qui hérite en cas de décès sans testament ?",
        "reference": "En l'absence de testament, la succession est dévolue aux héritiers légaux : conjoint survivant, descendants, ascendants, et collatéraux, selon l'ordre et les règles de la dévolution légale (articles 570 et suivants du Code de la famille).",
    },
    {
        "question": "Quelle est la durée de la période de viduité pour une veuve ?",
        "reference": "La femme divorcée ou veuve doit observer une période de viduité de 4 mois et 10 jours avant de contracter un nouveau mariage (article 116 du Code de la famille).",
    },
    {
        "question": "Comment est fixée la pension alimentaire ?",
        "reference": "La pension alimentaire est fixée par le juge en fonction des besoins du créancier et des ressources du débiteur (articles 275 et suivants du Code de la famille).",
    },

    # ---------- QUESTIONS OUT-OF-SCOPE (doivent être refusées) ----------
    {
        "question": "Quelle est la capitale du Sénégal ?",
        "reference": "Je n'ai pas cette information dans mes documents. Veuillez consulter un professionnel du droit ou les textes officiels.",
    },
    {
        "question": "Quel est le montant de la bourse pour un étudiant en médecine ?",
        "reference": "Je n'ai pas cette information dans mes documents. Veuillez consulter un professionnel du droit ou les textes officiels.",
    },
    {
        "question": "Qui a gagné la Coupe d'Afrique des Nations 2024 ?",
        "reference": "Je n'ai pas cette information dans mes documents. Veuillez consulter un professionnel du droit ou les textes officiels.",
    },
    {
        "question": "Comment obtenir un visa pour la France ?",
        "reference": "Je n'ai pas cette information dans mes documents. Veuillez consulter un professionnel du droit ou les textes officiels.",
    },
]


# ============================================================
# FONCTIONS D'ÉVALUATION (inchangées)
# ============================================================

def build_evaluator_llm():
    """Construit le LLM évaluateur (Groq)."""
    return LangchainLLMWrapper(
        ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0,
        )
    )


def build_evaluator_embeddings():
    """Construit les embeddings pour l'évaluation."""
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )


def main():
    logger.info("🔍 Démarrage de l'évaluation RAGAS — Assistant Juridique Sénégal")

    # 1. Exécuter le RAG sur toutes les questions
    questions = []
    answers = []
    contexts = []
    references = []

    for i, test in enumerate(TEST_SET, 1):
        logger.info(f"Question {i}/{len(TEST_SET)} : {test['question'][:60]}...")

        result = answer_question(test["question"], k=5)

        questions.append(test["question"])
        answers.append(result["answer"])
        contexts.append(result["context_used"])
        references.append(test["reference"])

    # 2. Construire le dataset RAGAS
    samples = []
    for i in range(len(questions)):
        samples.append(SingleTurnSample(
            user_input=questions[i],
            response=answers[i],
            retrieved_contexts=contexts[i],
            reference=references[i],
        ))

    dataset = EvaluationDataset(samples=samples)

    # 3. Définir les métriques
    llm = build_evaluator_llm()
    embeddings = build_evaluator_embeddings()

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    # 4. Lancer l'évaluation
    logger.info("📊 Calcul des métriques...")
    result = evaluate(dataset=dataset, metrics=metrics)

    # 5. Afficher et sauvegarder les résultats
    df = result.to_pandas()

    print("\n" + "=" * 60)
    print("📊 RÉSULTATS DE L'ÉVALUATION RAGAS — JURIDIQUE")
    print("=" * 60)

    summary = {}
    for col in df.columns:
        if col not in ["user_input", "response", "retrieved_contexts", "reference"]:
            mean_score = df[col].mean()
            summary[col] = round(float(mean_score), 4)
            print(f"  {col:25s} : {mean_score:.4f}")

    # 6. Calcul du taux de refus correct (out-of-scope)
    out_of_scope_indices = [i for i, t in enumerate(TEST_SET) if "Je n'ai pas cette information" in t["reference"]]
    if out_of_scope_indices:
        refusals = sum(
            1 for i in out_of_scope_indices
            if "Je n'ai pas cette information" in answers[i]
            or "consultez un professionnel" in answers[i].lower()
            or "pas dans mes documents" in answers[i].lower()
        )
        taux_refus = refusals / len(out_of_scope_indices)
        summary["taux_refus_correct"] = round(taux_refus, 4)
        print(f"  {'taux_refus_correct':25s} : {taux_refus:.4f}  ({refusals}/{len(out_of_scope_indices)})")

    # 7. Sauvegarder dans un fichier JSON
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eval_juridique_{timestamp}.json"

    output_data = {
        "timestamp": timestamp,
        "domaine": "juridique_senegal",
        "summary": summary,
        "per_sample": df.to_dict(orient="records"),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 Résultats sauvegardés : {output_file}")

    return summary


if __name__ == "__main__":
    main()