"""
Évaluation du système RAG avec RAGAS.
Usage : python -m scripts.evaluate_rag
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


# === Jeu de test : 15 questions avec réponses de référence ===
TEST_SET = [
    {
        "question": "Qui est éligible à la bourse ?",
        "reference": "Les bacheliers des séries S1, S3, S4, S5, T1, T2, F6 (demi-bourse), S2, S2A, G, STEG, L, les orphelins, les handicapés physiques et les malades chroniques.",
    },
    {
        "question": "Quand obtient-on une bourse d'excellence ?",
        "reference": "Les bacheliers de l'année en cours inscrits, ayant obtenu la mention Très Bien, Bien, ou étant lauréat du concours général.",
    },
    {
        "question": "Quel est le montant de la bourse pour un étudiant en médecine ?",
        "reference": "Je ne trouve pas cette information dans mes documents.",
    },
    {
        "question": "Quand débutera le paiement des bourses de septembre 2026 ?",
        "reference": "Le 07 septembre 2026.",
    },
    {
        "question": "Quels sont les critères pour une bourse sociale ?",
        "reference": "Être orphelin, enfant de parents indigents, handicapé physique ou atteint de maladie chronique.",
    },
    {
        "question": "Comment renouveler sa bourse ?",
        "reference": "Le renouvellement dépend du cycle d'étude et de l'âge (moins de 30 ans), avec un redoublement autorisé.",
    },
    {
        "question": "Quelle est la capitale du Sénégal ?",
        "reference": "Je ne trouve pas cette information dans mes documents.",
    },
    {
        "question": "Quelles sont les séries de bac qui donnent droit à une demi-bourse ?",
        "reference": "Les séries S1, S3, S4, S5, T1, T2 et F6.",
    },
    {
        "question": "Y a-t-il une bourse pour les étudiants handicapés ?",
        "reference": "Oui, une bourse sociale est octroyée aux handicapés physiques.",
    },
    {
        "question": "Quel est le numéro de téléphone de la Direction des Bourses ?",
        "reference": "Le 33 960 11 11.",
    },
    # Ajoute 5 questions supplémentaires selon tes documents
]


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
    logger.info("🔍 Démarrage de l'évaluation RAGAS")
    
    # 1. Exécuter le RAG sur toutes les questions
    questions = []
    answers = []
    contexts = []
    references = []
    
    for i, test in enumerate(TEST_SET, 1):
        logger.info(f"Question {i}/{len(TEST_SET)} : {test['question'][:50]}...")
        
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
    print("📊 RÉSULTATS DE L'ÉVALUATION RAGAS")
    print("=" * 60)
    
    summary = {}
    for col in df.columns:
        if col not in ["user_input", "response", "retrieved_contexts", "reference"]:
            mean_score = df[col].mean()
            summary[col] = round(float(mean_score), 4)
            print(f"  {col:25s} : {mean_score:.4f}")
    
    # 6. Sauvegarder dans un fichier JSON
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eval_{timestamp}.json"
    
    output_data = {
        "timestamp": timestamp,
        "summary": summary,
        "per_sample": df.to_dict(orient="records"),
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 Résultats sauvegardés : {output_file}")
    
    return summary


if __name__ == "__main__":
    main()