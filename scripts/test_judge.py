"""Test isolé du LLM juge."""
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.utils.config import GROQ_API_KEY


JUDGE_MODEL = "qwen/qwen3.8-27b"   # ← LE changement

llm = ChatGroq(
    model=JUDGE_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=10,
)

prompt = ChatPromptTemplate.from_template(
    "La réponse répond-elle à la question ?\n\n"
    "QUESTION: {question}\nRÉPONSE: {answer}\n\n"
    "Réponds uniquement 'OUI' ou 'NON'."
)

chain = prompt | llm

# Test 1 : réponse pertinente (attendu OUI)
r1 = chain.invoke({
    "question": "Qui est éligible à la bourse ?",
    "answer": "Les bacheliers des séries S1, S3, S4, S5, T1, T2, F6, ainsi que les orphelins et handicapés.",
})
print(f"Test 1 (attendu OUI) : {r1.content!r}")

# Test 2 : réponse hors-sujet (attendu NON)
r2 = chain.invoke({
    "question": "Qui est éligible à la bourse ?",
    "answer": "Le Sénégal est un pays d'Afrique de l'Ouest.",
})
print(f"Test 2 (attendu NON) : {r2.content!r}")

# Test 3 : refus correct (attendu OUI)
r3 = chain.invoke({
    "question": "Quel est le montant pour la médecine ?",
    "answer": "Je ne trouve pas cette information dans mes documents.",
})
print(f"Test 3 (attendu OUI - refus correct) : {r3.content!r}")