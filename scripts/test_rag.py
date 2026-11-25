"""Test du moteur RAG complet."""
import time
from src.rag.chain import answer_question


QUESTIONS = [
    "Qui est éligible à la bourse ?",
    "Quand obtient-on une bourse d'excellence ?",
    "Quand débutera le paiement des bourses de septembre 2026 ?",
    "Quel est le montant de la bourse pour un étudiant en médecine ?",
    "Quelle est la capitale du Sénégal ?",
]


def main():
    for question in QUESTIONS:
        print("\n" + "=" * 70)
        print(f"❓ {question}")
        print("=" * 70)
        
        start = time.time()
        result = answer_question(question)
        elapsed = time.time() - start
        
        print(f"\n💬 RÉPONSE ({elapsed:.1f}s) :\n")
        print(result["answer"])
        
        print(f"\n📚 Sources :")
        for src in result["sources"]:
            print(f"  • {src['source']} (page {src['page']})")
        
        print(f"\n⏱️  Temps total : {elapsed:.1f}s")


if __name__ == "__main__":
    main()