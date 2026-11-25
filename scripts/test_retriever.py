"""Test du retriever."""
from src.rag.retriever import search

questions = [
    "qui sont eligible à la bourse",
    "quand obtient-on une bourse d'excellence ?",
    "quand debutera le paiement des bourse pour le mois de septembre 2026",
]

for q in questions:
    print("\n" + "=" * 70)
    print(f"❓ {q}")
    print("=" * 70)
    
    results = search(q, k=3)
    
    for i, doc in enumerate(results, 1):
        print(f"\n--- Résultat {i} ---")
        print(f"Source : {doc.metadata['source']} (page {doc.metadata['page']})")
        print(f"Extrait : {doc.page_content[:250]}...")