"""Script de test du loader."""
from src.ingestion.loader import load_all_pdfs
docs = load_all_pdfs()

for doc in docs[:3]:
    print("=" * 60)
    print(f"Source : {doc.metadata['source']}")
    print(f"Page   : {doc.metadata['page']}")
    print(f"Longueur : {len(doc.page_content)} caractères")
    print("-" * 60)
    print(doc.page_content[:300])
    print()