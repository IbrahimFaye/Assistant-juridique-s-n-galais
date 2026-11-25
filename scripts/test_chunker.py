"""Script de test du chunker."""
from src.ingestion.loader import load_all_pdfs
from src.ingestion.chunker import chunk_documents

docs = load_all_pdfs()
chunks = chunk_documents(docs)

print(f"\n{len(chunks)} chunks créés\n")

for chunk in chunks[:3]:
    print("=" * 60)
    print(f"ID     : {chunk.metadata['chunk_id']}")
    print(f"Source : {chunk.metadata['source']} - page {chunk.metadata['page']}")
    print(f"Taille : {len(chunk.page_content)} caractères")
    print("-" * 60)
    print(chunk.page_content)
    print()