"""
Benchmark : mesure le temps de réponse du LLM pur (sans RAG).
"""
import time
from src.rag.llm import get_llm


PROMPT_SIMPLE = "Réponds en une phrase : quelle est la capitale du Sénégal ?"


def main():
    print("🔧 Initialisation du LLM...")
    llm = get_llm()
    
    print("\n⏱️  Test 1 : premier appel (chargement du modèle)")
    start = time.time()
    response = llm.invoke(PROMPT_SIMPLE)
    t1 = time.time() - start
    print(f"   Temps : {t1:.1f}s")
    print(f"   Réponse : {response.content[:100]}")
    
    print("\n⏱️  Test 2 : deuxième appel (modèle en cache)")
    start = time.time()
    response = llm.invoke(PROMPT_SIMPLE)
    t2 = time.time() - start
    print(f"   Temps : {t2:.1f}s")
    
    print("\n⏱️  Test 3 : troisième appel")
    start = time.time()
    response = llm.invoke(PROMPT_SIMPLE)
    t3 = time.time() - start
    print(f"   Temps : {t3:.1f}s")
    
    print(f"\n📊 Résumé :")
    print(f"   Appel 1 (cold) : {t1:.1f}s")
    print(f"   Appel 2 (warm) : {t2:.1f}s")
    print(f"   Appel 3 (warm) : {t3:.1f}s")


if __name__ == "__main__":
    main()