"""
Interface chat Streamlit.
"""
import time
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/v1"

# === Fonction pour afficher le texte caractère par caractère ===
def stream_char_by_char(text: str, delay: float = 0.01):
    """Générateur qui produit le texte caractère par caractère avec une pause."""
    for char in text:
        yield char
        time.sleep(delay)

# === Configuration de la page ===
st.set_page_config(
    page_title="Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# === Sidebar ===
with st.sidebar:
    st.title("🎓 Droit Sénégalais")
    st.markdown(
        "Assistant virtuel pour le droit sénégalais"
        
    )
    st.divider()
    
    st.markdown("### 💡 Exemples de questions")
    exemples = [
        "Quel est le préavis pour un CDD ?",
        "Comment divorcer au Sénégal ?",
        "Qui hérite en cas de décès sans testament ?",
        "Quelle est la durée légale du travail ?",
    ]
    for ex in exemples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_question = ex
            st.rerun()
    
    st.divider()
    
    # État de l'API
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            st.success(f"✅ API OK ({data['vectorstore_size']} chunks)")
            st.caption(f"LLM : {data['llm_provider']}")
        else:
            st.error(f"⚠️ API status {r.status_code}")
    except Exception as e:
        st.error("❌ API inaccessible")
        st.caption("Lance l'API : `uvicorn src.api.main:app --reload`")
    
    st.divider()
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# === Header principal ===
st.title("Assistant Droit sénégalais")
st.caption(
    "Pose ta question sur le droit sénégalais "
    "Je réponds uniquement à partir des documents officiels."
)

# === Historique de conversation ===
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- `{src['source']}` (page {src['page']})")
        if msg.get("latency_ms"):
            st.caption(f"⏱️ {msg['latency_ms'] / 1000:.1f}s")


# === Saisie utilisateur ===
question = st.session_state.pop("pending_question", None)
if not question:
    question = st.chat_input("Pose ta question ici...")

if question:
    # Affiche la question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    # Affiche la réponse
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les documents..."):
            try:
                r = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question, "k": 5},
                    timeout=600,
                )
                r.raise_for_status()
                data = r.json()
            except requests.exceptions.Timeout:
                st.error("⏱️ Délai dépassé. Réessaie avec une question plus courte.")
                data = None
            except Exception as e:
                st.error(f"❌ Erreur : {e}")
                st.caption("Vérifie que l'API est bien lancée sur le port 8000.")
                data = None

        if data:
            # Affichage caractère par caractère
            response_text = st.write_stream(stream_char_by_char(data["answer"], delay=0.005))
            
            if data["sources"]:
                with st.expander("📚 Sources"):
                    for src in data["sources"]:
                        st.markdown(f"- `{src['source']}` (page {src['page']})")
            
            st.caption(f"⏱️ {data['latency_ms'] / 1000:.1f}s")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "sources": data["sources"],
                "latency_ms": data["latency_ms"],
            })