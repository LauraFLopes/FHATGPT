import os
import streamlit as st
import google.generativeai as genai
import fitz
from sentence_transformers import SentenceTransformer
import chromadb

# --- KONFIGURATION ---
API_KEY = st.secrets.get("GEMINI_API_KEY", None)
if not API_KEY:
    st.error("Gemini API key not found. Bitte setze GEMINI_API_KEY in den Streamlit Secrets.")
    st.stop()

@st.cache_resource # effizienter, da die Info im Cache gespeichert wird
def load_gemini_model(api_key, model_name):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

@st.cache_resource
def get_chroma_client(path):
    return chromadb.PersistentClient(path=path)

@st.cache_resource
def get_pdf_knowledge_collection(_client):
    return _client.get_or_create_collection("pdf_knowledge")

@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

# --- FUNKTION ZUM VERARBEITEN EINER PDF-DATEI ---
def process_pdf(pdf_path, embedding_model):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        st.error(f"Fehler beim Lesen von {pdf_path}: {e}")
        return None

    chunks = text.split("\n\n")
    embeddings = embedding_model.encode(chunks)
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_id, "chunk": i} for i in range(len(chunks))]

    return chunks, embeddings, ids, metadatas

# --- FUNKTION ZUM LADEN UND VERARBEITEN ALLER PDF-DATEIEN ---
def load_and_process_pdfs(pdf_folder, embedding_model, collection):
    all_chunks = []
    all_embeddings = []
    all_ids = []
    all_metadatas = []

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            result = process_pdf(pdf_path, embedding_model)
            if result:
                chunks, embeddings, ids, metadatas = result
                all_chunks.extend(chunks)
                all_embeddings.extend(embeddings)
                all_ids.extend(ids)
                all_metadatas.extend(metadatas)

    if all_embeddings:
        collection.add(
            embeddings=all_embeddings,
            documents=all_chunks,
            ids=all_ids,
            metadatas=all_metadatas
        )
        st.success(f"✅ {len(all_embeddings)} Text-Chunks aus {len(os.listdir(pdf_folder))} PDF-Dateien verarbeitet und in die Vektordatenbank geladen.")
    else:
        st.warning("⚠️ Keine PDF-Dateien zum Verarbeiten gefunden.")

# --- relevantesten Text-Chunks aus der ChromaDB-Sammlung
def get_relevant_chunks(question, collection, embedding_model, n_results=3):
    """Ruft die relevantesten Text-Chunks aus der Vektordatenbank basierend auf der Frage ab."""
    question_embedding = embedding_model.encode([question])
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results
    )
    relevant_chunks = results['documents'][0] if results and results['documents'] else []
    return relevant_chunks

# --- FUNKTION ZUM BEANTWORTEN VON FRAGEN MIT CHATVERLAUF ---
def ask_chatbot(question, chat_history, model, collection, embedding_model):
    """Beantwortet die Frage des Nutzers unter Berücksichtigung des relevanten Kontexts aus der Vektordatenbank."""
    relevant_chunks = get_relevant_chunks(question, collection, embedding_model)
    context = "\n\n".join(relevant_chunks)

    # Erstelle den Prompt mit Kontext
    prompt_with_context = f"Beantworte die folgende Frage basierend auf den bereitgestellten Informationen:\n\n{context}\n\nFrage: {question}\n\nAntwort:"

    context_messages = []
    for message in chat_history:
        if message["role"] != "system":
            context_messages.append({"role": message["role"], "parts": [message["content"]]})

    context_messages.append({"role": "user", "parts": [prompt_with_context]})

    try:
        response = model.generate_content(contents=context_messages)
        return response.text
    except Exception as e:
        st.error(f"Fehler bei der Gemini-Anfrage mit Kontext: {e}")
        return None

# --- STREAMLIT UI ---
st.title("Chatbot der FH Wedel")

# Initialisiere Session State
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [{"role": "assistant", "content": "Hallo! Ich bin dein Chatbot für Fragen rund um die FH Wedel. Frag mich alles, was du wissen möchtest!"}]

if "pdfs_processed" not in st.session_state:
    st.session_state.pdfs_processed = False

# Lade gecachte Ressourcen
model = load_gemini_model(API_KEY, "gemini-1.5-flash")
embedding_model = load_embedding_model('all-mpnet-base-v2')
client = get_chroma_client("./chroma_db")
collection = get_pdf_knowledge_collection(client)

# Button zum Laden und Verarbeiten der PDFs (wird nur einmal beim Start ausgeführt)
if not st.session_state.pdfs_processed:
    with st.spinner("Verarbeite PDF-Dateien..."):
        load_and_process_pdfs("./pdf_docs", embedding_model, collection)
        st.session_state.pdfs_processed = True

# Zeige den Chatverlauf
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Eingabefeld für die Benutzerfrage
user_question = st.chat_input("Stelle hier deine Frage zur Universität:")

if user_question:
    st.session_state["chat_history"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("Denke nach..."):
        # Rufe die angepasste ask_chatbot-Funktion auf und übergebe die Vektordatenbank und das Embedding-Modell
        answer = ask_chatbot(user_question, st.session_state["chat_history"], model, collection, embedding_model)
        if answer:
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)