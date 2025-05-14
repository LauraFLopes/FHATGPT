import os
import streamlit as st
import google.generativeai as genai
import fitz
from sentence_transformers import SentenceTransformer
import chromadb

# --- KONFIGURATION (bleibt gleich) ---
API_KEY = st.secrets.get("GEMINI_API_KEY", None)
if not API_KEY:
    st.error("Gemini API key not found. Bitte setze GEMINI_API_KEY in den Streamlit Secrets.")
    st.stop()
genai.configure(api_key=API_KEY)
GEMINI_MODEL = "gemini-1.5-flash"
model = genai.GenerativeModel(GEMINI_MODEL)
#test
# --- INITIALISIERUNG DER VEKTORDATENBANK (bleibt gleich) ---
CHROMA_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("pdf_knowledge")

# --- INITIALISIERUNG DES EMBEDDING-MODELS (bleibt gleich) ---
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# --- FUNKTION ZUM VERARBEITEN EINER PDF-DATEI ---
def process_pdf(pdf_path):
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
def load_and_process_pdfs(pdf_folder):
    all_chunks = []
    all_embeddings = []
    all_ids = []
    all_metadatas = []

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            result = process_pdf(pdf_path)
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

# --- FUNKTION ZUM BEANTWORTEN VON FRAGEN MIT CHATVERLAUF ---
def ask_chatbot(question, chat_history):
    # Extrahiere nur die Nachrichten des Benutzers und des Assistenten für den Kontext
    context_messages = []
    for message in chat_history:
        if message["role"] != "system":
            context_messages.append({"role": message["role"], "parts": [message["content"]]})

    # Füge die aktuelle Benutzerfrage hinzu
    context_messages.append({"role": "user", "parts": [question]})

    try:
        response = model.generate_content(contents=context_messages)
        return response.text
    except Exception as e:
        st.error(f"Fehler bei der Gemini-Anfrage mit Chatverlauf: {e}")
        return None

# --- STREAMLIT UI ---
st.title("Chatbot der FH Wedel")

# Initialisiere den Chatverlauf im Session State
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
    # Füge die Begrüßungsnachricht am Anfang hinzu
    st.session_state["chat_history"].append({"role": "assistant", "content": "Hallo! Ich bin dein Chatbot für Fragen rund um die FH Wedel. Frag mich alles, was du wissen möchtest!"})

# Zeige den Chatverlauf oben an
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Eingabefeld für die Benutzerfrage unten
user_question = st.chat_input("Stelle hier deine Frage zur Universität:")

if user_question:
    # Füge die aktuelle Benutzerfrage sofort zum Chatverlauf hinzu, damit sie angezeigt wird
    st.session_state["chat_history"].append({"role": "user", "content": user_question})
    # Zeige die Frage sofort im Chat
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("Denke nach..."):
        # Verwende die aktualisierte ask_chatbot-Funktion mit dem Chatverlauf
        answer = ask_chatbot(user_question, st.session_state["chat_history"])
        if answer:
            # Füge die Antwort des Assistenten zum Chatverlauf hinzu
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})

            # Zeige die Antwort des Assistenten im Chat
            with st.chat_message("assistant"):
                st.markdown(answer)

# Button zum Laden und Verarbeiten der PDFs (wird nur einmal beim Start ausgeführt)
if "pdfs_processed" not in st.session_state:
    with st.spinner("Verarbeite PDF-Dateien..."):
        load_and_process_pdfs("./pdf_docs")
        st.session_state.pdfs_processed = True
elif not st.session_state.get("pdfs_processed", False):
    with st.spinner("Verarbeite PDF-Dateien..."):
        load_and_process_pdfs("./pdf_docs")
        st.session_state.pdfs_processed = True