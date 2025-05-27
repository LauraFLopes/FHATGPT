import os
import time
import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
import hashlib

# ========================
# HILFSFUNKTIONEN
# ========================

# Misst die Ausführungszeit eines Funktionsaufrufs und gibt sie aus.
def timed_step(name, func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    st.write(f"**{name}** in {end - start:.2f} Sekunden.")
    return result

# Erzeugt einen eindeutigen Hash für eine Datei.
# Dies wird verwendet, um schnell zu erkennen, ob sich eine PDF-Datei seit der letzten
# Verarbeitung geändert hat.
def get_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

# ========================
# MODELL- UND DATENBANK-SETUP
# ========================

# Lädt das generative Gemini-Modell von Google mit API-Key und Modellnamen.
@st.cache_resource
def load_gemini_model(api_key, model_name):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

# Lädt ein SentenceTransformer-Modell für Embedding-Zwecke
# (Text wird in numerischen Darstellung umgewandelt → schneller als Text).
@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

# Erstellt oder lädt einen persistenten (wird auf Festplatte gespeichert und geht
# somit nicht verloren) ChromaDB-Client zur Vektorspeicherung.
@st.cache_resource
def get_chroma_client(path):
    return chromadb.PersistentClient(path=path)

# Erstellt oder holt eine Collection (Datenbanktabelle) für PDF-Inhalte.
@st.cache_resource
def get_pdf_knowledge_collection(_client):
    return _client.get_or_create_collection("pdf_knowledge")

# ========================
# PDF VERARBEITUNG
# ========================

# Zerlegt eine PDF-Datei in Text-Chunks, berechnet Embeddings und Metadaten.
@st.cache_data(show_spinner=False, max_entries=10)
def process_pdf(pdf_path, _embedding_model):
    file_hash = get_file_hash(pdf_path)
    text = ""

    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        st.error(f"Fehler beim Lesen von {pdf_path}: {e}")
        return None

    # Zerlege Text in Abschnitte (Chunks).
    chunks = text.split("\n\n")

    # Berechne Embeddings für jeden Chunk
    embeddings = _embedding_model.encode(chunks)

    # Generiert die IDs und Metadaten für die Speicherung in ChromaDB.
    # Jeder Chunk benötigt eine eindeutige ID und zusätzliche beschreibende Metadaten.
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0] # Eindeutige ID für das gesamte PDF-Dokument (Dateiname ohne Endung).
    ids = [f"{doc_id}-{i}" # Eindeutige ID für jeden einzelnen Chunk (Dokumenten-ID + Chunk-Nummer).
                for i in range(len(chunks))
                ]
    # Zusätzliche Informationen für jeden Chunk, z.B. Herkunft und Dateihash für Updates.
    metadatas = [{"source": doc_id, "chunk": i, "file_hash": file_hash} for i in range(len(chunks))]

    return chunks, embeddings, ids, metadatas

# Vergleicht bestehende PDF-Datenbank mit Ordnerinhalt, aktualisiert nur neue/geänderte Dateien.
def update_pdfs_in_collection(pdf_folder, embedding_model, collection):
    # Lade vorhandene Dokument-Hashes aus der DB
    existing = collection.get(include=["metadatas"])
    existing_hashes = {
        doc_id.split("-")[0]: metadata.get("file_hash", None)
        for doc_id, metadata in zip(existing["ids"], existing["metadatas"])
    }

    # Lese aktuelle Dateien und berechne neue Hashes
    current_hashes = {
        os.path.splitext(f)[0]: get_file_hash(os.path.join(pdf_folder, f))
        for f in os.listdir(pdf_folder)
        if f.endswith(".pdf")
    }

    # Bestimme, welche Dateien neu/aktualisiert oder gelöscht wurden
    to_add = [doc for doc, h in current_hashes.items()
              if doc not in existing_hashes or existing_hashes[doc] != h]
    to_remove = [doc for doc in existing_hashes if doc not in current_hashes]

    # Entferne veraltete Einträge
    if to_remove:
        collection.delete(where={"source": {"$in": to_remove}})

    # Füge neue/aktualisierte Dateien hinzu
    for doc_id in to_add:
        pdf_path = os.path.join(pdf_folder, f"{doc_id}.pdf")
        result = timed_step(f"{doc_id} verarbeiten", process_pdf, pdf_path, embedding_model)
        if result:
            chunks, embeddings, ids, metadatas = result
            collection.delete(where={"source": doc_id})
            collection.add(embeddings=embeddings, documents=chunks, ids=ids, metadatas=metadatas)

    return len(to_add), len(to_remove)

# ========================
# FRAGEN & ANTWORTEN
# ========================

# Ermittelt die relevantesten Text-Abschnitte zur Beantwortung einer Frage.
def get_relevant_chunks(question, collection, embedding_model, n_results=3):
    start = time.time()
    question_embedding = embedding_model.encode([question])
    embedding_time = time.time()

    results = collection.query(query_embeddings=question_embedding, n_results=n_results)
    query_time = time.time()

    documents = results['documents'][0] if results and results['documents'] else []

    st.write(f"**Embedding der Frage:** {embedding_time - start:.2f} Sekunden.")
    st.write(f"**Vektor-Suche (ChromaDB):** {query_time - embedding_time:.2f} Sekunden.")
    return documents

# Generiert eine Antwort mithilfe der gefundenen Dokument-Chunks und Chatverlauf.
def ask_chatbot(question, chat_history, model, collection, embedding_model):
    start = time.time()
    relevant_chunks = get_relevant_chunks(question, collection, embedding_model)
    context = "\n\n".join(relevant_chunks)

    # TODO: hier Prompt
    messages = []

    # Kontext als Assistant-Nachricht (klingt natürlicher als System-Prompt).
    messages.append({
        "role": "assistant",
        "parts": [f"Hier sind relevante Informationen aus Dokumenten:\n\n{context}"]
    })

    # Bisherige Konversation anhängen.
    messages.extend([{"role": m["role"], "parts": [m["content"]]} for m in chat_history])

    # Neue Nutzerfrage anhängen.
    messages.append({"role": "user", "parts": [question]})

    # Antwort generieren lassen.
    llm_start = time.time()
    try:
        response = model.generate_content(contents=messages)
        llm_end = time.time()
        st.write(f"**Antwort vom Sprachmodell:** {llm_end - llm_start:.2f} Sekunden.")
        st.write(f"**Gesamtdauer der Anfrage:** {llm_end - start:.2f} Sekunden.")
        return response.text
    except Exception as e:
        st.error(f"Fehler bei der Antwortgenerierung: {e}")
        return None

# ========================
# STREAMLIT UI
# ========================

st.title("Chatbot der FH Wedel")

# API Key prüfen
API_KEY = st.secrets.get("GEMINI_API_KEY", None)
if not API_KEY:
    st.error("Gemini API key not found. Bitte setze GEMINI_API_KEY in den Streamlit Secrets.")
    st.stop()

# Chat-Historie initialisieren
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [{
        "role": "assistant",
        "content": "Hallo! Ich bin dein Chatbot für Fragen rund um die FH Wedel. Frag mich alles, was du wissen möchtest!"
    }]

if "pdfs_processed" not in st.session_state:
    st.session_state.pdfs_processed = False

# Modelle und Datenbank mit Zeitmessung laden
model = timed_step("Modell 'gemini-1.5-flash' geladen", load_gemini_model, API_KEY, "gemini-1.5-flash")
embedding_model = timed_step("Embedding-Modell 'all-mpnet-base-v2' geladen", load_embedding_model, 'all-mpnet-base-v2')
client = timed_step("ChromaDB-Client geladen", get_chroma_client, "./chroma_db")
collection = timed_step("ChromaDB-Sammlung geladen", get_pdf_knowledge_collection, client)

# PDFs nur einmal verarbeiten
if not st.session_state.pdfs_processed:
    with st.spinner("PDFs werden verarbeitet..."):
        added, removed = timed_step("Update PDFs in Collection", update_pdfs_in_collection, "./pdf_docs", embedding_model, collection)
        st.success(f"{added} PDFs hinzugefügt/aktualisiert, {removed} entfernt.")
        st.session_state.pdfs_processed = True
else:
    st.success(f"Datenbank enthält {collection.count()} Einträge.")

# Chat-Historie anzeigen
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Frage eingeben
user_question = st.chat_input("Stelle hier deine Frage zur FH Wedel:")

if user_question:
    st.session_state["chat_history"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("Denke nach..."):
        answer = ask_chatbot(user_question, st.session_state["chat_history"], model, collection, embedding_model)
        if answer:
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
