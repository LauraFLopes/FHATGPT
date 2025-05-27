import os
import time
import google.generativeai as genai
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
import hashlib
import re
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ========================
# HILFSFUNKTIONEN
# ========================

# Misst die Ausführungszeit eines Funktionsaufrufs und gibt sie aus.
def timed_step(name, func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"**{name}** in {end - start:.2f} Sekunden.")
    return result

# Erzeugt einen eindeutigen Hash für eine Datei.
# Dies wird verwendet, um schnell zu erkennen, ob sich eine PDF-Datei seit der letzten
# Verarbeitung geändert hat.
def get_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

# Teilt einen gegebenen Markdown-Text anhand von Überschriften (## etc.) in logische Abschnitte ("Chunks").
# Wenn ein Chunk zu lang wird (basierend auf max_length), wird er unabhängig von Überschriften ebenfalls geteilt.
# Dies ist nützlich für semantisches Chunking bei Dokumenten mit Markdown-Struktur für Modulhandbücher.
def split_text_by_markdown_headings(text, max_length=1000):
    """
    Teilt Markdown-Text in semantische Abschnitte basierend auf Überschriften.
    Ein Abschnitt (Chunk) endet bei einer neuen Überschrift oder wenn die maximale Länge überschritten wird.
    """
    header_pattern = re.compile(r"^#{1,6}\s+.*")
    chunks = []
    current_chunk = []
    current_length = 0

    for line in text.splitlines():
        is_header = header_pattern.match(line)

        # Wenn Überschrift und aktueller Chunk nicht leer → Chunk speichern
        if is_header and current_chunk:
            chunks.append("\n".join(current_chunk).strip())
            current_chunk = [line]
            current_length = len(line)
            continue

        current_chunk.append(line)
        current_length += len(line)

        # Falls zu lang, auch ohne Header trennen
        if current_length > max_length:
            chunks.append("\n".join(current_chunk).strip())
            current_chunk = []
            current_length = 0

    # Letzten Chunk nicht vergessen
    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())

    return [chunk for chunk in chunks if chunk]

# ========================
# MODELL- UND DATENBANK-SETUP
# ========================

# Lädt das generative Gemini-Modell von Google mit API-Key und Modellnamen.
def load_gemini_model(api_key, model_name):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

# Lädt ein SentenceTransformer-Modell für Embedding-Zwecke
# (Text wird in numerischen Darstellung umgewandelt → schneller als Text).
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

# Erstellt oder lädt einen persistenten (wird auf Festplatte gespeichert und geht
# somit nicht verloren) ChromaDB-Client zur Vektorspeicherung.
def get_chroma_client(path):
    return chromadb.PersistentClient(path=path)

# Erstellt oder holt eine Collection (Datenbanktabelle) für PDF-Inhalte.
def get_pdf_knowledge_collection(_client):
    return _client.get_or_create_collection("pdf_knowledge")

# ========================
# PDF VERARBEITUNG
# ========================

# Zerlegt eine PDF-Datei in Text-Chunks, berechnet Embeddings und Metadaten.
def process_pdf(pdf_path, _embedding_model):
    file_hash = get_file_hash(pdf_path)
    text = ""

    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Fehler beim Lesen von {pdf_path}: {e}")
        return None

    # Markdown-basiertes Chunking
    chunks = split_text_by_markdown_headings(text)

    # Berechne Embeddings für jeden Chunk
    embeddings = _embedding_model.encode(chunks)

    # Generiere IDs und Metadaten
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
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

    print(f"**Embedding der Frage:** {embedding_time - start:.2f} Sekunden.")
    print(f"**Vektor-Suche (ChromaDB):** {query_time - embedding_time:.2f} Sekunden.")
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
        print(f"**Antwort vom Sprachmodell:** {llm_end - llm_start:.2f} Sekunden.")
        print(f"**Gesamtdauer der Anfrage:** {llm_end - start:.2f} Sekunden.")
        return response.text
    except Exception as e:
        print(f"Fehler bei der Antwortgenerierung: {e}")
        return None
