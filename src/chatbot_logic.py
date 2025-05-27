import os
import fitz
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import chromadb

# --- KONFIGURATION ---
def load_gemini_model(api_key, model_name):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def get_chroma_client(path):
    return chromadb.PersistentClient(path=path)

def get_pdf_knowledge_collection(_client):
    return _client.get_or_create_collection("pdf_knowledge")

def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

# --- VERBESSERTES CHUNKING ---
def process_pdf(pdf_path, embedding_model):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        raise ValueError(f"Fehler beim Lesen von {pdf_path}: {e}")

    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]

    # Robustere Chunk-Erstellung
    raw_chunks = [chunk.strip() for chunk in text.split("\n") if len(chunk.strip()) > 100]

    chunks = []
    current_chunk = ""
    for para in raw_chunks:
        if len(current_chunk) + len(para) < 500:
            current_chunk += " " + para
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())

    if not chunks:
        print(f"⚠️ Keine verwertbaren Chunks in {doc_id}")
        return None

    print(f"📄 {doc_id}: {len(chunks)} Chunks extrahiert.")

    embeddings = normalize(embedding_model.encode(chunks))
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_id, "chunk": i} for i in range(len(chunks))]

    return chunks, embeddings, ids, metadatas

# --- PDFS VERARBEITEN ---
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
        print(f"✅ {len(all_embeddings)} Text-Chunks verarbeitet.")
    else:
        print("⚠️ Keine PDF-Dateien mit verwertbarem Inhalt gefunden.")

# --- CHUNKS ABFRAGEN ---
def get_relevant_chunks(question, collection, embedding_model, n_results=3):
    question_embedding = normalize(embedding_model.encode([question]))
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results
    )
    relevant_chunks = results['documents'][0] if results and results['documents'] else []
    return relevant_chunks

# --- ANTWORT GENERIEREN ---
def ask_chatbot(question, chat_history, model, collection, embedding_model, max_chars=3000):
    relevant_chunks = get_relevant_chunks(question, collection, embedding_model)

    print(f"🔎 Frage: {question}")
    print(f"📄 Gefundene relevante Chunks: {len(relevant_chunks)}")

    # Kontext auf max. Länge beschränken
    context = ""
    for chunk in relevant_chunks:
        if len(context) + len(chunk) > max_chars:
            break
        context += chunk + "\n\n"

    prompt_with_context = f"Beantworte die folgende Frage basierend auf den bereitgestellten Informationen:\n\n{context}\n\nFrage: {question}\n\nAntwort:"

    context_messages = []
    for message in chat_history:
        role = message["role"]
        if role == "bot":
            role = "model"
        context_messages.append({"role": role, "parts": [message["content"]]})


    context_messages.append({"role": "user", "parts": [prompt_with_context]})

    try:
        response = model.generate_content(contents=context_messages)
        return response.text
    except Exception as e:
        raise RuntimeError(f"Fehler bei der Gemini-Anfrage: {e}")