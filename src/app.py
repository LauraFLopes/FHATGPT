import os
from flask import Flask, render_template, request, jsonify
from chatbot_logic import *
import toml

app = Flask(__name__)

# --- Lade Secrets ---
secrets = toml.load(".streamlit/secrets.toml")
API_KEY = secrets.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌ Kein API-Schlüssel gefunden. Setze 'GEMINI_API_KEY' in secrets.toml")

# --- Initialisiere Modelle & Datenbank ---
model = load_gemini_model(API_KEY, "gemini-1.5-flash")
embedding_model = load_embedding_model("all-mpnet-base-v2")
client = get_chroma_client("./chroma_db")
collection = get_pdf_knowledge_collection(client)

# --- PDFs bei Start verarbeiten ---
pdf_folder = "./pdf_docs"
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)

print("📁 Scanne PDF-Verzeichnis...")
print("📄 Dateien:", os.listdir(pdf_folder))
load_and_process_pdfs(pdf_folder, embedding_model, collection)
print(f"✅ ChromaDB enthält nun {collection.count()} Chunks")

# --- Chatverlauf (im RAM, einfach gehalten) ---
chat_history = [{"role": "assistant", "content": "Hallo! Ich bin dein Chatbot für Fragen rund um die FH Wedel. Frag mich einfach!"}]

@app.route("/")
def index():
    return render_template("index.html", chat_history=chat_history)

@app.route("/reset", methods=["POST"])
def reset_chat():
    global chat_history
    chat_history = [{"role": "assistant", "content": "Hallo! Ich bin dein Chatbot für Fragen rund um die FH Wedel. Frag mich einfach!"}]
    return "", 204


@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"answer": "❗ Bitte gib eine Frage ein."}), 400

    print(f"💬 Frage: {user_question}")
    chat_history.append({"role": "user", "content": user_question})

    try:
        answer = ask_chatbot(user_question, chat_history, model, collection, embedding_model)
    except Exception as e:
        print(f"❗ Fehler: {e}")
        return jsonify({"answer": f"Fehler: {str(e)}"}), 500

    chat_history.append({"role": "bot", "content": answer})
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
