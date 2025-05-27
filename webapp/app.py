from chatbot_logic import *

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
update_pdfs_in_collection(pdf_folder, embedding_model, collection)
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

'''''
http://127.0.0.1:5000

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

'''''