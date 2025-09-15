"""
Dieses Modul initialisiert die Flask-Webanwendung und definiert alle HTTP-Endpunkte 
für den Chatbot sowie die Admin-Schnittstellen.
"""
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import toml
from openai import OpenAI
from chatbot_logic import load_assistant, ask_assistant
from sync_scheduler import integrate_with_flask
from functools import wraps
import secrets as py_secrets
import json
import os
import time


app = Flask(__name__)

# Secrets laden
secrets = toml.load(".secrets/secrets.toml")

# Admin-Token generieren oder aus secrets laden
if "ADMIN_TOKEN" not in secrets:
    # Generiere ein Token beim ersten Start
    secrets["ADMIN_TOKEN"] = py_secrets.token_urlsafe(32)
    with open(".secrets/secrets.toml", "w") as f:
        toml.dump(secrets, f)
    print(f"🔐 Admin-Token generiert: {secrets['ADMIN_TOKEN']}")
    print("   Speichere dieses Token sicher!")
else:
    print("🔐 Admin-Token geladen")

ADMIN_TOKEN = secrets["ADMIN_TOKEN"]

# OpenAI Client initialisieren
client = OpenAI(api_key=secrets.get("API_KEY"))
if not secrets.get("API_KEY"):
    raise ValueError("❌ Kein API-Schlüssel gefunden. Setze 'API_KEY' in secrets.toml")

# Model festlegen
model = "gpt-4o"

# Assistant laden
print("🚀 Initialisiere FH Wedel Chatbot...")
assistant = load_assistant(model, secrets, client)

# Chatverlauf
chat_history = [{"role": "assistant", "content": "Hallo! Ich bin dein Chatbot für Fragen rund um die FH Wedel. Frag mich einfach!"}]

# Thread-ID speichern (pro Session ein Thread)
thread_id = None


"""
Initialisiert das Overlay für den Chatbot.

@return Overlay für den Chatbot.
"""
@app.route("/")
def index():
    return render_template("index.html", chat_history=chat_history)


"""
HTTP-Request für das Zurücksetzen des Chatverlaufs und des OpenAi-Threads.

@return Die Startnachricht für den Chat.
"""
@app.route("/reset", methods=["POST"])
def reset_chat():
    global chat_history, thread_id
    
    initial_message = "Hallo! Ich bin dein Chatbot für Fragen rund um die FH Wedel. Frag mich einfach!"
    chat_history = [{"role": "assistant", "content": initial_message}]
    
    # Neuen Thread für neue Konversation erstellen
    thread_id = None
    
    # Sende die Willkommensnachricht zurück
    return jsonify({"message": initial_message}), 200


"""
HTTP-Request für das Abschicken einer User-Nachricht.
"""
@app.route("/ask", methods=["POST"])
def ask():
    global thread_id
    
    # Fehlermeldung, falls keine Frage eingegeben
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"answer": "❗ Bitte gib eine Frage ein."}), 400

    # Hinzufügen der Nachricht des Users an den Chatverlauf
    print(f"💬 Frage: {user_question}")
    chat_history.append({"role": "user", "content": user_question})

    # Prüfe ob Streaming gewünscht ist
    stream = request.json.get("stream", False)
    
    try:
        # Thread erstellen, falls noch nicht vorhanden
        if not thread_id:
            thread = client.beta.threads.create()
            thread_id = thread.id
            print(f"📝 Neuer Thread erstellt: {thread_id}")
        
        if stream:
            # Streaming Response
            def generate():
                full_answer = ""
                try:
                    # Sende Start-Signal
                    yield f"data: {json.dumps({'type': 'start'})}\n\n"
                    
                    # Stream vom Assistant
                    for chunk in ask_assistant(user_question, assistant.id, thread_id, client):
                        full_answer += chunk
                        # Server-Sent Events Format
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                    
                    # Speichere vollständige Antwort im Chat-Verlauf
                    chat_history.append({"role": "assistant", "content": full_answer})
                    
                    # Sende End-Signal
                    yield f"data: {json.dumps({'type': 'end'})}\n\n"
                    
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            
            return Response(
                stream_with_context(generate()),
                mimetype="text/event-stream",
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'  # Disable Nginx buffering
                }
            )
        else:
            # Normale Response
            answer = ""
            for chunk in ask_assistant(user_question, assistant.id, thread_id, client):
                answer += chunk
            
            if not answer:
                answer = "❗ Entschuldigung, ich konnte keine Antwort generieren."
            
            chat_history.append({"role": "assistant", "content": answer})
            return jsonify({"answer": answer})
    
    except Exception as e:
        print(f"❗ Fehler: {e}")
        answer = f"Fehler: {str(e)}"
        return jsonify({"answer": answer}), 500


"""
Kümmert sich um die Identifizierung der einzelnen HTTP-Requests, die eine Admin-Indentifizierung benötigen.

@return Bestätigt die Admin-Identifikation
"""
# Admin-Authentifizierung Decorator
def require_admin_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Token aus Header oder Query-Parameter
        token = request.headers.get('X-Admin-Token') or request.args.get('token')
        
        if not token:
            return jsonify({"error": "Keine Berechtigung"}), 401
        
        if token != ADMIN_TOKEN:
            return jsonify({"error": "Ungültiger Token"}), 403
            
        return f(*args, **kwargs)
    return decorated_function


"""
HTTP-Request für den Umgang mit dem Synchronisationsvorgang der lokalen-Dateien und der Dateien im OpenAi-Cloud-Vektorspeicher.
Benötigt Admin-Indentifikation.

@return Angabe, ob die Synchronisation erfolgreich war.
"""
@app.route("/admin/sync", methods=["POST"])
@require_admin_token
def manual_sync():
    # Prüfe ob Streaming gewünscht ist
    stream = request.json.get("stream", False) if request.is_json else False
    
    if stream:
        # Streaming Response mit Progress Updates
        def generate():
            progress_queue = []
            
            def progress_callback(update):
                progress_queue.append(update)
            
            try:
                from vector_store_manager import VectorStoreManager
                manager = VectorStoreManager(client=client, secrets=secrets)
                
                # Starte Sync in separatem Thread
                import threading
                sync_result = [None]
                
                def run_sync():
                    sync_result[0] = manager.sync_vector_stores(progress_callback)
                
                sync_thread = threading.Thread(target=run_sync)
                sync_thread.start()
                
                # Sende Progress Updates
                while sync_thread.is_alive() or progress_queue:
                    if progress_queue:
                        update = progress_queue.pop(0)
                        yield f"data: {json.dumps(update)}\n\n"
                    else:
                        time.sleep(0.1)
                
                # Finales Update
                if sync_result[0]:
                    # Assistant neu laden
                    global assistant
                    assistant = load_assistant(model, secrets, client)
                    yield f"data: {json.dumps({'message': '✅ Synchronisation abgeschlossen - Assistant neu geladen', 'status': 'success', 'complete': True})}\n\n"
                else:
                    yield f"data: {json.dumps({'message': '✅ Synchronisation abgeschlossen - keine Änderungen', 'status': 'success', 'complete': True})}\n\n"
                    
            except Exception as e:
                yield f"data: {json.dumps({'message': f'❌ Fehler: {str(e)}', 'status': 'error', 'complete': True})}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
    else:
        # Normale Response
        try:
            from vector_store_manager import VectorStoreManager
            manager = VectorStoreManager(client=client, secrets=secrets)
            has_changes = manager.sync_vector_stores()
            
            if has_changes:
                # Assistant neu laden um neuen Vector Store zu verwenden
                global assistant
                assistant = load_assistant(model, secrets, client)
                return jsonify({
                    "status": "sync completed", 
                    "changes_found": True,
                    "message": "Vector Store wurde aktualisiert und Assistant neu geladen."
                }), 200
            else:
                return jsonify({
                    "status": "sync completed", 
                    "changes_found": False,
                    "message": "Keine Änderungen gefunden."
                }), 200
        except Exception as e:
            print(f"❌ Fehler bei manueller Synchronisation: {e}")
            return jsonify({"error": str(e)}), 500


"""
HTTP-Request um den Status des OpenAi-Cloud-Vektorspeichers zurückzugeben.
Benötigt Admin-Identifikation.

@return Den Status im JSON-Format.
"""
@app.route("/admin/status", methods=["GET"])
@require_admin_token
def sync_status():
    """Status des Vector Stores abrufen."""
    try:
        from vector_store_manager import VectorStoreManager
        manager = VectorStoreManager(client=client, secrets=secrets)
        status = manager.get_sync_status()
        
        # Füge Assistant-Info hinzu
        status["assistant_id"] = assistant.id if assistant else None
        
        return jsonify(status), 200
    except Exception as e:
        print(f"❌ Fehler beim Abrufen des Status: {e}")
        return jsonify({"error": str(e)}), 500


"""
HTTP-Request um das komplette System zurückzusetzen und beendet die App.
Benötigt Admin-Identifikation.

@return Bestätigung, ob es geklappt hat.
"""
@app.route("/admin/reset", methods=["POST"])
@require_admin_token
def reset_system():
    """Setzt das gesamte System zurück und beendet die App."""
    # Zusätzliche Sicherheitsabfrage über Parameter
    confirm = request.json.get("confirm", False) if request.is_json else False
    
    if not confirm:
        return jsonify({
            "error": "Bestätigung erforderlich",
            "message": "Sende {'confirm': true} um den Reset zu bestätigen.",
            "warning": "DIES LÖSCHT ALLE VECTOR STORES, FILES UND EINSTELLUNGEN!"
        }), 400
    
    try:
        from vector_store_manager import VectorStoreManager
        manager = VectorStoreManager(client=client, secrets=secrets)
        
        # Führe Reset durch
        success = manager.reset_all(keep_admin_token=True)
        
        if success:
            # Response senden bevor wir die App beenden
            response = jsonify({
                "status": "reset completed",
                "message": "System wurde vollständig zurückgesetzt. Die App wird beendet.",
                "warning": "Die App wird in 2 Sekunden beendet!"
            })
            
            # Shutdown-Funktion in separatem Thread starten
            def shutdown_server():
                import time
                time.sleep(2)  # Warte 2 Sekunden damit die Response gesendet wird
                print("\n🛑 Beende App nach System-Reset...")
                os._exit(0)  # Harter Exit
            
            import threading
            shutdown_thread = threading.Thread(target=shutdown_server)
            shutdown_thread.daemon = True
            shutdown_thread.start()
            
            return response, 200
        else:
            return jsonify({
                "error": "Reset fehlgeschlagen"
            }), 500
            
    except Exception as e:
        print(f"❌ Fehler beim System-Reset: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


"""
Intitialisierung des Overlays der Admin-Seite.

@return Overlay für die Admin-Seite
"""
@app.route("/admin", methods=["GET"])
def admin_panel():
    """Admin Web-Interface (Token-Eingabe erfolgt im Frontend)."""
    return render_template("admin.html")

# Startet die Flask-App und initialisiert den automatischen Synchronisations-Scheduler.
if __name__ == "__main__":
    # Initialisierung des Dateien-Synchronisations-Schedulers.
    try:
        scheduler = integrate_with_flask(client, secrets)
        print("✅ Automatische Synchronisation aktiviert")
    except Exception as e:
        print(f"⚠️  Sync-Scheduler konnte nicht gestartet werden: {e}")
        print("    Die App läuft trotzdem, aber ohne automatische Synchronisation.")
    
    app.run(
        port=5000, 
        debug=False,  # Kein Auto-Reload von Python-Dateien
        use_reloader=False  # Explizit deaktivieren
    )
