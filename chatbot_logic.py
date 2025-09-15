"""
Die Klasse kümmert sich um die Logik des Chatbots.
"""
import time
import toml


"""
Erstellt oder Lädt den OpenAI-Cloud-Vektorspeicher für diesen ChatBot.

@param secrets Die Datenbank mit den wichtigen Infos, wie die ID des Vektorspeichers. 
@param client Der OpenAI-Client auf dem der Chatbot läuft.
@return ID des Vektorspeichers.
"""
def load_or_create_vector_store(secrets, client):
    from vector_store_manager import VectorStoreManager

    manager = VectorStoreManager(client=client, secrets=secrets)
    vector_store_id, _ = manager.create_or_update_vector_store()
    
    return vector_store_id


"""
Erstellt oder lädt den Assistenten auf dem der Chatbot basiert.

@param model_version Die Modellversion, die der Chatbot nutzt.
@param secrets Die Datenbank mit den wichtigen Infos, wie die ID des Vektorspeichers.
@param client Der OpenAI-Client auf dem der Chatbot läuft.
@return Der Assistent
"""
def load_assistant(model_version, secrets, client):
    # Vector Store laden/erstellen
    vector_store_id = load_or_create_vector_store(secrets, client)
    
    if not vector_store_id:
        raise ValueError("❌ Kein Vector Store gefunden oder erstellt!")
    
    # Assistent wird geladen
    if "ASSISTANT" in secrets:
        print("📌 Lade existierenden Assistant...")
        assistant = client.beta.assistants.retrieve(secrets.get("ASSISTANT"))
        
        # Prüfe ob Vector Store aktualisiert werden muss
        current_vector_stores = []
        if hasattr(assistant, 'tool_resources') and assistant.tool_resources:
            if hasattr(assistant.tool_resources, 'file_search') and assistant.tool_resources.file_search:
                file_search = assistant.tool_resources.file_search
                if hasattr(file_search, 'vector_store_ids'):
                    current_vector_stores = file_search.vector_store_ids or []
        
        # Prüfe ob es der richtige Vector Store ist
        if not current_vector_stores or (len(current_vector_stores) > 0 and current_vector_stores[0] != vector_store_id):
            print("🔄 Aktualisiere Vector Store im Assistant...")
            assistant = client.beta.assistants.update(
                assistant_id=assistant.id,
                tool_resources={
                    "file_search": {
                        "vector_store_ids": [vector_store_id]
                    }
                }
            )
            print(f"✅ Assistant aktualisiert mit Vector Store: {vector_store_id}")
        else:
            print(f"✅ Assistant geladen: {assistant.id}")
    else:
        print("🔨 Erstelle neuen Assistant...")
        assistant = client.beta.assistants.create(
            name="Experte für die Fachhochschule Wedel",
            instructions=(
                "Du bist ein Experte für die Fachhochschule Wedel und beantwortest Fragen "
                "anhand der dir bereitgestellten PDF-Dateien."
                "Beantworte alle Fragen präzise basierend auf den Dokumenten. "
                "Wenn eine Information nicht in den Dokumenten enthalten ist, sage dies klar. "
                "Erfinde keine Informationen und suche nicht im Internet. "
                "Bei Fragen zu spezifischen Studiengängen achte darauf, ob es sich um Bachelor "
                "oder Master handelt und zitiere aus den entsprechenden Dokumenten."
            ),
            tools=[{"type": "file_search"}],
            model=model_version,
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store_id]
                }
            }
        )
        
        # Assistant-ID in secrets speichern
        secrets["ASSISTANT"] = assistant.id
        with open(".secrets/secrets.toml", "w") as f:
            toml.dump(secrets, f)
        print(f"✅ Neuer Assistant erstellt: {assistant.id}")
    
    return assistant


"""
Gibt die User-Frage an den Thread weiter und streamt die antwort.

@param question Die vom User gestellte Frage.
@param assistant_id Die ID des Assitants.
@param thread_id Die ID des Threads.
@param Der OpenAI-Client auf dem der Chatbot basiert.
@return Die einzelen gestreamten Antwortpassagen
"""
def ask_assistant(question, assistant_id, thread_id, client):
    start = time.time()
    
    try:
        # 1. User-Nachricht an den Thread anhängen
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=question
        )
        print(f"📤 Nachricht gesendet an Thread {thread_id}")
        
        # 2. Run mit Streaming starten
        stream = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            stream=True
        )
        
        print(f"🏃 Run gestartet mit Streaming")
        
        # 3. Stream verarbeiten
        full_response = ""
        for event in stream:
            # Prüfe verschiedene Event-Typen
            if hasattr(event, 'data') and hasattr(event.data, 'object'):
                if event.data.object == 'thread.message.delta':
                    # Text-Delta empfangen
                    if hasattr(event.data, 'delta') and hasattr(event.data.delta, 'content'):
                        for content in event.data.delta.content:
                            if hasattr(content, 'text') and hasattr(content.text, 'value'):
                                # Gib Text-Chunk zurück
                                yield content.text.value
                                full_response += content.text.value
                                
                elif event.data.object == 'thread.run':
                    # Run Status Update
                    if hasattr(event.data, 'status'):
                        if event.data.status == 'completed':
                            print(f"✅ Run abgeschlossen nach {time.time() - start:.2f} Sekunden")
                        elif event.data.status in ['failed', 'cancelled', 'expired']:
                            print(f"❌ Run fehlgeschlagen: {event.data.status}")
                            yield f"\n\n❌ Fehler: Der Assistant konnte die Anfrage nicht bearbeiten."
                            return
        
        if not full_response:
            yield "Keine Antwort vom Assistant erhalten."
            
    except Exception as e:
        print(f"❌ Fehler bei der Assistant-Anfrage: {e}")
        yield f"\n\n❌ Fehler bei der Verarbeitung: {str(e)}"