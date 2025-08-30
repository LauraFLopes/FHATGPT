import time
import toml


def timed_step(name, func, *args, **kwargs):
    """Misst die Ausfuehrungszeit eines Funktionsaufrufs und gibt sie aus."""
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"**{name}** in {end - start:.2f} Sekunden.")
    return result


def load_or_create_single_vector_store(secrets, client):
    """Erstellt oder laedt einen einzigen Vector Store fuer alle PDFs."""
    from vector_store_manager import VectorStoreManager
    
    # Nutze den VectorStoreManager fuer konsistente Verwaltung
    manager = VectorStoreManager(client=client, secrets=secrets)
    vector_store_id, _ = manager.create_or_update_vector_store()
    
    return vector_store_id


def load_assistant(model_version, secrets, client):
    """Laedt einen existierenden Assistant oder erstellt einen neuen."""
    # Vector Store laden/erstellen (nur einen!)
    vector_store_id = load_or_create_single_vector_store(secrets, client)
    
    if not vector_store_id:
        raise ValueError("❌ Kein Vector Store gefunden oder erstellt!")
    
    if "ASSISTANT" in secrets:
        print("📌 Lade existierenden Assistant...")
        assistant = client.beta.assistants.retrieve(secrets.get("ASSISTANT"))
        
        # Pruefe ob Vector Store aktualisiert werden muss
        current_vector_stores = []
        if hasattr(assistant, 'tool_resources') and assistant.tool_resources:
            # tool_resources ist ein Pydantic-Objekt, kein Dictionary
            if hasattr(assistant.tool_resources, 'file_search') and assistant.tool_resources.file_search:
                file_search = assistant.tool_resources.file_search
                if hasattr(file_search, 'vector_store_ids'):
                    current_vector_stores = file_search.vector_store_ids or []
        
        # Pruefe ob es der richtige Vector Store ist
        if not current_vector_stores or (len(current_vector_stores) > 0 and current_vector_stores[0] != vector_store_id):
            print("🔄 Aktualisiere Vector Store im Assistant...")
            assistant = client.beta.assistants.update(
                assistant_id=assistant.id,
                tool_resources={
                    "file_search": {
                        "vector_store_ids": [vector_store_id]  # Nur ein Vector Store!
                    }
                }
            )
            print(f"✅ Assistant aktualisiert mit Vector Store: {vector_store_id}")
        else:
            print(f"✅ Assistant geladen: {assistant.id}")
    else:
        print("🔨 Erstelle neuen Assistant...")
        assistant = client.beta.assistants.create(
            name="Experte fuer die Fachhochschule Wedel",
            instructions=(
                "Du bist ein Experte fuer die Fachhochschule Wedel und beantwortest Fragen "
                "anhand der dir bereitgestellten PDF-Dateien. Die Dokumente sind nach Kategorien "
                "organisiert:\n\n"
                "- Bachelor Modulhandbuch: Modulbeschreibungen fuer Bachelor-Studiengaenge\n"
                "- Bachelor Studienordnung: Regelungen und Ordnungen fuer Bachelor-Studiengaenge\n"
                "- Bachelor Studienverlaufsplan: Empfohlene Studienverlaufsplaene fuer Bachelor\n"
                "- Master Modulhandbuch: Modulbeschreibungen fuer Master-Studiengaenge\n"
                "- Master Studienordnung: Regelungen und Ordnungen fuer Master-Studiengaenge\n"
                "- Master Studienverlaufsplan: Empfohlene Studienverlaufsplaene fuer Master\n"
                "- Regularien: Allgemeine Hochschulregelungen\n\n"
                "Beantworte alle Fragen praezise basierend auf den Dokumenten. "
                "Wenn eine Information nicht in den Dokumenten enthalten ist, sage dies klar. "
                "Erfinde keine Informationen und suche nicht im Internet. "
                "Bei Fragen zu spezifischen Studiengaengen achte darauf, ob es sich um Bachelor "
                "oder Master handelt und zitiere aus den entsprechenden Dokumenten."
            ),
            tools=[{"type": "file_search"}],
            model=model_version,
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store_id]  # Nur ein Vector Store!
                }
            }
        )
        
        # Assistant-ID in secrets speichern
        secrets["ASSISTANT"] = assistant.id
        with open(".streamlit/secrets.toml", "w") as f:
            toml.dump(secrets, f)
        print(f"✅ Neuer Assistant erstellt: {assistant.id}")
    
    return assistant


def ask_assistant(question, assistant_id, thread_id, client):
    """Stellt eine Frage an den Assistant und wartet auf die Antwort."""
    start = time.time()
    
    try:
        # 1. User-Nachricht an den Thread anhaengen
        message = client.beta.threads.messages.create(
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
            # Pruefe verschiedene Event-Typen
            if hasattr(event, 'data') and hasattr(event.data, 'object'):
                if event.data.object == 'thread.message.delta':
                    # Text-Delta empfangen
                    if hasattr(event.data, 'delta') and hasattr(event.data.delta, 'content'):
                        for content in event.data.delta.content:
                            if hasattr(content, 'text') and hasattr(content.text, 'value'):
                                # Gib Text-Chunk zurueck
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