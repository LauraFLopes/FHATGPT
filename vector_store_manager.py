"""
Dieses Modul verwaltet die Synchronisation zwischen lokal gespeicherten PDF-Dateien 
und dem OpenAI-gestützten Vektorspeicher.
"""
import os
import hashlib
import json
from datetime import datetime
from openai import OpenAI
import toml


"""
Verwaltet alle Aufgaben der Synchronisation zwischen lokalen PDFs und dem OpenAI-Vektorspeicher:
- Scannen und Hashing der lokalen Dateien
- Verwaltung des Manifests
- Upload, Update und Löschung von Dateien im Vektorspeicher
- Status- und Reset-Funktionen

@param self Die eigene Instanz
@param client Der OpenAI-Client auf dem der Chatbot läuft.
@param secrets Die Datenbank mit den wichtigen Infos, wie die ID des Vektorspeichers.
@param secrets_path Der Ort, wo die Datenbank mit den wichtigen Infos gespeichert ist.
@param pdf_base_path Der Ort, wo die lokalen PDF-Dateien gespeichert sind.
"""
class VectorStoreManager:
    def __init__(self, client=None, secrets=None, secrets_path=".streamlit/secrets.toml", pdf_base_path="pdfs"):
        self.secrets_path = secrets_path
        self.pdf_base_path = pdf_base_path
        self.manifest_path = os.path.join(pdf_base_path, ".vector_store_manifest.json")
        
        # Secrets entweder übergeben oder laden
        if secrets:
            self.secrets = secrets
        else:
            self.secrets = toml.load(secrets_path)
        
        # Client entweder übergeben oder neu erstellen
        if client:
            self.client = client
        else:
            self.client = OpenAI(api_key=self.secrets.get("API_KEY"))


    """
    Speichert den Hash-Wert der im Manifest jeder Datei zugeordnet wird,
    damit geprüft werden kann, ob es Änderungen an den Dateien gab.

    @param self Die eigene Instanz.
    @param filepath Der Ort, wo das Manifest gespeichert ist.
    @return Der Hash-Wert einer Datei.
    """
    def calculate_file_hash(self, filepath):
        sha256_hash = hashlib.sha256()

        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()
    

    """
    Lädt das Manifest in den Arbeitspeicher

    @param self Die eigene Instanz
    @return Das geladene Manifest
    """
    def load_manifest(self):
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        
        return {}
    

    """
    Speichert das Manifest.

    @param self Die eigene Instanz
    @param manifest Das Manifest
    """
    def save_manifest(self, manifest):
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    

    """
    Sucht alle PDF-Dateien aus dem PDF-Ordner heraus und behält Ordnerstruktur bei, speichert Dateiinformationen inkl. Hash, Größe, Änderungsdatum.

    @param self Die eigene Instanz.
    @return Die File-Infos zu allen PDF-Dateien
    """
    def scan_all_pdfs(self):
        all_files = {}
        
        if not os.path.exists(self.pdf_base_path):
            print(f"❌ Ordner '{self.pdf_base_path}' nicht gefunden!")
            return all_files
        
        # Durchsuche alle Unterordner
        for root, dirs, files in os.walk(self.pdf_base_path):
            # Ignoriere versteckte Ordner
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    # Relativer Pfad vom pdf_base_path
                    relative_path = os.path.relpath(file_path, self.pdf_base_path)
                    
                    # Lade existierendes Manifest für File IDs
                    old_manifest = self.load_manifest()
                    
                    file_info = {
                        "hash": self.calculate_file_hash(file_path),
                        "size": os.path.getsize(file_path),
                        "modified": os.path.getmtime(file_path),
                        "full_path": file_path,
                        "category": os.path.dirname(relative_path) or "root"
                    }
                    
                    # Behalte File ID wenn vorhanden
                    if relative_path in old_manifest and "file_id" in old_manifest[relative_path]:
                        file_info["file_id"] = old_manifest[relative_path]["file_id"]
                    
                    all_files[relative_path] = file_info
        
        return all_files


    """
    Vergleicht die aktuellen Dateien mit den Dateien im Manifest und hält fest,
    welche Art von Änderungen stattgefunden haben.

    @param self Die eigene Instanz.
    @return changes Alle Änderungen im PDF-Ordner.
    @return current_files Alles aktuellen Dateien im PDF-Ordner.
    """
    def scan_changes(self):
        old_manifest = self.load_manifest()
        current_files = self.scan_all_pdfs()
        
        changes = {
            "new_files": [],
            "modified_files": [],
            "deleted_files": [],
            "stats": {
                "total_files": len(current_files),
                "categories": list(set(f["category"] for f in current_files.values()))
            }
        }
        
        # Neue und modifizierte Dateien finden
        for path, info in current_files.items():
            if path not in old_manifest:
                changes["new_files"].append(path)
            elif info["hash"] != old_manifest[path]["hash"]:
                changes["modified_files"].append(path)
        
        # Gelöschte Dateien finden
        for path in old_manifest:
            if path not in current_files:
                changes["deleted_files"].append(path)
        
        return changes, current_files
    

    """
    Speichert ab, wann der letzte Synchronisationsversuch stattgefunden hat
    und wann das letzte Mal etwas erfolgreich verändert wurde.

    @param self Die eigene Instanz.
    @param changes_found Ob zuletzt etwas Verändert wurde.
    """
    def save_sync_history(self, changes_found):
        sync_history_file = os.path.join(self.pdf_base_path, ".sync_history.json")
        
        try:
            if os.path.exists(sync_history_file):
                with open(sync_history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = {}
            
            history["last_sync_attempt"] = datetime.now().isoformat()
            if changes_found:
                history["last_successful_sync"] = datetime.now().isoformat()
            
            os.makedirs(os.path.dirname(sync_history_file), exist_ok=True)
            with open(sync_history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"⚠️  Konnte Sync-Historie nicht speichern: {e}")
    

    """
    Holt alle Dateien aus dem OpenAI-Vektorspeicher

    @param self Die eigene Instanz.
    @param vector_store_id Die ID des Vektorspeichers.
    @return Die Dateien aus dem Vektorspeicher.
    """
    def get_vector_store_files(self, vector_store_id):
        file_mapping = {}
        try:
            # Hole alle Dateien aus dem Vector Store
            files = self.client.vector_stores.files.list(
                vector_store_id=vector_store_id,
                limit=100
            )
            
            for file in files.data:
                # Hole Datei-Details
                try:
                    file_obj = self.client.files.retrieve(file.id)
                    # Filename enthält den relativen Pfad
                    file_mapping[file_obj.filename] = file.id
                except:
                    pass
                    
            # Handle pagination falls mehr als 100 Dateien
            while files.has_more:
                files = self.client.vector_stores.files.list(
                    vector_store_id=vector_store_id,
                    limit=100,
                    after=files.data[-1].id
                )
                for file in files.data:
                    try:
                        file_obj = self.client.files.retrieve(file.id)
                        file_mapping[file_obj.filename] = file.id
                    except:
                        pass
                        
        except Exception as e:
            print(f"⚠️  Fehler beim Abrufen der Vector Store Dateien: {e}")
            
        return file_mapping
    

    """
    Entfernt spezifische Dateien aus dem Vektorspeicher und löscht sie.

    @param self Die eigene Instanz
    @param vector_store_id Die ID des Vektorspeichers.
    @param file_paths Liste mit den Pfaden der PDF-Dateien.
    @param old_manifest Das alte Manifest.
    @return Ob die richtige Menge an Dateien entfernt wurden
    """
    def remove_files_from_vector_store(self, vector_store_id, file_paths, old_manifest):
        if not file_paths:
            return True
                 
        # Hole File-ID Mapping vom Vector Store
        vector_store_files = self.get_vector_store_files(vector_store_id)
        removed_count = 0
        
        for file_path in file_paths:
            # Finde die File ID
            file_id = None
            
            # Versuche aus Manifest
            if file_path in old_manifest and "file_id" in old_manifest[file_path]:
                file_id = old_manifest[file_path]["file_id"]
            else:
                # Versuche über Filename-Matching
                for vs_filename, vs_file_id in vector_store_files.items():
                    if file_path in vs_filename or vs_filename.endswith(file_path):
                        file_id = vs_file_id
                        break
            
            if file_id:
                try:
                    # 1. Aus Vector Store entfernen
                    self.client.vector_stores.files.delete(
                        vector_store_id=vector_store_id,
                        file_id=file_id
                    )
                    
                    # 2. Datei komplett löschen
                    self.client.files.delete(file_id)
                    
                    print(f"   ✅ {file_path}")
                    removed_count += 1
                    
                except Exception as e:
                    print(f"   ⚠️  {file_path}: {str(e)[:50]}...")
            else:
                print(f"   ⚠️  {file_path}: File ID nicht gefunden")
        
        return removed_count == len(file_paths)
    

    """
    Lädt Dateien in den Vektorspeicher hoch und gibt die ID dieser Dateien zurück.

    @param self Die eigene Instanz.
    @param vector_store_id Die ID für den Vektorspeicher.
    @param file_paths Der Pfad zu den Dateien.
    @param current_files Die aktuellen Dateien im PDF-Ordner.
    @return Die IDs der Dateien.
    """
    def upload_files_and_get_ids(self, vector_store_id, file_paths, current_files):
        file_id_mapping = {}
        
        for pdf_path in file_paths:
            # Korrigiere den relativen Pfad - entferne 'pdfs\' Präfix falls vorhanden
            if os.path.isabs(pdf_path):
                relative = os.path.relpath(pdf_path, self.pdf_base_path)
            else:
                relative = pdf_path
                
            # Entferne 'pdfs\' oder 'pdfs/' Präfix falls vorhanden
            if relative.startswith('pdfs\\') or relative.startswith('pdfs/'):
                relative = relative[5:]
                
            # Hole den vollen Pfad aus current_files
            if relative in current_files:
                full_path = current_files[relative]["full_path"]
            else:
                # Fallback für Pfade die nicht in current_files sind
                full_path = pdf_path if os.path.isabs(pdf_path) else os.path.join(self.pdf_base_path, relative)
            
            try:
                with open(full_path, "rb") as f:
                    # Upload mit relativem Pfad als Filename
                    file = self.client.files.create(
                        file=(relative, f),
                        purpose="assistants"
                    )
                    
                    # Zum Vector Store hinzufügen
                    self.client.vector_stores.files.create(
                        vector_store_id=vector_store_id,
                        file_id=file.id
                    )
                    
                    file_id_mapping[relative] = file.id
                    print(f"   ✅ {relative}")
                    
            except Exception as e:
                print(f"   ❌ {relative}: {str(e)[:50]}...")
                
        return file_id_mapping
    

    """
    Erstellt oder aktualisiert den Vektorspeicher für den Chatbot.

    @param self Die eigene Instanz.
    @param progress_callback Ob eine Rückgabe von Infos an das Webinterface stattfinden soll.
    @return vector_store_id Die ID des Vektorspeichers.
    @return has_changes Gibt zurück, ob es Änderungen gab.
    """
    def create_or_update_vector_store(self, progress_callback=None):
        changes, current_files = self.scan_changes()
        old_manifest = self.load_manifest()
        
        """
        Gibt Änderungen aus.

        @param message Die Nachricht, die Ausgegeben werden soll.
        @param status Die Art der Nachricht.
        @param details Die Art von Änderungen, die stattgefunden haben
        """
        def update_progress(message, status="info", details=None):
            print(message)
            if details:
                if 'categories' in details:
                    print(f"   Kategorien: {', '.join(details['categories'])}")
                if 'counts' in details:
                    counts = details['counts']
                    if counts.get('new', 0) > 0:
                        print(f"   ➕ Neue Dateien: {counts['new']}")
                    if counts.get('modified', 0) > 0:
                        print(f"   🔄 Geänderte Dateien: {counts['modified']}")
                    if counts.get('deleted', 0) > 0:
                        print(f"   ➖ Gelöschte Dateien: {counts['deleted']}")
                if 'new' in details and details['new']:
                    print("   Neue Dateien:")
                    for f in details['new'][:5]:  # Zeige max 5
                        print(f"     • {f}")
                    if len(details['new']) > 5:
                        print(f"     ... und {len(details['new']) - 5} weitere")
                if 'deleted' in details and details['deleted']:
                    print("   Gelöschte Dateien:")
                    for f in details['deleted'][:5]:  # Zeige max 5
                        print(f"     • {f}")
                    if len(details['deleted']) > 5:
                        print(f"     ... und {len(details['deleted']) - 5} weitere")
            
            # Web-Interface Callback (wenn vorhanden)
            if progress_callback:
                progress_callback({
                    "message": message,
                    "status": status,
                    "details": details
                })
        
        # Zeige Statistiken
        update_progress(f"📊 Gefunden: {changes['stats']['total_files']} PDFs in {len(changes['stats']['categories'])} Kategorien", "info", {
            "categories": changes['stats']['categories']
        })
        
        # Prüfe ob es Änderungen gibt
        has_changes = any([changes["new_files"], changes["modified_files"], changes["deleted_files"]])
        
        # Vector Store ID holen
        vector_store_id = self.secrets.get("MAIN_VECTOR_STORE")
        
        if not has_changes and vector_store_id:
            update_progress("✅ Keine Änderungen gefunden", "success")
            return vector_store_id, False
        
        # Zeige Änderungen
        if has_changes:
            change_details = {
                "new": changes["new_files"][:10] if changes["new_files"] else [],
                "modified": changes["modified_files"][:10] if changes["modified_files"] else [],
                "deleted": changes["deleted_files"][:10] if changes["deleted_files"] else [],
                "counts": {
                    "new": len(changes["new_files"]),
                    "modified": len(changes["modified_files"]),
                    "deleted": len(changes["deleted_files"])
                }
            }
            update_progress("📋 Änderungen erkannt", "info", change_details)
        
        # Vector Store erstellen oder aktualisieren
        if vector_store_id and has_changes:
            # Versuche gezieltes Update für ALLE Änderungen
            update_progress("🔄 Aktualisiere Vector Store...", "info")
            
            update_success = True
            
            # 1. Gelöschte Dateien entfernen
            if changes["deleted_files"]:
                update_progress(f"🗑️ Entferne {len(changes['deleted_files'])} Dateien...", "info")
                update_success = self.remove_files_from_vector_store(
                    vector_store_id, 
                    changes["deleted_files"], 
                    old_manifest
                )
                if update_success:
                    update_progress(f"✅ {len(changes['deleted_files'])} Dateien entfernt", "success")
            
            # 2. Modifizierte Dateien: Alte Version löschen, neue hochladen
            if update_success and changes["modified_files"]:
                update_progress("🔄 Aktualisiere modifizierte Dateien...", "info")
                self.remove_files_from_vector_store(
                    vector_store_id, 
                    changes["modified_files"], 
                    old_manifest
                )
                
                # Dann neu hochladen
                modified_paths = [current_files[path]["full_path"] for path in changes["modified_files"]]
                file_id_mapping = self.upload_files_and_get_ids(vector_store_id, modified_paths, current_files)
                
                # Manifest mit neuen File IDs aktualisieren
                for path, file_id in file_id_mapping.items():
                    if path in current_files:
                        current_files[path]["file_id"] = file_id

                updated_count = len(file_id_mapping)
                if updated_count == len(changes["modified_files"]):
                    update_progress(f"✅ {updated_count} Dateien erfolgreich aktualisiert", "success")
                else:
                    update_progress(f"⚠️ {updated_count}/{len(changes['modified_files'])} Dateien aktualisiert", "warning")
            
            # 3. Neue Dateien hinzufügen
            if update_success and changes["new_files"]:
                update_progress(f"➕ Füge {len(changes['new_files'])} neue Dateien hinzu...", "info")
                
                new_file_paths = [current_files[path]["full_path"] for path in changes["new_files"]]
                file_id_mapping = self.upload_files_and_get_ids(vector_store_id, new_file_paths, current_files)
                
                # Update Manifest mit neuen File IDs
                for path, file_id in file_id_mapping.items():
                    if path in current_files:
                        current_files[path]["file_id"] = file_id
                
                added_count = len(file_id_mapping)
                if added_count == len(changes["new_files"]):
                    update_progress(f"✅ {added_count} neue Dateien erfolgreich hinzugefügt", "success")
                else:
                    update_progress(f"⚠️ {added_count}/{len(changes['new_files'])} neue Dateien hinzugefügt", "warning")
            
            # Falls Update fehlgeschlagen, dann kompletter Rebuild
            if not update_success:
                update_progress("⚠️ Update fehlgeschlagen - erstelle Vector Store neu...", "warning")
                try:
                    self.client.vector_stores.delete(vector_store_id)
                    print("🗑️  Alter Vector Store gelöscht")
                except:
                    pass
                vector_store_id = None
            else:
                # Speichere Manifest nach erfolgreichem Update
                self.save_manifest(current_files)
        
        # Neuen Vector Store erstellen wenn nötig
        if not vector_store_id:
            # Neuen Vector Store erstellen
            update_progress("🔨 Erstelle neuen Vector Store...", "info")
            try:
                vector_store = self.client.vector_stores.create(
                    name="FH Wedel - Alle Dokumente"
                )
                vector_store_id = vector_store.id
                update_progress(f"📁 Vector Store erstellt: {vector_store_id}", "success")
                
                # Alle PDFs hochladen mit Fehlerbehandlung
                all_pdf_paths = [info["full_path"] for info in current_files.values()]
                failed_files = []
                successfully_uploaded = {}
                
                # Upload in Batches (max 100 pro Batch)
                batch_size = 100
                total_batches = (len(all_pdf_paths) - 1) // batch_size + 1
                
                for i in range(0, len(all_pdf_paths), batch_size):
                    batch_files = all_pdf_paths[i:i+batch_size]
                    file_streams = []
                    batch_relatives = []
                    
                    batch_num = i // batch_size + 1
                    update_progress(f"📤 Lade Batch {batch_num}/{total_batches} ({len(batch_files)} Dateien)...", "info", {
                        "progress": {
                            "current": i + len(batch_files),
                            "total": len(all_pdf_paths)
                        }
                    })
                    
                    # Erstelle eine Mapping von Index zu relativem Pfad für später
                    batch_mapping = {}
                    for idx, pdf_path in enumerate(batch_files):
                        relative = os.path.relpath(pdf_path, self.pdf_base_path)
                        batch_relatives.append(relative)
                        batch_mapping[idx] = relative
                        print(f"   - {relative}")
                        f = open(pdf_path, "rb")
                        file_streams.append((relative, f))
                    
                    # Upload und Verarbeitung
                    try:
                        file_batch = self.client.vector_stores.file_batches.upload_and_poll(
                            vector_store_id=vector_store_id,
                            files=file_streams
                        )
                        
                        # Prüfe auf fehlgeschlagene Dateien
                        if hasattr(file_batch, 'file_counts'):
                            if file_batch.file_counts.failed > 0:
                                print(f"⚠️  {file_batch.file_counts.failed} Dateien fehlgeschlagen")
                                failed_files.extend(batch_files)
                            
                            update_progress(f"✅ Batch {batch_num} verarbeitet", "success", {
                                "batch_results": {
                                    "completed": file_batch.file_counts.completed,
                                    "failed": file_batch.file_counts.failed,
                                    "total": file_batch.file_counts.total
                                }
                            })
                            
                    except Exception as e:
                        update_progress(f"❌ Fehler beim Batch-Upload: {str(e)[:100]}", "error")
                        failed_files.extend(batch_files)
                        
                    finally:
                        # Streams schließen
                        for _, stream in file_streams:
                            stream.close()
                
                # Versuche fehlgeschlagene Dateien einzeln hochzuladen
                if failed_files:
                    update_progress(f"🔄 Versuche {len(failed_files)} fehlgeschlagene Dateien einzeln hochzuladen...", "warning")
                    retry_success = 0
                    still_failed = []
                    
                    for pdf_path in failed_files:
                        relative = os.path.relpath(pdf_path, self.pdf_base_path)
                        file_id_mapping = self.upload_files_and_get_ids(vector_store_id, [pdf_path], current_files)
                        if relative in file_id_mapping:
                            successfully_uploaded[relative] = file_id_mapping[relative]
                            retry_success += 1
                        else:
                            still_failed.append(relative)
                    
                    update_progress(f"✅ {retry_success}/{len(failed_files)} Dateien erfolgreich nachgeladen", "success" if retry_success > 0 else "warning")
                    
                    # Wenn immer noch Dateien fehlen, Vector Store als unvollständig markieren
                    if still_failed:
                        error_details = {
                            "failed_files": still_failed[:10],  # Zeige max 10
                            "total_failed": len(still_failed)
                        }
                        update_progress(f"❌ {len(still_failed)} Dateien konnten nicht hochgeladen werden", "error", error_details)
                        
                        # Entscheide basierend auf Anzahl der Fehler
                        if len(still_failed) > 5 or len(still_failed) > len(successfully_uploaded) * 0.1:
                            # Mehr als 5 Fehler oder mehr als 10% fehlgeschlagen
                            update_progress("⚠️ Zu viele Fehler - Vector Store wird verworfen", "error")
                            try:
                                self.client.vector_stores.delete(vector_store_id)
                                print("🗑️  Vector Store gelöscht")
                            except:
                                pass
                            return None, False
                        else:
                            update_progress("⚠️ Vector Store wurde mit fehlenden Dateien erstellt", "warning")
                            print("   Führe manuell eine Synchronisation durch um die fehlenden Dateien hinzuzufügen")
                
                # Update Manifest mit allen File IDs
                for relative_path, file_id in successfully_uploaded.items():
                    if relative_path in current_files:
                        current_files[relative_path]["file_id"] = file_id
                
                # Speichere Vector Store ID
                self.secrets["MAIN_VECTOR_STORE"] = vector_store_id
                
                # Entferne alte VECTOR_STORES falls vorhanden
                if "VECTOR_STORES" in self.secrets:
                    del self.secrets["VECTOR_STORES"]
                
                with open(self.secrets_path, "w") as f:
                    toml.dump(self.secrets, f)
                update_progress("💾 Vector Store ID gespeichert", "success")
                
                # WICHTIG: Speichere das Manifest nach dem Erstellen eines neuen Vector Stores
                self.save_manifest(current_files)
                update_progress("💾 Manifest erstellt/aktualisiert", "success")
                
            except Exception as e:
                update_progress(f"❌ Fehler beim Erstellen des Vector Stores: {str(e)}", "error")
                import traceback
                traceback.print_exc()
                return None, False
        
        # Manifest speichern falls noch nicht geschehen
        if has_changes:
            self.save_manifest(current_files)
        
        update_progress("✅ Synchronisation abgeschlossen", "success")
        
        return vector_store_id, has_changes
    

    """
    Hauptfunktion zum Synchronisieren.

    @param self Die eigene Instanz.
    @param progress_callback Ob eine Rückgabe von Infos an das Webinterface stattfinden soll
    @return Ob Änderungen vorliegen.
    """
    def sync_vector_stores(self, progress_callback=None):
        _, has_changes = self.create_or_update_vector_store(progress_callback)
        self.save_sync_history(has_changes)
        
        return has_changes
    

    """
    Gibt einen detaillierten Status über den Vektorstore zurück

    @param self Die eigene Instanz
    @return Der Status
    """
    def get_sync_status(self):
        manifest = self.load_manifest()
        status = {
            "last_sync": None,
            "last_sync_attempt": None,
            "vector_store_id": self.secrets.get("MAIN_VECTOR_STORE"),
            "vector_store_size": None,
            "categories": {},
            "total_files": len(manifest),
            "total_size_local": 0
        }
        
        # Hole Vector Store Details von OpenAI
        if status["vector_store_id"]:
            try:
                vector_store = self.client.vector_stores.retrieve(status["vector_store_id"])
                # Vector Store hat file_counts und usage_bytes
                if hasattr(vector_store, 'file_counts'):
                    status["vector_store_file_count"] = vector_store.file_counts.total
                if hasattr(vector_store, 'usage_bytes'):
                    status["vector_store_size"] = vector_store.usage_bytes
            except Exception as e:
                print(f"⚠️  Konnte Vector Store Details nicht abrufen: {e}")
        
        # Lade Sync-Historie
        sync_history_file = os.path.join(self.pdf_base_path, ".sync_history.json")
        if os.path.exists(sync_history_file):
            try:
                with open(sync_history_file, 'r') as f:
                    sync_history = json.load(f)
                    status["last_sync"] = sync_history.get("last_successful_sync")
                    status["last_sync_attempt"] = sync_history.get("last_sync_attempt")
            except:
                pass
        
        if manifest:
            # Gruppiere nach Kategorien
            for path, info in manifest.items():
                category = info.get("category", "root")
                if category not in status["categories"]:
                    status["categories"][category] = {
                        "files": 0,
                        "size": 0,
                        "file_list": []
                    }
                
                status["categories"][category]["files"] += 1
                status["categories"][category]["size"] += info["size"]
                status["categories"][category]["file_list"].append(os.path.basename(path))
                status["total_size_local"] += info["size"]
        
        return status


    """
    Setzt alles zurück, also löscht OpenAI-Vectorspeicher,
    Dateien (Keine PDF-Dateien), Manifest und Secrets (Außer API-Key).

    @param self Die eigene Instanz
    @param keep_admin_token Ob der Token für die Admin-Seite behalten werden soll, oder nicht
    @return Ob der Reset erfolgreich war
    """
    def reset_all(self, keep_admin_token=True):
        print("\n🔄 RESET: Beginne vollständigen Reset...")
        
        # 1. Vector Store und alle Files loeschen
        if "MAIN_VECTOR_STORE" in self.secrets:
            vector_store_id = self.secrets["MAIN_VECTOR_STORE"]
            print(f"🗑️  Lösche Vector Store: {vector_store_id}")
            
            try:
                # Hole alle Files aus dem Vector Store
                vs_files = self.get_vector_store_files(vector_store_id)
                print(f"   Gefunden: {len(vs_files)} Dateien zum Löschen")
                
                # Lösche alle Files aus OpenAI Storage
                deleted_count = 0
                for filename, file_id in vs_files.items():
                    try:
                        self.client.files.delete(file_id)
                        deleted_count += 1
                        if deleted_count <= 5:  # Zeige nur erste 5
                            print(f"   ✅ Gelöscht: {filename}")
                    except Exception as e:
                        print(f"   ⚠️  Fehler beim Löschen von {filename}: {str(e)[:50]}")
                
                if deleted_count > 5:
                    print(f"   ... und {deleted_count - 5} weitere Dateien")
                
                # Lösche den Vector Store selbst
                try:
                    self.client.vector_stores.delete(vector_store_id)
                    print(f"✅ Vector Store gelöscht")
                except Exception as e:
                    print(f"⚠️  Fehler beim Löschen des Vector Stores: {e}")
                    
            except Exception as e:
                print(f"❌ Fehler beim Abrufen der Vector Store Files: {e}")
        
        # 2. Assistant löschen
        if "ASSISTANT" in self.secrets:
            assistant_id = self.secrets["ASSISTANT"]
            print(f"🗑️  Lösche Assistant: {assistant_id}")
            try:
                self.client.beta.assistants.delete(assistant_id)
                print("✅ Assistant gelöscht")
            except Exception as e:
                print(f"⚠️  Fehler beim Löschen des Assistants: {e}")
        
        # 3. Lokale Dateien löschen
        # Manifest
        if os.path.exists(self.manifest_path):
            os.remove(self.manifest_path)
            print("✅ Manifest gelöscht")
            
        # Sync-Historie
        sync_history_file = os.path.join(self.pdf_base_path, ".sync_history.json")
        if os.path.exists(sync_history_file):
            os.remove(sync_history_file)
            print("✅ Sync-Historie gelöscht")
        
        # 4. Secrets bereinigen (behalte nur API_KEY und optional ADMIN_TOKEN)
        api_key = self.secrets.get("API_KEY")
        admin_token = self.secrets.get("ADMIN_TOKEN") if keep_admin_token else None
        
        # Alles löschen
        self.secrets.clear()
        
        # Wichtige Keys wiederherstellen
        self.secrets["API_KEY"] = api_key
        if admin_token and keep_admin_token:
            self.secrets["ADMIN_TOKEN"] = admin_token
            
        # Speichern
        with open(self.secrets_path, "w") as f:
            toml.dump(self.secrets, f)
        print("✅ Secrets bereinigt (API_KEY" + (" und ADMIN_TOKEN" if keep_admin_token else "") + " behalten)")
        
        print("\n✅ RESET ABGESCHLOSSEN!")
        print("   Das System ist jetzt im Ausgangszustand.")
        print("   Beim nächsten Start wird alles neu erstellt.")
        
        return True

# Beispiel-Nutzung
if __name__ == "__main__":
    manager = VectorStoreManager()
    
    # Status anzeigen
    status = manager.get_sync_status()
    print("\n📊 Vector Store Status:")
    print(f"   Vector Store ID: {status['vector_store_id']}")
    print(f"   Letzte Synchronisation: {status['last_sync']}")
    print(f"   Gesamtanzahl Dateien: {status['total_files']}")
    print(f"   Gesamtgroeße: {status['total_size_local'] / (1024*1024):.2f} MB")
    print(f"\n   Kategorien:")
    for cat, info in status['categories'].items():
        print(f"   - {cat}: {info['files']} Dateien ({info['size'] / (1024*1024):.2f} MB)")
    
    # Synchronisation durchfuehren
    print("\n" + "="*50)
    manager.create_or_update_vector_store()
