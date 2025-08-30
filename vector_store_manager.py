import os
import hashlib
import json
from datetime import datetime
from openai import OpenAI
import toml

class VectorStoreManager:
    def __init__(self, client=None, secrets=None, secrets_path=".streamlit/secrets.toml", pdf_base_path="pdfs"):
        self.secrets_path = secrets_path
        self.pdf_base_path = pdf_base_path
        self.manifest_path = os.path.join(pdf_base_path, ".vector_store_manifest.json")
        
        # Secrets entweder uebergeben oder laden
        if secrets:
            self.secrets = secrets
        else:
            self.secrets = toml.load(secrets_path)
        
        # Client entweder uebergeben oder neu erstellen
        if client:
            self.client = client
        else:
            self.client = OpenAI(api_key=self.secrets.get("API_KEY"))
        
    def calculate_file_hash(self, filepath):
        """Berechnet SHA256 Hash einer Datei."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def load_manifest(self):
        """Laedt das Manifest mit Datei-Informationen."""
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_manifest(self, manifest):
        """Speichert das Manifest."""
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def scan_all_pdfs(self):
        """Scannt alle PDFs in allen Unterordnern und behaelt die Ordnerstruktur bei."""
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
                    
                    # Lade existierendes Manifest fuer File IDs
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
    
    def scan_changes(self):
        """Vergleicht aktuelle Dateien mit dem Manifest."""
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
        
        # Geloeschte Dateien finden
        for path in old_manifest:
            if path not in current_files:
                changes["deleted_files"].append(path)
        
        return changes, current_files
    
    def save_sync_history(self, changes_found):
        """Speichert Sync-Historie fuer besseres Tracking."""
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
    
    def get_vector_store_files(self, vector_store_id):
        """Holt alle Dateien aus einem Vector Store mit ihren Metadaten."""
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
                    # Filename enthaelt den relativen Pfad
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
    
    def remove_files_from_vector_store(self, vector_store_id, file_paths, old_manifest):
        """Entfernt spezifische Dateien aus dem Vector Store und loescht sie komplett."""
        if not file_paths:
            return True
            
        print(f"🗑️  Entferne {len(file_paths)} Dateien aus Vector Store...")
        
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
                # Versuche ueber Filename-Matching
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
                    
                    # 2. Datei komplett loeschen
                    self.client.files.delete(file_id)
                    
                    print(f"   ✅ {file_path}")
                    removed_count += 1
                    
                except Exception as e:
                    print(f"   ⚠️  {file_path}: {str(e)[:50]}...")
            else:
                print(f"   ⚠️  {file_path}: File ID nicht gefunden")
        
        print(f"✅ {removed_count}/{len(file_paths)} Dateien erfolgreich entfernt")
        return removed_count == len(file_paths)
    
    def upload_files_and_get_ids(self, vector_store_id, file_paths, current_files):
        """Laedt Dateien hoch und gibt die File IDs zurueck."""
        file_id_mapping = {}
        
        for pdf_path in file_paths:
            # Korrigiere den relativen Pfad - entferne 'pdfs\' Praefix falls vorhanden
            if os.path.isabs(pdf_path):
                relative = os.path.relpath(pdf_path, self.pdf_base_path)
            else:
                relative = pdf_path
                
            # Entferne 'pdfs\' oder 'pdfs/' Praefix falls vorhanden
            if relative.startswith('pdfs\\') or relative.startswith('pdfs/'):
                relative = relative[5:]
                
            # Hole den vollen Pfad aus current_files
            if relative in current_files:
                full_path = current_files[relative]["full_path"]
            else:
                # Fallback fuer Pfade die nicht in current_files sind
                full_path = pdf_path if os.path.isabs(pdf_path) else os.path.join(self.pdf_base_path, relative)
            
            try:
                with open(full_path, "rb") as f:
                    # Upload mit relativem Pfad als Filename (ohne pdfs/ Praefix)
                    file = self.client.files.create(
                        file=(relative, f),
                        purpose="assistants"
                    )
                    
                    # Zum Vector Store hinzufuegen
                    self.client.vector_stores.files.create(
                        vector_store_id=vector_store_id,
                        file_id=file.id
                    )
                    
                    file_id_mapping[relative] = file.id
                    print(f"   ✅ {relative}")
                    
            except Exception as e:
                print(f"   ❌ {relative}: {str(e)[:50]}...")
                
        return file_id_mapping
    
    def create_or_update_vector_store(self, progress_callback=None):
        """Erstellt oder aktualisiert den einzigen Vector Store mit Progress-Callback."""
        changes, current_files = self.scan_changes()
        old_manifest = self.load_manifest()
        
        # Progress-Update Helper
        def update_progress(message, status="info", details=None):
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
        
        # Pruefe ob es Aenderungen gibt
        has_changes = any([changes["new_files"], changes["modified_files"], changes["deleted_files"]])
        
        # Pruefe auch ob Vector Store vollstaendig ist
        vector_store_id = self.secrets.get("MAIN_VECTOR_STORE")
        needs_repair = False
        
        if vector_store_id and not has_changes:
            # Pruefe ob alle Dateien im Vector Store sind
            try:
                vector_store = self.client.vector_stores.retrieve(vector_store_id)
                if hasattr(vector_store, 'file_counts'):
                    actual_files = vector_store.file_counts.completed
                    expected_files = len(current_files)
                    if actual_files < expected_files:
                        needs_repair = True
                        update_progress(f"⚠️ Vector Store unvollstaendig: {actual_files}/{expected_files} Dateien", "warning")
                        update_progress("Starte Reparatur...", "info")
            except:
                pass
        
        if not has_changes and not needs_repair and vector_store_id:
            update_progress("✅ Keine Aenderungen gefunden", "success")
            return vector_store_id, False
        
        # Zeige Aenderungen
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
            update_progress("📋 Aenderungen erkannt", "info", change_details)
        
        # Vector Store erstellen oder aktualisieren
        if vector_store_id and has_changes:
            # Versuche gezieltes Update fuer ALLE Aenderungen
            print("🔄 Aktualisiere Vector Store...")
            
            update_success = True
            
            # 1. Geloeschte Dateien entfernen
            if changes["deleted_files"]:
                update_progress(f"🗑️ Entferne {len(changes['deleted_files'])} Dateien...", "info")
                update_success = self.remove_files_from_vector_store(
                    vector_store_id, 
                    changes["deleted_files"], 
                    old_manifest
                )
                if update_success:
                    update_progress(f"✅ {len(changes['deleted_files'])} Dateien entfernt", "success")
            
            # 2. Modifizierte Dateien: Alte Version loeschen, neue hochladen
            if update_success and changes["modified_files"]:
                print("🔄 Aktualisiere modifizierte Dateien...")
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
            
            # 3. Neue Dateien hinzufuegen
            if update_success and changes["new_files"]:
                print(f"\n➕ Fuege {len(changes['new_files'])} neue Dateien hinzu...")
                
                new_file_paths = [current_files[path]["full_path"] for path in changes["new_files"]]
                file_id_mapping = self.upload_files_and_get_ids(vector_store_id, new_file_paths, current_files)
                
                # Update Manifest mit neuen File IDs
                for path, file_id in file_id_mapping.items():
                    if path in current_files:
                        current_files[path]["file_id"] = file_id
            
            # Falls Update fehlgeschlagen oder Reparatur noetig, dann kompletter Rebuild
            if not update_success or needs_repair:
                print("⚠️  Update fehlgeschlagen - erstelle Vector Store neu...")
                try:
                    self.client.vector_stores.delete(vector_store_id)
                    print("🗑️  Alter Vector Store geloescht")
                except:
                    pass
                vector_store_id = None
            else:
                # Speichere Manifest nach erfolgreichem Update
                self.save_manifest(current_files)
        
        # Neuen Vector Store erstellen wenn noetig
        if not vector_store_id:
            # Neuen Vector Store erstellen
            print("🔨 Erstelle neuen Vector Store...")
            try:
                vector_store = self.client.vector_stores.create(
                    name="FH Wedel - Alle Dokumente"
                )
                vector_store_id = vector_store.id
                print(f"📁 Vector Store erstellt: {vector_store_id}")
                
                # Alle PDFs hochladen mit Fehlerbehandlung
                all_pdf_paths = [info["full_path"] for info in current_files.values()]
                failed_files = []
                successfully_uploaded = {}
                
                # Upload in Batches (max 100 pro Batch)
                batch_size = 100
                for i in range(0, len(all_pdf_paths), batch_size):
                    batch_files = all_pdf_paths[i:i+batch_size]
                    file_streams = []
                    batch_relatives = []
                    
                    print(f"\n📤 Lade Batch {i//batch_size + 1}/{(len(all_pdf_paths)-1)//batch_size + 1} ({len(batch_files)} Dateien)...")
                    
                    # Erstelle eine Mapping von Index zu relativem Pfad fuer spaeter
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
                        
                # Nach erfolgreichen Batches ohne spezifische Fehlerbehandlung oben
                        
                        # Pruefe auf fehlgeschlagene Dateien
                        if hasattr(file_batch, 'file_counts'):
                            if file_batch.file_counts.failed > 0:
                                print(f"⚠️  {file_batch.file_counts.failed} Dateien fehlgeschlagen")
                                failed_files.extend(batch_files)
                            
                            print(f"✅ Batch verarbeitet: {file_batch.file_counts}")
                            
                    except Exception as e:
                        print(f"❌ Fehler beim Batch-Upload: {e}")
                        failed_files.extend(batch_files)
                        
                    finally:
                        # Streams schließen
                        for _, stream in file_streams:
                            stream.close()
                
                # Versuche fehlgeschlagene Dateien einzeln hochzuladen
                if failed_files:
                    print(f"\n🔄 Versuche {len(failed_files)} fehlgeschlagene Dateien einzeln hochzuladen...")
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
                    
                    print(f"✅ {retry_success}/{len(failed_files)} Dateien erfolgreich nachgeladen")
                    
                    # Wenn immer noch Dateien fehlen, Vector Store als unvollstaendig markieren
                    if still_failed:
                        print(f"\n❌ {len(still_failed)} Dateien konnten nicht hochgeladen werden:")
                        for f in still_failed[:10]:  # Zeige max 10
                            print(f"   - {f}")
                        if len(still_failed) > 10:
                            print(f"   ... und {len(still_failed) - 10} weitere")
                        
                        # Entscheide basierend auf Anzahl der Fehler
                        if len(still_failed) > 5 or len(still_failed) > len(successfully_uploaded) * 0.1:
                            # Mehr als 5 Fehler oder mehr als 10% fehlgeschlagen
                            print("\n⚠️  Zu viele Fehler - Vector Store wird verworfen")
                            try:
                                self.client.vector_stores.delete(vector_store_id)
                                print("🗑️  Vector Store geloescht")
                            except:
                                pass
                            return None, False
                        else:
                            print("\n⚠️  Vector Store wurde mit fehlenden Dateien erstellt")
                            print("   Fuehre manuell eine Synchronisation durch um die fehlenden Dateien hinzuzufuegen")
                
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
                print("💾 Vector Store ID gespeichert")
                
                # WICHTIG: Speichere das Manifest nach dem Erstellen eines neuen Vector Stores
                self.save_manifest(current_files)
                print("💾 Manifest erstellt/aktualisiert")
                
            except Exception as e:
                print(f"❌ Fehler beim Erstellen des Vector Stores: {e}")
                import traceback
                traceback.print_exc()
                return None, False
        
        # Manifest speichern falls noch nicht geschehen
        if has_changes:
            self.save_manifest(current_files)
        
        print("\n✅ Synchronisation abgeschlossen")
        
        return vector_store_id, has_changes
    
    def sync_vector_stores(self, progress_callback=None):
        """Hauptfunktion zum Synchronisieren mit Progress-Callback."""
        _, has_changes = self.create_or_update_vector_store(progress_callback)
        self.save_sync_history(has_changes)
        return has_changes
    
    def get_sync_status(self):
        """Gibt einen detaillierten Status ueber den Vector Store zurueck."""
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

    def reset_all(self, keep_admin_token=True):
        """Setzt alles zurueck - loescht Vector Store, Files, Manifest und Secrets (außer API Key)."""
        print("\n🔄 RESET: Beginne vollstaendigen Reset...")
        
        # 1. Vector Store und alle Files loeschen
        if "MAIN_VECTOR_STORE" in self.secrets:
            vector_store_id = self.secrets["MAIN_VECTOR_STORE"]
            print(f"🗑️  Loesche Vector Store: {vector_store_id}")
            
            try:
                # Hole alle Files aus dem Vector Store
                vs_files = self.get_vector_store_files(vector_store_id)
                print(f"   Gefunden: {len(vs_files)} Dateien zum Loeschen")
                
                # Loesche alle Files aus OpenAI Storage
                deleted_count = 0
                for filename, file_id in vs_files.items():
                    try:
                        self.client.files.delete(file_id)
                        deleted_count += 1
                        if deleted_count <= 5:  # Zeige nur erste 5
                            print(f"   ✅ Geloescht: {filename}")
                    except Exception as e:
                        print(f"   ⚠️  Fehler beim Loeschen von {filename}: {str(e)[:50]}")
                
                if deleted_count > 5:
                    print(f"   ... und {deleted_count - 5} weitere Dateien")
                
                # Loesche den Vector Store selbst
                try:
                    self.client.vector_stores.delete(vector_store_id)
                    print(f"✅ Vector Store geloescht")
                except Exception as e:
                    print(f"⚠️  Fehler beim Loeschen des Vector Stores: {e}")
                    
            except Exception as e:
                print(f"❌ Fehler beim Abrufen der Vector Store Files: {e}")
        
        # 2. Assistant loeschen
        if "ASSISTANT" in self.secrets:
            assistant_id = self.secrets["ASSISTANT"]
            print(f"🗑️  Loesche Assistant: {assistant_id}")
            try:
                self.client.beta.assistants.delete(assistant_id)
                print("✅ Assistant geloescht")
            except Exception as e:
                print(f"⚠️  Fehler beim Loeschen des Assistants: {e}")
        
        # 3. Lokale Dateien loeschen
        # Manifest
        if os.path.exists(self.manifest_path):
            os.remove(self.manifest_path)
            print("✅ Manifest geloescht")
            
        # Sync-Historie
        sync_history_file = os.path.join(self.pdf_base_path, ".sync_history.json")
        if os.path.exists(sync_history_file):
            os.remove(sync_history_file)
            print("✅ Sync-Historie geloescht")
        
        # 4. Secrets bereinigen (behalte nur API_KEY und optional ADMIN_TOKEN)
        api_key = self.secrets.get("API_KEY")
        admin_token = self.secrets.get("ADMIN_TOKEN") if keep_admin_token else None
        
        # Alles loeschen
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
        print("   Beim naechsten Start wird alles neu erstellt.")
        
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