"""
Dieses Modul enthält den Hintergrund-Scheduler für die Synchronisation:
- Überwacht lokale PDF-Dateien.
- Stößt regelmäßig eine Synchronisation mit dem OpenAI-Vektorspeicher an.
- Wird in die Flask-App integriert, um im Hintergrund zu laufen.
"""
import schedule
import time
import threading
from datetime import datetime
import signal
import sys


"""
Der Scheduler, der die lokalen PDF-Dateien überwacht.

@param self Die eigene Instanz
@param client Der OpenAI-Client auf dem der Chatbot läuft.
@param secrets Die Datenbank mit den wichtigen Infos, wie die ID des Vektorspeichers.
@param check_interval_minutes Die Häufigkeit, wie oft die Synchronisation geprüft wird.
"""
class VectorStoreSyncScheduler:
    def __init__(self, client, secrets, check_interval_minutes=30):
        self.client = client
        self.secrets = secrets
        self.check_interval = check_interval_minutes
        self.running = False
        self.thread = None
        self.manager = None
    

    """
    Gibt den Vektorspeicher-Manager zurück und speichert ihn falls nötig in der Klasse.

    @param self Die eigene Instanz.
    @return Der Manager für den Vektorspeicher.
    """
    def get_manager(self):
        if not self.manager:
            from vector_store_manager import VectorStoreManager
            self.manager = VectorStoreManager(client=self.client, secrets=self.secrets)
        
        return self.manager
        
    
    """
    Initialisiert einen Synchronisationsvorgang

    @param self Die eigene Instanz.
    """
    def sync_job(self):
        # Log-Ausgabe mit aktuellem Zeitstempel
        print(f"\n⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starte geplante Synchronisation...")
        try:
             # Vektorspeicher-Manager laden
            manager = self.get_manager()

            # Synchronisation starten: True, wenn Änderungen gefunden wurden
            has_changes = manager.sync_vector_stores()
            
            if has_changes:
                print("✅ Synchronisation erfolgreich abgeschlossen - Änderungen gefunden")
                print("⚠️  WICHTIG: Der Assistant sollte neu geladen werden!")
                print("    Rufe /admin/sync auf oder starte die App neu.")
            else:
                print("ℹ️  Keine Änderungen gefunden")
        except Exception as e:
            print(f"❌ Fehler bei der Synchronisation: {e}")
            import traceback
            traceback.print_exc()
    

    """
    Startet den Scheduler im Hintergrund
    
    @param self Die eigene Instanz.
    @param skip_initial_sync Gibt die Möglichkeit, ob man direkt beim Start der App eine Synchronisation möchte oder nicht
    """
    def start(self, skip_initial_sync=False):
        if self.running:
            print("⚠️  Scheduler läuft bereits")
            return
        
        self.running = True
        
        # Optionale initiale Synchronisation
        if not skip_initial_sync:
            print("🚀 Starte Vector Store Sync Scheduler...")
            self.sync_job()
        else:
            print("🚀 Starte Vector Store Sync Scheduler (ohne initiale Synchronisation)...")
        
        # Plane regelmäßige Synchronisation
        schedule.every(self.check_interval).minutes.do(self.sync_job)
        
        # Starte Scheduler-Thread
        self.thread = threading.Thread(target=self.run_scheduler, daemon=True)
        self.thread.start()
        
        print(f"✅ Scheduler gestartet - Prüfung alle {self.check_interval} Minuten")
    
    """
    Interne Funktion für den Scheduler-Thread.

    @param self Die eigene Instanz
    """
    def run_scheduler(self):
        while self.running:
            schedule.run_pending()
            time.sleep(1)
    

    """
    Beendet den Scheduler.

    @param self Die eigene Instanz.
    """
    def stop(self):
        """Stoppt den Scheduler."""
        print("🛑 Stoppe Scheduler...")
        self.running = False
        if self.thread:
            self.thread.join()
        print("✅ Scheduler gestoppt")
    

"""
Initialisierung des Schedulers.

@param client Der OpenAI-Client auf dem der Chatbot läuft.
@param secrets Die Datenbank mit den wichtigen Infos, wie die ID des Vektorspeichers.
@return Der Scheduler.
"""
def integrate_with_flask(client, secrets):
    """Integriert den Sync-Scheduler in die Flask-App."""
    scheduler = VectorStoreSyncScheduler(
        client=client, 
        secrets=secrets,
        check_interval_minutes=30
    )
    
    # Starte Scheduler OHNE initiale Sync (wurde bereits beim Assistant-Laden gemacht)
    scheduler.start(skip_initial_sync=True)
    
    """
    Stoppe Scheduler beim Beenden
    """
    def shutdown_scheduler(signum=None):
        scheduler.stop()
        if signum:
            sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown_scheduler)
    signal.signal(signal.SIGTERM, shutdown_scheduler)
    
    return scheduler
