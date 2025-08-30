import schedule
import time
import threading
from datetime import datetime
import signal
import sys


class VectorStoreSyncScheduler:
    def __init__(self, client, secrets, check_interval_minutes=30):
        self.client = client
        self.secrets = secrets
        self.check_interval = check_interval_minutes
        self.running = False
        self.thread = None
        self.manager = None
        
    def _get_manager(self):
        """Lazy-load VectorStoreManager um zirkulaere Imports zu vermeiden."""
        if not self.manager:
            from vector_store_manager import VectorStoreManager
            self.manager = VectorStoreManager(client=self.client, secrets=self.secrets)
        return self.manager
        
    def sync_job(self):
        """Job-Funktion fuer die Synchronisation."""
        print(f"\n⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starte geplante Synchronisation...")
        try:
            manager = self._get_manager()
            has_changes = manager.sync_vector_stores()
            if has_changes:
                print("✅ Synchronisation erfolgreich abgeschlossen - Aenderungen gefunden")
                print("⚠️  WICHTIG: Der Assistant sollte neu geladen werden!")
                print("    Rufe /admin/sync auf oder starte die App neu.")
            else:
                print("ℹ️  Keine Aenderungen gefunden")
        except Exception as e:
            print(f"❌ Fehler bei der Synchronisation: {e}")
            import traceback
            traceback.print_exc()
    
    def start(self, skip_initial_sync=False):
        """Startet den Scheduler im Hintergrund."""
        if self.running:
            print("⚠️  Scheduler laeuft bereits")
            return
        
        self.running = True
        
        # Optionale initiale Synchronisation
        if not skip_initial_sync:
            print("🚀 Starte Vector Store Sync Scheduler...")
            self.sync_job()
        else:
            print("🚀 Starte Vector Store Sync Scheduler (ohne initiale Synchronisation)...")
        
        # Plane regelmaeßige Synchronisation
        schedule.every(self.check_interval).minutes.do(self.sync_job)
        
        # Starte Scheduler-Thread
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        
        print(f"✅ Scheduler gestartet - Pruefung alle {self.check_interval} Minuten")
    
    def _run_scheduler(self):
        """Interne Funktion fuer den Scheduler-Thread."""
        while self.running:
            schedule.run_pending()
            time.sleep(1)
    
    def stop(self):
        """Stoppt den Scheduler."""
        print("🛑 Stoppe Scheduler...")
        self.running = False
        if self.thread:
            self.thread.join()
        print("✅ Scheduler gestoppt")
    
    def force_sync(self):
        """Erzwingt eine sofortige Synchronisation."""
        print("🔄 Erzwinge Synchronisation...")
        self.sync_job()

# Integration in die Flask-App
def integrate_with_flask(app, client, secrets):
    """Integriert den Sync-Scheduler in die Flask-App."""
    scheduler = VectorStoreSyncScheduler(
        client=client, 
        secrets=secrets,
        check_interval_minutes=30
    )
    
    # Starte Scheduler OHNE initiale Sync (wurde bereits beim Assistant-Laden gemacht)
    scheduler.start(skip_initial_sync=True)
    
    # Admin-Routes werden direkt in app.py definiert
    
    # Stoppe Scheduler beim Beenden
    def shutdown_scheduler(signum=None, frame=None):
        scheduler.stop()
        if signum:
            sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown_scheduler)
    signal.signal(signal.SIGTERM, shutdown_scheduler)
    
    return scheduler