# FH Wedel Chatbot 🤖

Ein Chatbot für die FH Wedel, der Fragen anhand lokaler PDF-Dokumente beantwortet.  
Die PDFs werden regelmäßig mit einem OpenAI Vector Store synchronisiert und können über ein Webinterface abgefragt werden.  
Außerdem gibt es ein Admin-Panel für Synchronisation und Systemverwaltung.

---

## 🚀 Features

- Chatbot mit **Streaming-Antworten** (Markdown wird unterstützt)
- **Automatische Synchronisation** von lokalen PDFs mit einem OpenAI Vector Store
- **Admin-Panel** mit Login-Token:
  - Vector Store Status abrufen
  - Synchronisation manuell starten
  - System-Reset durchführen
- Sitzungsverwaltung (temporär oder persistent)
- Hintergrund-Scheduler für periodische Sync-Jobs

---

## 📂 Projektstruktur

```
.
├── app.py                 # Flask-App (HTTP-Endpunkte, Admin, Chat)
├── chatbot_logic.py       # Logik für Assistant, Fragen stellen, Threads
├── sync_scheduler.py      # Hintergrund-Scheduler für PDF-Synchronisation
├── templates/
│   ├── index.html         # Chat-Oberfläche
│   ├── admin.html         # Admin-Panel
├── static/
│   ├── style.css          # Styles für Chat
│   ├── admin.css          # Styles für Admin-Panel
├── .secrets/              # Konfigurationsordner
│   └── secrets.toml       # API-Keys und Einstellungen
├── requirements.txt       # Python-Abhängigkeiten
└── README.md              # Projektbeschreibung
```

---

## 🔧 Installation

1. Repository klonen:
   ```bash
   git clone https://github.com/dein-user/chatbot.git
   cd chatbot
   ```

2. Virtuelle Umgebung erstellen und aktivieren:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
   ```

3. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

4. Konfiguration in `.secrets/secrets.toml` anlegen:
   ```toml
   API_KEY = "dein-openai-api-key"
   ADMIN_TOKEN = "ein-geheimes-admin-token"
   VECTOR_STORE_ID = ""   # wird beim ersten Sync erzeugt
   ASSISTANT_ID = ""      # wird beim ersten Sync erzeugt
   ```

---

## ▶️ Starten

```bash
python app.py
```

- Chatbot ist erreichbar unter: [http://localhost:5000](http://localhost:5000)  
- Admin-Panel: [http://localhost:5000/admin](http://localhost:5000/admin)  

---

## ⚙️ Admin-Panel

Funktionen:
- **📊 Status abrufen**: Zeigt aktuellen Vector Store, Kategorien und Dateigrößen  
- **🔄 Synchronisation starten**: Löst sofortigen Sync mit OpenAI aus  
- **🚨 Reset**: Löscht alle Vector Stores, Dateien und den Assistant (Vorsicht!)  

---

## 🗂 PDF-Synchronisation

- Lokale PDFs liegen im `pdf/`-Ordner
- Änderungen werden automatisch erkannt und regelmäßig synchronisiert (Intervall: Standard 60 Min.)  
- Manuelle Synchronisation über Admin-Panel möglich  

---

## 👩‍💻 Entwicklung

- **Frontend**: HTML + CSS + Vanilla JS (mit SSE für Streaming)  
- **Backend**: Flask + OpenAI Python SDK  
- **Scheduler**: `schedule`-Library für Hintergrundjobs  

---

## 🛡️ Sicherheit

- Zugriff aufs Admin-Panel nur mit gültigem `ADMIN_TOKEN`  
- Sitzungs-Timeout: 15 Minuten (optional „Angemeldet bleiben“ für 30 Tage)  
- Empfehlung: Projekt nur hinter einer abgesicherten Umgebung laufen lassen (z. B. VPN oder Passwortschutz)  

---
