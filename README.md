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
├── app.py                   # Haupteinstiegspunkt: Flask-App mit HTTP-Endpunkten (Chat & Admin)
├── chatbot_logic.py         # Kernlogik des Chatbots: Erstellung des Assistants, Beantwortung von Fragen, Verwaltung von Threads
├── sync_scheduler.py        # Hintergrundprozess: plant und steuert die regelmäßige Synchronisation von Dokumenten
├── vector_store_manager.py  # Verwaltung der Vektordatenbank: Import, Aktualisierung und Abgleich von PDF-Inhalten
├── templates/               # HTML-Vorlagen für die Benutzeroberfläche
│   ├── index.html           # Chat-Frontend für Nutzer
│   ├── admin.html           # Admin-Panel für Verwaltung und Monitoring
├── static/                  # Statische Dateien (CSS, Bilder, JS)
│   ├── style.css            # Stylesheet für die Chat-Oberfläche
│   ├── admin.css            # Stylesheet für das Admin-Panel
├── .secrets/                # Konfigurationsordner (lokal, nicht ins Repo einchecken)
│   └── secrets.toml         # API-Keys und weitere geheime Einstellungen
├── requirements.txt         # Liste der Python-Abhängigkeiten
└── README.md                # Projektdokumentation und Installationshinweise
```

---

## 🔧 Installation

Voraussetzungen
- Python ≥ 3.9 (empfohlen: 3.10 oder 3.11)
- pip zur Installation von Abhängigkeiten
- Internetverbindung (für die Nutzung der OpenAI-Schnittstelle)

Das Programm ist betriebssystemunabhängig und läuft auf Windows, macOS und Linux, solange die passende Python-Version installiert ist.

### 1. Repository klonen
  ```bash
  git clone https://github.com/LauraFLopes/FHATGPT.git
  cd chatbot
   ```

### 2. Virtuelle Umgebung erstellen

Windows
  ```bash
  python -m venv venv
  venv\Scripts\activate
   ```

Linux / macOS
  ```bash
  python3 -m venv venv
  source venv/bin/activate
   ```

3. Abhängigkeiten installieren
  ```bash
pip install -r requirements.txt
   ```

4. OpenAI API-Key hinterlegen
1. API-Key auf https://platform.openai.com/api-keys generieren
2. Datei .secrets/secrets.toml anlegen (Ordner muss existieren)
3. API-Key dort eintragen:
   ```bash
   API_KEY = "dein-openai-api-key"
   ADMIN_TOKEN = "ein-geheimes-admin-token"
   VECTOR_STORE_ID = ""   # wird beim ersten Sync erzeugt
   ASSISTANT_ID = ""      # wird beim ersten Sync erzeugt
   ```

5. Programm starten
  ```bash
  python app.py
  ```

Chatbot ist erreichbar unter: http://localhost:5000
Admin-Panel: http://localhost:5000/admin






1. Repository klonen:
   ```bash
   git clone https://github.com/LauraFLopes/FHATGPT.git
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
