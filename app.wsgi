import sys
import site

# venv-Pfad und site-packages zur sys.path hinzufügen
venv_path = '/var/www/FHATGPT/venv'
site.addsitepackages(venv_path + '/lib/python3.11/site-packages')

# Flask-Projektpfad zur sys.path hinzufügen
sys.path.insert(0, '/var/www/FHATGPT')

# Importiere die Flask-App (sie muss in app.py liegen und "app" heißen)
from app import app as application