# Plán vytvoření Flask webové aplikace

## Analýza současného stavu projektu
- Projekt aktuálně používá FastAPI jako hlavní web framework
- V requirements.txt není Flask, bude potřeba ho přidat
- Struktura projektu je připravena pro webové aplikace

## Kroky implementace

### 1. Příprava závislostí
- Přidat Flask do requirements.txt
- Instalovat Flask pomocí pip

### 2. Vytvoření základní Flask aplikace
- Vytvořit soubor app.py v kořenovém adresáři
- Implementovat základní Flask aplikaci s route "/"
- Route bude vracet "Hello, World!"

### 3. Struktura app.py
```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
```

### 4. Kontrola kvality
- Použít lint nástroj pro kontrolu syntaxe a stylu kódu
- Ověřit funkčnost aplikace

### 5. Dokumentace
- Přidat instrukce pro spuštění do README nebo komentářů

## Očekávané výsledky
- Funkční Flask aplikace dostupná na localhost:5000
- Zobrazení "Hello, World!" na hlavní stránce
- Čistý a kvalitní kód odpovídající standardům
