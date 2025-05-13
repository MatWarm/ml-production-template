## Séance 2 – Récupération des données

### 1. Objectifs pédagogiques

* Maîtriser le **scraping** web simple avec BeautifulSoup.
* Consommer une **API REST** avec gestion de la pagination et de l’authentification.
* **Stocker** les données collectées dans une base (SQLite via SQLAlchemy ; MongoDB via PyMongo).
* Mettre en place des **tests** et une CI pour valider l’ingestion.

---

### 2. Préparation de l’environnement

1. **Mettre à jour `requirements.txt`**

   ```text
   beautifulsoup4
   requests
   sqlalchemy
   pymongo
   python-dotenv
   ```

2. **Installer les dépendances**

   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Structure du projet**

   ```
   ml-production-template/
   ├── config/
   │   ├── scraper.json        # paramétrage scraper web
   │   └── api_config.json     # endpoint, clé API, pagination
   ├── data/
   │   └── raw/                # fichiers collectés
   ├── scripts/
   │   ├── scrape_web.py       # à créer
   │   ├── fetch_api.py        # à créer
   │   └── store_data.py       # à créer
   ├── src/
   ├── tests/
   │   ├── test_scrape_web.py
   │   ├── test_fetch_api.py
   │   └── test_store_data.py
   └── … (autres dossiers déjà en place)
   ```

4. **Fichiers de config**

   * **`config/scraper.json`**

     ```json
     {
       "url": "https://example.com/articles",
       "selectors": {
         "container": "div.article",
         "title": "h2.title",
         "body": "div.content p"
       },
       "limit": 20
     }
     ```
   * **`config/api_config.json`**

     ```json
     {
       "endpoint": "https://api.example.com/v1/items",
       "headers": {
         "Authorization": "Bearer YOUR_API_KEY"
       },
       "params": {
         "page": 1,
         "per_page": 50
       },
       "max_pages": 5
     }
     ```

---

### 3. Live‑coding : Web Scraping

**`scripts/scrape_web.py`**

```python
import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path

def load_config(path="config/scraper.json"):
    return json.load(open(path))

def scrape():
    cfg = load_config()
    resp = requests.get(cfg["url"], timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    articles = soup.select(cfg["selectors"]["container"])[: cfg["limit"]]
    data = []
    for art in articles:
        title = art.select_one(cfg["selectors"]["title"]).get_text(strip=True)
        body = art.select_one(cfg["selectors"]["body"]).get_text(strip=True)
        data.append({"title": title, "body": body})
    return data

def save(data):
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(data):
        path = out_dir / f"article_{i}.json"
        path.write_text(json.dumps(item, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    items = scrape()
    save(items)
    print(f"{len(items)} articles extraits et sauvegardés dans data/raw/")
```

> **Action** : exécutez `python scripts/scrape_web.py`, vérifiez que `data/raw/article_0.json` existe et contient “title” & “body”.

---

### 4. Live‑coding : Appel d’une API REST

**`scripts/fetch_api.py`**

```python
import json, time
import requests
from pathlib import Path

def load_config(path="config/api_config.json"):
    return json.load(open(path))

def fetch_all():
    cfg = load_config()
    all_items = []
    for page in range(1, cfg["max_pages"] + 1):
        params = cfg["params"].copy()
        params["page"] = page
        resp = requests.get(cfg["endpoint"], headers=cfg["headers"], params=params, timeout=10)
        resp.raise_for_status()
        chunk = resp.json().get("data", [])
        if not chunk:
            break
        all_items.extend(chunk)
        time.sleep(0.5)  # pour ne pas surcharger le serveur
    return all_items

def save(data):
    out_dir = Path("data/raw/api")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "items.json").write_text(json.dumps(data, indent=2))

if __name__ == "__main__":
    items = fetch_all()
    save(items)
    print(f"{len(items)} éléments téléchargés et sauvegardés dans data/raw/api/items.json")
```

> **Action** : lancez `python scripts/fetch_api.py`, vérifiez `data/raw/api/items.json`.

---

### 5. Live‑coding : Stockage en base de données

**`scripts/store_data.py`**

```python
import json
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from pymongo import MongoClient

# --- SQL setup ---
Base = declarative_base()
class Article(Base):
    __tablename__ = "articles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255))
    body = Column(Text)

engine = create_engine("sqlite:///data/raw/data.db")
Session = sessionmaker(bind=engine)

# --- Mongo setup ---
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["mlops_db"]
mongo_col = mongo_db["items"]

def store_sql():
    Base.metadata.create_all(engine)
    session = Session()
    for file in Path("data/raw").glob("article_*.json"):
        obj = json.loads(file.read_text())
        session.add(Article(title=obj["title"], body=obj["body"]))
    session.commit()
    session.close()
    print("Articles stockés dans SQLite")

def store_mongo():
    items = json.loads(Path("data/raw/api/items.json").read_text())
    mongo_col.insert_many(items)
    print("Items stockés dans MongoDB")

if __name__ == "__main__":
    store_sql()
    store_mongo()
```

> **Action** :
>
> 1. Démarrez MongoDB local (`mongod`).
> 2. Exécutez `python scripts/store_data.py`.
> 3. Vérifiez dans SQLite (`sqlite3 data/raw/data.db`) :
>
>    ```sql
>    SELECT COUNT(*) FROM articles;
>    ```
> 4. Vérifiez dans Mongo (`mongo mlops_db --eval "db.items.count()"`).

---

### 6. Tests & CI

* **`tests/test_scrape_web.py`**

  ```python
  import os
  from scripts.scrape_web import scrape, save

  def test_scrape_and_save(tmp_path):
      data = scrape()
      save(data)
      files = os.listdir("data/raw")
      assert any(f.startswith("article_") for f in files)
  ```
* **`tests/test_fetch_api.py`**

  ```python
  from scripts.fetch_api import fetch_all
  def test_fetch_api_returns_list():
      items = fetch_all()
      assert isinstance(items, list)
  ```
* **`tests/test_store_data.py`**

  ```python
  import sqlite3
  def test_sql_db_exists():
      conn = sqlite3.connect("data/raw/data.db")
      cursor = conn.cursor()
      cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
      tables = [t[0] for t in cursor.fetchall()]
      assert "articles" in tables
  ```
* **CI** : vos nouveaux tests seront automatiquement lancés via le workflow `ci.yml`.

---

### 7. Exercices & livrables

* **En séance**

  1. `scrape_web.py`, `fetch_api.py` et `store_data.py` fonctionnels.
  2. Tests unitaires passants.
* **À la maison**

  1. **Robustifier** le scraping : gérer les erreurs réseau (retries), logger les exceptions.
  2. **Dockeriser** le pipeline d’ingestion : écrire un `Dockerfile` qui exécute les trois scripts dans l’ordre.
  3. Ouvrir 3 PRs :

     * `feature/robust-scraper`
     * `feature/docker-ingest`
     * `feature/tests-enhanced`

---

### 8. Ressources

* **Documentation**

  * BeautifulSoup : [https://www.crummy.com/software/BeautifulSoup/](https://www.crummy.com/software/BeautifulSoup/)
  * SQLAlchemy : [https://docs.sqlalchemy.org/](https://docs.sqlalchemy.org/)
  * PyMongo : [https://pymongo.readthedocs.io/](https://pymongo.readthedocs.io/)
* **Notebooks d’exemple**

  * `notebooks/2_scraping.ipynb`
  * `notebooks/2_api_fetch.ipynb`
* **Slides PDF**

  * `docs/Session2_Cours.pdf`
* **Branche solutions**

  * `solutions/session2`
