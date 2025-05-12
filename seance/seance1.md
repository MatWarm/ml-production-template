## Séance 1 – Introduction & cadrage projet

### 1. Objectifs pédagogiques

* **Comprendre** le workflow MLOps : de la collecte de données au monitoring.
* **Choisir** et formuler un cas d’usage concret (NLP ou Computer Vision).
* **Initialiser** le projet en Git : repo, README, issues, branches, CI & tests.

---

### 2. Mise en place du projet

#### 2.1. Clonage et environnements

1. **Cloner** le dépôt template

   ```bash
   git clone https://github.com/votre-org/ml-production-template.git
   cd ml-production-template
   ```
2. **Créer** et activer un virtualenv

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Structure** du projet

   ```
   ml-production-template/
   ├── data/
   │   └── raw/                  # bruts
   ├── config/
   │   └── scraper.json          # à remplir
   ├── notebooks/
   ├── scripts/
   │   └── scraper_init.py       # à créer
   ├── src/
   ├── tests/
   ├── .github/
   │   ├── workflows/ci.yml
   │   └── ISSUE_TEMPLATE/
   ├── README.md
   ├── PROBLEM.md                # à remplir
   ├── pyproject.toml
   └── .flake8
   ```

#### 2.2. Templates de configuration

* **`pyproject.toml`**

  ```toml
  [tool.black]
  line-length = 88

  [tool.isort]
  profile = "black"
  ```
* **`.flake8`**

  ```ini
  [flake8]
  max-line-length = 88
  extend-ignore = E203
  ```
* **`config/scraper.json`** (exemple à adapter)

  ```json
  {
    "type": "NLP",
    "endpoint": "https://api.exemple.com/data",
    "params": { "q": "#yourhashtag", "limit": 50 }
  }
  ```
* **`.github/ISSUE_TEMPLATE/feature_request.md`**

  ```markdown
  ---
  name: Feature request
  about: Décrire une nouvelle fonctionnalité
  ---
  **Description**
  Décrivez clairement la fonctionnalité souhaitée.

  **Étapes pour reproduire**
  1. …
  2. …
  ```

---

### 3. Cadrage du cas d’usage

* **Fichier** `PROBLEM.md`

  ```markdown
  # Cadrage du projet

  ## Entrée
  - Type : image (224×224 JPEG) / texte (.txt)

  ## Sortie
  - Label binaire (0/1) ou multiclasses (N catégories)

  ## Métriques
  - Classification : Accuracy, F1‑score, ROC‑AUC
  - CV détection : mAP, IoU

  ## Contraintes
  - Latence < 200 ms par requête
  - Taille modèle < 100 Mo
  - Budget GPU : T4 max
  ```

> **Action** : chaque binôme remplit ce PROBLEM.md pour son cas avant la fin de séance.

---

### 4. Initialisation Git & README

1. **Nouvelle branche**

   ```bash
   git checkout -b feature/init-readme
   ```
2. **README.md** (extrait)

   ````markdown
   # Mon Projet MLOps

   Ce projet cible la classification d’images de déchets (plastique vs non‑plastique).

   ## Installation
   ```bash
   git clone … && cd …
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ````

   ## Usage

   ```bash
   python scripts/scraper_init.py --config config/scraper.json
   ```

   ![CI](https://github.com/votre-org/ml-production-template/actions/workflows/ci.yml/badge.svg)

   ```
   ```
3. **Commit & PR**

   ```bash
   git add README.md PROBLEM.md
   git commit -m "Init README + cadrage projet"
   git push origin feature/init-readme
   ```
4. **Review**

   * Assigner un pair pour approbation.
   * Vérifier le badge CI (qui va apparaître “pending”).

---

### 5. Issues & Kanban “Sprint 1”

* **Milestone** : “Sprint 1”

* **Issues à créer** (via GitHub)

  1. `data: scraper initial`

     * Tâche : implémenter `scripts/scraper_init.py` → télécharger 50 fichiers dans `data/raw/`.
  2. `data: connexion API / authentification`
  3. `infra: dockerfile de base`
  4. `ci: pipeline test minimal`

* **Board** :

  * Colonnes : To do / In progress / Done
  * Déplacer chaque issue selon l’avancement.

---

### 6. Conventions de code & CI

1. **Linting**

   ```bash
   pip install black flake8 isort
   black . && flake8
   isort .
   ```
2. **Tests**

   * Créer `tests/test_smoke.py`

     ```python
     def test_smoke():
         assert 1 + 1 == 2
     ```
3. **Workflow GitHub Actions**
   `.github/workflows/ci.yml`

   ```yaml
   name: CI
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - uses: actions/setup-python@v2
           with: { python-version: "3.8" }
         - run: pip install -r requirements.txt
         - run: black --check .
         - run: flake8
         - run: pytest --maxfail=1 --disable-warnings -q
   ```
4. **Vérification**

   * S’assurer que le badge du README passe au vert après merge.

---

### 7. Exercices & livrables

* **En séance**

  * PR `feature/init-readme` mergée.
  * 4 issues “Sprint 1” ouvertes et assignées.
* **À la maison**

  1. Implémenter `scripts/scraper_init.py` :

     ```python
     # scripts/scraper_init.py
     import json, os, requests
     from pathlib import Path

     def load_config(path="config/scraper.json"):
         return json.load(open(path))

     def main():
         cfg = load_config()
         os.makedirs("data/raw", exist_ok=True)
         params = cfg.get("params", {})
         resp = requests.get(cfg["endpoint"], params=params)
         for i, item in enumerate(resp.json()[:50]):
             fname = Path("data/raw") / f"{i}.json"
             fname.write_text(json.dumps(item))
         print("50 fichiers téléchargés dans data/raw/")

     if __name__ == "__main__":
         main()
     ```
  2. Ajouter un **test** `tests/test_scraper.py`

     ```python
     import os
     def test_scraper_creates_files():
         files = os.listdir("data/raw")
         assert len(files) >= 1
     ```
  3. Ouvrir une PR `feature/scraper-init`.

