## Séance 3 – Pré‑traitement des données

### 1. Objectifs pédagogiques

* **Nettoyer** un jeu de données brut (valeurs manquantes, doublons).
* **Transformer** les données selon le type :

  * **NLP** : tokenisation, suppression de stop‑words, lemmatisation.
  * **Vision** : redimensionnement, normalisation, augmentations basiques.
* **Composer** un pipeline scikit‑learn réutilisable (`Pipeline`).
* **Tester** et **CI‑fier** ces transformations.

---

### 2. Préparation de l’environnement

1. **Mise à jour** de `requirements.txt`

   ```text
   pandas
   numpy
   scikit-learn
   nltk
   spacy
   Pillow
   opencv-python
   ```

2. **Installation**

   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Structure attendue**

   ```
   ml-production-template/
   ├── config/
   │   └── preprocessing.json     # paramètres (cols à imputer, stop‑words…)
   ├── data/
   │   └── raw/                   # CSV/JSON bruts
   ├── scripts/
   │   └── preprocess.py          # à créer
   ├── src/
   │   └── preprocessing.py       # code réutilisable
   ├── tests/
   │   ├── test_preprocess.py
   │   └── test_text_pipeline.py
   ├── notebooks/
   │   └── 3_preprocessing_demo.ipynb
   └── … (CI, README, etc.)
   ```

4. **Fichier de config** `config/preprocessing.json`

   ```json
   {
     "numeric": ["age", "price", "quantity"],
     "categorical": ["gender", "country"],
     "text": ["review_text"],
     "image_size": [224, 224]
   }
   ```

---

### 3. Live‑coding : script de pré‑traitement

#### 3.1. `scripts/preprocess.py`

```python
#!/usr/bin/env python3
import json
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# NLP
import spacy
from sklearn.base import TransformerMixin

# Vision
from PIL import Image
import cv2

# --- Chargement de la config ---
def load_config(path="config/preprocessing.json"):
    return json.load(open(path))

# --- Nettoyage général ---
def drop_duplicates_and_na(df):
    df = df.drop_duplicates()
    df = df.dropna(subset=cfg["numeric"] + cfg["categorical"], how="all")
    return df

# --- Transformateur NLP custom ---
class SpacyTokenizer(TransformerMixin):
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)
    def fit(self, X, y=None): return self
    def transform(self, texts):
        return [" ".join([tok.lemma_ for tok in self.nlp(text) if not tok.is_stop])
                for text in texts]

# --- Pré‑traitement tabulaire via scikit-learn ---
def build_tabular_pipeline(cfg):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, cfg["numeric"]),
        ("cat", categorical_pipeline, cfg["categorical"])
    ], remainder="drop")
    return preprocessor

# --- Pré‑traitement texte via pipeline scikit ---
def build_text_pipeline(cfg):
    return Pipeline([
        ("tokenizer", SpacyTokenizer()),
        ("vectorizer", 
         __import__("sklearn.feature_extraction.text").feature_extraction.text.TfidfVectorizer(
             max_features=5000))
    ])

# --- Transformations d’images ---
def process_image(path, size):
    img = Image.open(path).convert("RGB")
    img = img.resize((size[0], size[1]))
    arr = np.array(img) / 255.0
    return arr

def main():
    cfg = load_config()
    # 1) Lecture
    df = pd.read_csv("data/raw/data.csv")
    # 2) Nettoyage
    df = drop_duplicates_and_na(df)
    # 3) Tabulaire
    tab_pipe = build_tabular_pipeline(cfg)
    X_tab = tab_pipe.fit_transform(df)
    # 4) Texte
    text_pipe = build_text_pipeline(cfg)
    X_text = text_pipe.fit_transform(df[cfg["text"][0]])
    # 5) Images (exemple)
    # img_arr = process_image("data/raw/img_0.jpg", cfg["image_size"])
    print("Pré‑traitement terminé :",
          X_tab.shape, X_text.shape)

if __name__ == "__main__":
    main()
```

---

### 4. Tests unitaires

#### `tests/test_preprocess.py`

```python
import pandas as pd
import numpy as np
from scripts.preprocess import drop_duplicates_and_na

def test_drop_duplicates_and_na(tmp_path):
    df = pd.DataFrame({
        "age": [25, 25, np.nan, 40],
        "country": ["fr", "fr", None, "de"]
    })
    cfg = {"numeric": ["age"], "categorical": ["country"]}
    # on monkey‑patch cfg global si nécessaire
    from scripts.preprocess import cfg as global_cfg
    global_cfg = cfg

    clean = drop_duplicates_and_na(df)
    assert clean.shape[0] == 2  # lignes uniques et non-tout-NA
```

#### `tests/test_text_pipeline.py`

```python
from scripts.preprocess import build_text_pipeline

def test_text_pipeline_outputs_list():
    pipe = build_text_pipeline({"text": ["Hello world"]})
    out = pipe.fit_transform(["Hello world"])
    assert hasattr(out, "toarray")
```

Vos tests seront déclenchés automatiquement dans le CI existant.

---

### 5. Exercices & livrables

* **En séance**

  1. Implémenter et exécuter `scripts/preprocess.py` sur le dataset `data/raw/data.csv`.
  2. Vérifier que `X_tab` et `X_text` sont générés sans erreur.
* **À la maison**

  1. **Ajoutez** une étape de **sélection de variables** (seulement les 10 plus corrélées à la target).
  2. **Enrichissez** le pipeline image : ajouter une rotation aléatoire et un flip horizontal.
  3. Ouvrir 2 PRs :

     * `feature/feature-selection`
     * `feature/image-augmentations`

---

### 6. Ressources

* **Documentation**

  * scikit‑learn Pipelines : [https://scikit-learn.org/stable/modules/compose.html](https://scikit-learn.org/stable/modules/compose.html)
  * spaCy Tokenizer : [https://spacy.io/usage](https://spacy.io/usage)
  * PIL vs OpenCV : [https://pillow.readthedocs.io/](https://pillow.readthedocs.io/) / [https://opencv.org/](https://opencv.org/)
* **Notebooks**

  * `notebooks/3_preprocessing_demo.ipynb`
* **Slides PDF**

  * `docs/Session3_Cours.pdf`
* **Branche solutions**

  * `solutions/session3`
