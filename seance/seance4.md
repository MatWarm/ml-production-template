## Séance 4 – Exploration de données & featurisation

### 1. Objectifs pédagogiques

* Réaliser une **Analyse Exploratoire (EDA)** structurée : distributions, corrélations et visualisations clés.
* Construire des **features** pertinentes : variables uni-/bivariées, TF-IDF pour texte et binning pour numérique.
* Séparer rigoureusement les jeux **train/validation/test** (StratifiedShuffleSplit).
* Intégrer le tout dans un **notebook reproductible** et un **pipeline** de featurisation.

---

### 2. Préparation de l’environnement

1. **Mise à jour** de `requirements.txt`

   ```text
   matplotlib
   seaborn
   scipy
   scikit-learn
   nltk
   ```

2. **Installation**

   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Structure attendue**

   ```
   ml-production-template/
   ├── data/
   │   └── processed/           # CSV après pré-traitement
   ├── notebooks/
   │   └── 4_eda_featurization.ipynb
   ├── src/
   │   └── features.py         # fonctions de featurisation
   ├── tests/
   │   └── test_features.py
   └── … (scripts, config déjà en place)
   ```

4. **Notebook de démo**

   * `notebooks/4_eda_featurization.ipynb` contiendra toutes les étapes.

---

### 3. Live-coding : Notebook EDA & featurisation

#### 3.1. Chargement et aperçu

```python
import pandas as pd
df = pd.read_csv("data/processed/data_clean.csv", parse_dates=["timestamp"])
df.head(), df.info()
```

#### 3.2. Statistiques descriptives

```python
# Numeric summary
num_cols = df.select_dtypes(include=["int64","float64"]).columns
df[num_cols].describe().T

# Categorical counts
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    print(col, df[col].nunique(), "valeurs uniques")
```

#### 3.3. Visualisations avec matplotlib

> *Chaque graphique dans une cellule distincte*

```python
import matplotlib.pyplot as plt

# Distribution d'une variable numérique
plt.figure()
plt.hist(df["price"].dropna(), bins=30)
plt.title("Distribution des prix")
plt.xlabel("Prix")
plt.ylabel("Fréquence")
plt.show()
```

```python
# Heatmap de corrélation
plt.figure()
corr = df[num_cols].corr()
plt.imshow(corr, interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(num_cols)), num_cols, rotation=90)
plt.yticks(range(len(num_cols)), num_cols)
plt.title("Matrice de corrélation")
plt.show()
```

#### 3.4. Feature engineering

* **Univariée** (ex. log-transform)

  ```python
  import numpy as np
  df["log_price"] = np.log1p(df["price"])
  ```
* **Binning** (histogram binning)

  ```python
  df["price_bin"] = pd.qcut(df["price"], q=5, labels=False)
  ```
* **TF-IDF** pour texte

  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer

  tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
  X_text = tfidf.fit_transform(df["review_text"].fillna(""))
  print("TF-IDF shape:", X_text.shape)
  ```
* **Embeddings simples** (optionnel)

  ```python
  # placeholder pour pré-entraînés
  ```

#### 3.5. Split train/validation/test

```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(df, df["target"]):
    train_set = df.loc[train_idx]
    test_set  = df.loc[test_idx]

# Validation séparée du train
split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_idx, val_idx in split2.split(train_set, train_set["target"]):
    train, val = train_set.loc[train_idx], train_set.loc[val_idx]
```

---

### 4. Code réutilisable : `src/features.py`

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def add_log_feature(df, col):
    df[col + "_log"] = np.log1p(df[col])
    return df

def add_price_bins(df, col, q=5):
    df[col + "_bin"] = pd.qcut(df[col], q=q, labels=False)
    return df

def build_tfidf(corpus, max_features=1000):
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = tfidf.fit_transform(corpus.fillna(""))
    return X, tfidf
```

---

### 5. Tests unitaires : `tests/test_features.py`

```python
import pandas as pd
import numpy as np
from src.features import add_log_feature, add_price_bins, build_tfidf

def test_add_log_feature():
    df = pd.DataFrame({"price":[0, 9, 99]})
    df2 = add_log_feature(df.copy(), "price")
    assert "price_log" in df2.columns
    assert np.isclose(df2.loc[1, "price_log"], np.log1p(9))

def test_add_price_bins():
    df = pd.DataFrame({"price":[10, 20, 30, 40, 50]})
    df2 = add_price_bins(df.copy(), "price", q=5)
    assert df2["price_bin"].nunique() == 5

def test_build_tfidf():
    corpus = pd.Series(["a b c", "b c d", "c d e"])
    X, tfidf = build_tfidf(corpus, max_features=3)
    assert X.shape == (3, 3)
    assert hasattr(tfidf, "transform")
```

---

### 6. Exercices & livrables

* **En séance**

  1. Exécuter l’EDA complet dans le notebook et commenter les insights.
  2. Générer au moins deux nouvelles features et visualiser leur distribution.
  3. Créer les splits train/val/test et vérifier la répartition de la target.

* **À la maison**

  1. **Automatiser** l’EDA : écrire une fonction qui génère et sauvegarde les graphiques pour toutes les variables numériques.
  2. **Intégrer** les fonctions de `src/features.py` dans un pipeline scikit-learn (`FeatureUnion`).
  3. Ouvrir 2 PRs :

     * `feature/automated-eda`
     * `feature/pipeline-features`

---

### 7. Ressources

* **Documentation**

  * matplotlib : [https://matplotlib.org/stable/tutorials/introductory/pyplot.html](https://matplotlib.org/stable/tutorials/introductory/pyplot.html)
  * StratifiedShuffleSplit : [https://scikit-learn.org/stable/modules/generated/sklearn.model\_selection.StratifiedShuffleSplit.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html)
  * pandas qcut : [https://pandas.pydata.org/docs/reference/api/pandas.qcut.html](https://pandas.pydata.org/docs/reference/api/pandas.qcut.html)

* **Notebooks**

  * `notebooks/4_eda_featurization.ipynb` (démo + template)

* **Slides PDF**

  * `docs/Session4_Cours.pdf`

* **Branche solutions**

  * `solutions/session4`

---

## Présentation Gamma enrichie

1. **Slide 1 – Titre**

   * **Session 4 – EDA & featurisation**
   * Sous-titre : “Insights • Features • Splits”
   * Icônes graphiques, histogramme

2. **Slide 2 – Objectifs**

   * Bullets animés :

     * EDA structurée
     * Création de features
     * Split train/val/test

3. **Slide 3 – Statistiques & Aperçu**

   * Capture d’écran du `df.describe().T`
   * Mini‐tableau des variables clés

4. **Slide 4 – Distribution & Corrélation**

   * Deux visuels côte à côte :

     * Histogramme `price`
     * Heatmap corrélation

5. **Slide 5 – Feature Engineering**

   * Code snippets pour `add_log_feature` et `price_bin`
   * Avant/après distribution de `price_log`

6. **Slide 6 – TF-IDF**

   * Workflow textuel : “Corpus → TF-IDF → matrice creuse”
   * Snippet et dimension

7. **Slide 7 – Splits stratifiés**

   * Diagramme fléché train → val → test
   * Vérification des proportions

8. **Slide 8 – Tests & CI**

   * Extrait d’un test de `test_features.py`
   * Badge CI

9. **Slide 9 – Exercices & tâches**

   * Liste des PRs à créer
   * Lien vers notebook template

10. **Slide 10 – Ressources & Q\&R**

    * Liens cliquables et contacts

---

Avec cette structure, votre séance 4 sera à la fois très concrète (code exécutable) et pédagogique (visualisations et bonnes pratiques). Dites-moi si vous souhaitez ajuster quelque chose !
