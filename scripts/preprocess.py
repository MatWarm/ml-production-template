import json
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# NLP
import spacy
from sklearn.base import TransformerMixin


def load_config(path=None):
    if path is None:
        # Chemin absolu vers le dossier racine du projet
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        print(root_dir)
        path = os.path.join(root_dir, "config", "preprocessing.json")
    with open(path, "r") as f:
        return json.load(f)
            

# --- Nettoyage général ---
def drop_duplicates_and_na(df, cfg):
    df = df.drop_duplicates()
    df = df.dropna(subset=cfg["numeric"], how="all")
    return df

class SpacyTokenizer(TransformerMixin):
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)
    def fit(self, X, y=None): return self
    def transform(self, texts):
        return [" ".join([tok.lemma_ for tok in self.nlp(text) if not tok.is_stop])
                for text in texts]

def remove_stopwords(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])


def build_tabular_pipeline(cfg):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, cfg["numeric"])
    ], remainder="drop")
    return preprocessor

def build_text_pipeline(cfg):
    return Pipeline([
        ("tokenizer", SpacyTokenizer()),
        ("vectorizer", 
         __import__("sklearn.feature_extraction.text").feature_extraction.text.TfidfVectorizer(
             max_features=5000))
    ])


def eval_cluster(embedding, df):
    kmeans = KMeans(n_clusters=9, random_state=42)
    y_pred = kmeans.fit_predict(embedding)
    
    # Evaluate the performance using ARI, NMI, and FMI
    ari = adjusted_rand_score(df, y_pred)
    nmi = normalized_mutual_info_score(df, y_pred)
    fmi = fowlkes_mallows_score(df, y_pred)

    # Print Metrics scores
    print("Adjusted Rand Index (ARI): {:.3f}".format(ari))
    print("Normalized Mutual Information (NMI): {:.3f}".format(nmi))
    print("Fowlkes-Mallows Index (FMI): {:.3f}".format(fmi))


def test_cluster_range(embedding, k_range):
    results = []
    if hasattr(embedding, "toarray"):
        embedding = embedding.toarray()

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embedding)

        sil = silhouette_score(embedding, labels)
        ch = calinski_harabasz_score(embedding, labels)
        db = davies_bouldin_score(embedding, labels)

        results.append({
            "k": k,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db
        })
    # Convert results to DataFrame
    results = pd.DataFrame(results)
    print(results.sort_values("silhouette", ascending=False))
    ks = results["k"]
    
    plt.figure(figsize=(12, 5))

    # Silhouette
    plt.subplot(1, 3, 1)
    plt.plot(ks, results["silhouette"], marker='o')
    plt.title("Silhouette Score")
    plt.xlabel("k")
    plt.ylabel("Score")

    # Calinski-Harabasz
    plt.subplot(1, 3, 2)
    plt.plot(ks, results["calinski_harabasz"], marker='o', color='green')
    plt.title("Calinski-Harabasz Index")
    plt.xlabel("k")

    # Davies-Bouldin
    plt.subplot(1, 3, 3)
    plt.plot(ks, results["davies_bouldin"], marker='o', color='red')
    plt.title("Davies-Bouldin Index")
    plt.xlabel("k")

    plt.tight_layout()
    plt.show()


def clustering(embedding, n_clusters= 90):
    kmeans = KMeans(n_clusters, random_state=42)

    # fit the model
    kmeans.fit(embedding)
    clusters = kmeans.labels_
    return clusters

    

def show_df(df):
    counts = df["ACTION"].value_counts()
    duplicates = counts[counts > 5]
    print(f"{duplicates.describe()}")


def show_histo(df, colonne):
    action_counts = df[colonne].value_counts()
    filtered_actions = action_counts[action_counts > 20]

    filtered_actions.plot(kind='bar')
    plt.title("Distribution of ACTION Column (> 200 occurrences)")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.ylim(0, 1000)  # Limite l’axe Y à 1000
    plt.xticks(rotation=45, ha='right')  # Améliore la lisibilité
    plt.tight_layout()
    plt.show()

def plot_elbow(X, max_k=100, metric='inertia'):
    """
    Trace la méthode du coude pour choisir le bon nombre de clusters.
    
    :param X: matrice des vecteurs (TF-IDF, embeddings, etc.)
    :param max_k: nombre maximum de clusters à tester
    :param metric: 'inertia' ou 'silhouette'
    """
    scores = []
    ks = range(2, max_k + 1)

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        if metric == 'inertia':
            scores.append(kmeans.inertia_)  # somme des distances intra-cluster
        elif metric == 'silhouette':
            score = silhouette_score(X, labels)
            scores.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, scores, 'bo-', linewidth=2)
    plt.xlabel("Nombre de clusters (k)")
    if metric == 'inertia':
        plt.ylabel("Inertie intra-cluster")
        plt.title("Méthode du coude (Inertie)")
    else:
        plt.ylabel("Score de silhouette")
        plt.title("Score de silhouette moyen")
    plt.grid(True)
    plt.show()