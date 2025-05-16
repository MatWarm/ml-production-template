import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

