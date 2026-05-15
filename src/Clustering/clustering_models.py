# ==========================================================
# CLUSTERING MODELS
# Supports KMeans + DBSCAN
# ==========================================================

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


class ClusteringModels:

    def __init__(self):
        self.results = []

    def train(self, X):

        models = {
            "KMeans": KMeans(n_clusters=3, random_state=42),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5)
        }

        best_model = None
        best_score = -1
        best_name = None
        best_labels = None

        for name, model in models.items():

            labels = model.fit_predict(X)

            # Skip if only 1 cluster (invalid for silhouette)
            if len(set(labels)) <= 1:
                score = -1
            else:
                score = silhouette_score(X, labels)

            self.results.append({
                "model": name,
                "score": score
            })

            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
                best_labels = labels

        return best_model, best_name, best_score, best_labels

    def get_results(self):
        return self.results