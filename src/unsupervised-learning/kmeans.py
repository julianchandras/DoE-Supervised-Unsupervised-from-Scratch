import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, init="k-means++"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.cluster_centers_ = None
        self.label_ = None

    def fit(self, X):
        # ndarray of shape (n_clusters, n_features)
        if self.init == "random":
            self.cluster_centers_ = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        else:
            self.cluster_centers_ = self._init_kmeans_plusplus(X)

        for i in range(self.max_iter):
            # add new dimention/axis for broadcasting
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
            self.label_ = np.argmin(distances, axis=1)

            new_cluster_centers = np.array([
                X[np.where(self.label_ == j)].mean(axis=0) if np.any(self.label_ == j) else self.cluster_centers_[j]
                for j in range(self.n_clusters)
            ])

            difference = new_cluster_centers - self.cluster_centers_
            frobenius_norm = np.linalg.norm(difference, 'fro')

            self.cluster_centers_ = new_cluster_centers

            if frobenius_norm < self.tol:
                break

    def predict(self, X):
        return np.argmin(np.linalg.norm(X[:, np.newaxis]-self.cluster_centers_, axis=2), axis=1)
    
    def _init_kmeans_plus_plus(self, X):
        # https://en.wikipedia.org/wiki/K-means%2B%2B
        remaining_points = X
        cluster_centers = np.zeros((self.n_clusters, X.shape[1]))
        idx = np.random.choice(remaining_points.shape[0], size=1)
        cluster_centers[0] = X[idx]
        remaining_points = np.delete(remaining_points, idx[0], axis=0)

        for i in range(1, self.n_clusters):
            distances = np.linalg.norm(cluster_centers[i-1] - remaining_points, axis=1)
            probabilities = (distances**2)/np.sum(distances**2)

            idx = np.random.choice(remaining_points.shape[0], size=1, p=probabilities)
            cluster_centers[i] = remaining_points[idx]
            remaining_points = np.delete(remaining_points, idx[0], axis=0)

        return cluster_centers
