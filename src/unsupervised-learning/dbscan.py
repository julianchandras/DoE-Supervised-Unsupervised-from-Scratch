import numpy as np

class DBSCAN():
    """
    Ref
    https://scrunts23.medium.com/dbscan-algorithm-from-scratch-in-python-475b82e0571c
    """
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.core_sample_indices_ = None
        self.components_ = None
        self.labels_ = None

    def fit(self, X):
        self.labels_ = np.zeros(X.shape[0], dtype=int)
        self.core_sample_indices_ = []

        current_claster_index = 0
        for point_index in range(self.labels_.shape[0]):
            if self.labels_[point_index] != 0:
                continue

            neighbors = self._get_neighbors(X, point_index)
            if neighbors.shape[0] >= self.min_samples:
                current_claster_index += 1
                self.core_sample_indices_.append(point_index)
                self._expand_cluster(X, point_index, neighbors, current_claster_index)
            else:
                self.labels_[point_index] = -1

        self.components_ = X[self.core_sample_indices_]
        self.labels_ -= 1


    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
    def _get_neighbors(self, X, point_index):
        distances = np.linalg.norm(X - X[point_index], axis=1)
        return np.where(distances < self.eps)[0]
    
    def _expand_cluster(self, X, point_index, neighbors, cluster_index):
        self.labels_[point_index] = cluster_index

        i = 0
        while i < len(neighbors):
            new_point_index = neighbors[i]

            if self.labels_[new_point_index] == -1:
                self.labels_[new_point_index] = cluster_index
            elif self.labels_[new_point_index] == 0:
                self.labels_[new_point_index] = cluster_index
                new_point_neighbors = self._get_neighbors(X, new_point_index)
                
                if new_point_neighbors.shape[0] >= self.min_samples:
                    self.core_sample_indices_.append(new_point_index)
                    neighbors = np.union1d(neighbors, new_point_neighbors)

            i += 1