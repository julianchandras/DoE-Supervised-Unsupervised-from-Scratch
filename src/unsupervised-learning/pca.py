import numpy as np

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X = X - self.mean_
        cov = np.cov(X.T)

        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T

        indices = np.argsort(eigenvalues)[::-1]
        self.explained_variance_ = eigenvalues[indices]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(self.explained_variance_)

        eigenvectors = eigenvectors[:, indices]
        self.components_ = eigenvectors[:, :self.n_components].T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)