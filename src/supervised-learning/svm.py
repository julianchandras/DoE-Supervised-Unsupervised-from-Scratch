import numpy as np

class SVM():
    """
    Ref:
    https://adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-linear-svm/#google_vignette
    https://adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-duality-problem/
    https://adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-kernel-svm/
    https://adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-training-algorithms/
    """
    def __init__(self, C, kernel, degree):
        self.C = C
        if kernel == "poly":
            self.kernel = self._polynomial_kernel
            self.degree = degree
        elif kernel == "linear":
            self.kernel = self._linear_kernel
        else:
            self.kernel = self._rbf_kernel

        self.ones = None
        self.alpha = None
        self.b = None
        self.X = None
        self.y = None

    def _linear_kernel(self, X1, X2):
        return X1.dot(X2.T)

    def _rbf_kernel(self, X1, X2):
        return np.exp(-(np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis=2) **2)/2)
    
    def _polynomial_kernel(self, X1, X2):
        return (1 + X1.dot(X2.T)) ** self.degree

    def fit(self, X, y, eta=1e-3, epochs=500):
        # y sudah +1 / -1
        self.X = X
        self.y = y
        self.alpha = np.random.random(y.shape[0])
        self.b = 0
        self.ones = np.ones_like(y)
        self.stored_ys_and_kernel = np.outer(y,y) * self.kernel(X,X)

        for _ in range(epochs):
            gradient = self.ones - self.stored_ys_and_kernel.dot(self.alpha)
            self.alpha += eta * gradient
            self.alpha = np.clip(self.alpha, 0, self.C)

        index = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
        b_i = y[index] - (self.alpha * y).dot(self.kernel(X, X[index]))
        self.b = np.mean(b_i)
    
    def predict(self, X):
        return np.sign((self.alpha * self.y).dot(self.kernel(self.X, X)) + self.b)
    
class SVC():
    # Implementing One Vs One SVM for multiclass classification
    def __init__(self, C=1.0, kernel="rbf", degree=3):
        self.C = C
        self.kernel = kernel
        self.degree = degree

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        pairs = []
        for i in range(len(self.classes_)):
            for j in range(i + 1, len(self.classes_)):
                pairs.append(self.classes_[i], self.classes_[j])

        self.SVMs = []
        for class1, class2 in pairs:
            mask = np.logical_or(y == class1, y == class2)
            X_pair = X[mask]
            y_pair = y[mask]
            y_pair = np.where(y_pair == class1, 1, -1)

            svm = SVM(C=self.C, kernel=self.kernel, degree=self.degree)
            svm.fit(X_pair, y_pair)
            self.SVMs.append((class1, class2, svm))

    def predict(self, X):
        votes = np.zeros((X.shape[0], len(self.class_labels)))
        for class1, class2, svm in self.SVMs:
            predictions = svm.predict(X)
            votes[:, self.classes_ == class1] += (predictions == 1)
            votes[:, self.classes_ == class2] += (predictions == -1)

        return self.classes_[np.argmax(votes, axis=1)]
