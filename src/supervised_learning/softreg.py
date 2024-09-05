import numpy as np

class LogisticRegression():
    def __init__(self, learning_rate, num_iter=100, regularization="l2", lambda_=0.1):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.regularization = regularization
        self.lambda_ = lambda_

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_mapped = np.array([np.where(self.classes_ == label)[0][0] for label in y])

        self.X = np.array(X)
        self.y = y_mapped

        self.theta = np.zeros((self.X.shape[1], len(self.classes_)))
    
        for _ in range(self.num_iter):
            gradient = self._compute_gradient()
            self.theta -= self.learning_rate * gradient

    def predict(self, X):
        X = np.array(X)
        Z = X @ self.theta
        Z -= np.max(Z, axis=1).reshape(-1, 1)
        e_z = np.exp(Z)
        probs = e_z / np.sum(e_z, axis=1, keepdims=True)

        class_indices = np.argmax(probs, axis=1)
        return self.classes_[class_indices]
    
    def _compute_gradient(self):
        """
        Ref: http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
        Pada referensi juga dibahas tentang validitas mencapai numerical stability dengan mengurangi Z dengan value yang sama)
        """
        m = self.X.shape[0]

        Z = self.X @ self.theta
        Z = Z - np.max(Z, axis=1, keepdims=True)
        e_z = np.exp(Z)
        probs = e_z / np.sum(e_z, axis=1, keepdims=True)

        y_true = np.zeros_like(probs)
        y_true[np.arange(m), self.y] = 1

        gradient = - self.X.T @ (y_true - probs)

        if self.regularization == "l2":
            gradient += self.lambda_ * self.theta
        elif self.regularization == "l1" :
            gradient += self.lambda_ * np.sign(self.theta)
        return gradient