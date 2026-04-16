import numpy as np

class SoftMarginSVM:
    def __init__(self, C=1.0, lr=0.001, epochs=100):
        self.C = C           
        self.lr = lr         
        self.epochs = epochs
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for idx, x_i in enumerate(X_shuffled):
                y_i = y_shuffled[idx] 
                condition = y_i * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * 1e-4 * self.w) 
                else:
                    self.w -= self.lr * (2 * 1e-4 * self.w - np.dot(x_i, y_i) * self.C)
                    self.b -= self.lr * (-y_i * self.C)

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.where(linear_output >= 0, 1, -1)