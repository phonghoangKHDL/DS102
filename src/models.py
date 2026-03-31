import numpy as np

class LogisticRegressionGD:
    def __init__(self, lr=0.1, epochs=50):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.epochs):
            z = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(z)

            loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            self.losses.append(loss)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            if i % 10 == 0:
                print(f"Epoch {i}: Loss = {loss:.4f}")

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        y_pred = self.sigmoid(z)
        return np.where(y_pred > 0.5, 1, 0)
    
    
class SoftmaxRegressionGD:
    def __init__(self, lr=0.1, epochs=50, n_classes=10):
        self.lr = lr
        self.epochs = epochs
        self.n_classes = n_classes
        self.w = None
        self.b = None
        self.losses = []

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def to_one_hot(self, y):
        one_hot = np.zeros((y.size, self.n_classes))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros((n_features, self.n_classes))
        self.b = np.zeros(self.n_classes)
        y_one_hot = self.to_one_hot(y)

        for i in range(self.epochs):
            z = np.dot(X, self.w) + self.b
            y_pred = self.softmax(z)

            loss = -np.mean(np.sum(y_one_hot * np.log(y_pred + 1e-15), axis=1))
            self.losses.append(loss)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y_one_hot))
            db = (1/n_samples) * np.sum(y_pred - y_one_hot, axis=0)

            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            if i % 10 == 0:
                print(f"Epoch {i}: Loss = {loss:.4f}")

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        y_pred = self.softmax(z)
        return np.argmax(y_pred, axis=1) 