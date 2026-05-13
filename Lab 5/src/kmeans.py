import numpy as np

class KMeans:
    # Thuật toán K-Means để phân cụm dữ liệu
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
    
    # Hàm fit để tìm tâm cụm tối ưu dựa trên dữ liệu đầu vào
    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, _ = X.shape
        random_idxs = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idxs]

        for i in range(self.max_iters):
            old_centroids = np.copy(self.centroids)
            self.labels = self._e_step(X)
            self._m_step(X)
            shift = np.linalg.norm(self.centroids - old_centroids)
            if shift < self.tol:
                print(f"K-Means hội tụ sau {i + 1} vòng lặp.")
                break
    
    def predict(self, X):
        return self._e_step(X)
    
    # E-step: Gán nhãn dựa trên khoảng cách đến các tâm cụm
    def _e_step(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    # M-step: Cập nhật tâm cụm dựa trên các điểm dữ liệu được gán nhãn
    def _m_step(self, X):
        for k in range(self.k):
            cluster_points = X[self.labels == k]
            if len(cluster_points) > 0:
                self.centroids[k] = np.mean(cluster_points, axis=0)