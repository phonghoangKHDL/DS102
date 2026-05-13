import numpy as np

class GaussianMixtureModel:
    # Thuật toán GMM sử dụng EM để phân cụm dữ liệu
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        
        self.pi = None      # Mixing coefficients 
        self.mu = None      # Tâm cụm 
        self.sigma = None   # Ma trận hiệp phương sai 
        self.gamma = None   # Trọng số mềm 
    
    # Tính xác suất của mỗi điểm dữ liệu thuộc về mỗi cụm dựa trên phân phối Gaussian
    def _gaussian_pdf(self, X, mu, sigma):
        D = X.shape[1]
        sigma_reg = sigma + np.eye(D) * 1e-6
        det = np.linalg.det(sigma_reg)
        inv = np.linalg.inv(sigma_reg)
        diff = X - mu
        exponent = -0.5 * np.sum(np.dot(diff, inv) * diff, axis=1)
        coef = 1.0 / (np.power(2 * np.pi, D / 2.0) * np.sqrt(det))
        return coef * np.exp(exponent)
    
    # Hàm fit thực hiện EM để tìm tham số tối ưu của mô hình GMM
    def fit(self, X):
        n_samples, n_features = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.pi = np.ones(self.k) / self.k
        random_idxs = np.random.choice(n_samples, self.k, replace=False)
        self.mu = X[random_idxs]
        
        global_cov = np.cov(X.T)
        self.sigma = [global_cov.copy() for _ in range(self.k)]
        
        log_likelihood_old = 0
        
        for i in range(self.max_iters):
            self.gamma = np.zeros((n_samples, self.k))
            for k in range(self.k):
                self.gamma[:, k] = self.pi[k] * self._gaussian_pdf(X, self.mu[k], self.sigma[k])
                
            sum_gamma = np.sum(self.gamma, axis=1, keepdims=True)
            self.gamma = self.gamma / sum_gamma
            
            N_k = np.sum(self.gamma, axis=0)
            for k in range(self.k):
                self.mu[k] = (1 / N_k[k]) * np.sum(self.gamma[:, k, np.newaxis] * X, axis=0)
                diff = X - self.mu[k]
                self.sigma[k] = (1 / N_k[k]) * np.dot((self.gamma[:, k, np.newaxis] * diff).T, diff)
                self.pi[k] = N_k[k] / n_samples
                
            log_likelihood_new = 0
            for k in range(self.k):
                log_likelihood_new += self.pi[k] * self._gaussian_pdf(X, self.mu[k], self.sigma[k])
            log_likelihood_new = np.sum(np.log(log_likelihood_new + 1e-10))
            
            if np.abs(log_likelihood_new - log_likelihood_old) < self.tol:
                print(f"GMM hội tụ sau {i + 1} vòng lặp.")
                break
            log_likelihood_old = log_likelihood_new

    def predict(self, X):
        n_samples = X.shape[0]
        gamma_pred = np.zeros((n_samples, self.k))
        for k in range(self.k):
            gamma_pred[:, k] = self.pi[k] * self._gaussian_pdf(X, self.mu[k], self.sigma[k])
        
        return np.argmax(gamma_pred, axis=1)