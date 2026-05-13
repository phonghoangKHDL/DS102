import numpy as np

def generate_kmeans_asm1():
    mu1 = np.array([2, 2])
    mu2 = np.array([8, 3])
    mu3 = np.array([3, 6])
    
    cov = np.array([[1, 0], 
                    [0, 1]])
    
    X1 = np.random.multivariate_normal(mu1, cov, 200)
    X2 = np.random.multivariate_normal(mu2, cov, 200)
    X3 = np.random.multivariate_normal(mu3, cov, 200)
    
    X = np.vstack((X1, X2, X3))
    
    y_true = np.array([0]*200 + [1]*200 + [2]*200)
    
    return X, y_true

def generate_kmeans_asm2():
    mu1 = np.array([2, 2])
    mu2 = np.array([8, 3])
    mu3 = np.array([3, 6])
    
    cov = np.array([[1, 0], 
                    [0, 1]])
    
    X1 = np.random.multivariate_normal(mu1, cov, 1200)
    X2 = np.random.multivariate_normal(mu2, cov, 200)
    X3 = np.random.multivariate_normal(mu3, cov, 1000)
    
    X = np.vstack((X1, X2, X3))
    y_true = np.array([0]*1200 + [1]*200 + [2]*1000)
    
    return X, y_true

def generate_kmeans_asm3():
    mu1 = np.array([2, 2])
    mu2 = np.array([8, 3])
    mu3 = np.array([3, 6])
    
    cov1 = np.array([[1, 0], 
                     [0, 1]])
    
    cov2 = np.array([[10, 0], 
                     [0,  1]])
    
    X1 = np.random.multivariate_normal(mu1, cov1, 200)
    X2 = np.random.multivariate_normal(mu2, cov1, 200)
    X3 = np.random.multivariate_normal(mu3, cov2, 200)
    
    X = np.vstack((X1, X2, X3))
    y_true = np.array([0]*200 + [1]*200 + [2]*200)
    
    return X, y_true