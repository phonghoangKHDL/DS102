import matplotlib.pyplot as plt

# Hàm plot_clusters để trực quan hóa kết quả phân cụm
def plot_clusters(X, labels, centroids=None, title="K-Means Clustering"):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='w', s=50)
    
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, 
                    label='Centroids', edgecolors='k', linewidths=2)
        plt.legend()
        
    plt.title(title, fontsize=14)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()