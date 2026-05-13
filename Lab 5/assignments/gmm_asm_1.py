import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_gen import generate_kmeans_asm3
from src.gmm import GaussianMixtureModel
from src.utils import plot_clusters

def main():
    X, _ = generate_kmeans_asm3()

    print("Đang huấn luyện GMM bằng EM...")
    gmm = GaussianMixtureModel(k=3, max_iters=100, random_state=42)
    gmm.fit(X)

    labels_pred = gmm.predict(X)
    plot_clusters(X, labels_pred, gmm.mu, title="Assignment 1: GMM Training (EM)")

if __name__ == "__main__":
    main()