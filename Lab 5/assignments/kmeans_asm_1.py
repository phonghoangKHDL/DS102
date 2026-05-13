import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_gen import generate_kmeans_asm1
from src.kmeans import KMeans
from src.utils import plot_clusters

def main():
    print("Đang tạo Toy Dataset cho Assignment 1...")
    X, y_true = generate_kmeans_asm1()

    print("Đang chạy K-Means...")
    
    model = KMeans(k=3, max_iters=100, random_state=42)
    model.fit(X)

    plot_clusters(X, model.labels, model.centroids, title="Assignment 1: K-Means")

"""
Nhận xét về kết quả phân cụm (Figure 1):

1. Kết quả phân cụm:
   K-Means nhóm thành công 3 cụm riêng biệt, khớp với dữ liệu tạo ra (3 cụm x 200 điểm, Sigma = I — phân bố hình tròn đều).

2. Ảnh hưởng của Random Initialization:
   Với bộ dữ liệu lý tưởng và tách biệt rõ ràng như Assignment 1, khởi tạo ngẫu nhiên không gây ảnh hưởng tiêu cực đến hiệu suất tổng thể.

3. Lý do:
   K-Means dùng EM để cập nhật liên tục. Các cụm cách xa nhau giúp thuật toán dễ hội tụ về global optimum bất chấp vị trí khởi tạo.

4. Lưu ý:
   Về lý thuyết, khởi tạo quá tệ vẫn có thể dẫn đến local optima, nhưng với dữ liệu phân tách tốt, trường hợp này rất hiếm xảy ra.
"""

if __name__ == "__main__":
    main()