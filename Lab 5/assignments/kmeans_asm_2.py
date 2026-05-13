import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_gen import generate_kmeans_asm2
from src.kmeans import KMeans
from src.utils import plot_clusters

def main():
    print("Đang tạo Toy Dataset cho Assignment 2...")
    X, y_true = generate_kmeans_asm2()

    print("Đang chạy K-Means...")

    model = KMeans(k=3, max_iters=100, random_state=42)
    model.fit(X)

    # Trực quan hóa
    plot_clusters(X, model.labels, model.centroids, title="Assignment 2: K-Means")

'''
Nhận xét về kết quả phân cụm (figure 2):

1. Tính chất thuật toán:
   K-Means dùng khoảng cách Euclidean (||x_n - mu_k||^2) ở bước E-step để gán nhãn, nên ranh giới phân định giữa các cụm luôn là 
   đường trung trực nằm chính giữa hai tâm cụm, bất kể kích thước hay mật độ của chúng.

2. Ảnh hưởng của kích thước cụm lệch nhau (1200 - 200 - 1000):
   K-Means ngầm giả định các cụm có kích thước tương đương nhau. Khi kích thước lệch quá nhiều, thuật toán bộc lộ điểm yếu:
   + Không xét đến density: các điểm ở rìa cụm lớn có thể bị gán nhầm sang cụm lân cận nếu vô tình nằm gần tâm của cụm đó hơn.
   + Có xu hướng chia đều không gian, dẫn đến việc lấn ranh giới vào vùng dữ liệu của cụm lớn.

3. Kết luận:
   K-Means hoạt động kém linh hoạt với các bộ dữ liệu mất cân bằng nghiêm trọng về số lượng điểm giữa các cụm.
'''

if __name__ == "__main__":
    main()