import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_gen import generate_kmeans_asm3
from src.kmeans import KMeans
from src.utils import plot_clusters

def main():
    print("Đang tạo Toy Dataset cho Assignment 3...")
    X, y_true = generate_kmeans_asm3()

    print("Đang chạy K-Means...")
    model = KMeans(k=3, max_iters=100, random_state=42)
    model.fit(X)
    
    plot_clusters(X, model.labels, model.centroids, title="Assignment 3: K-Means")

'''
Nhận xét về kết quả phân cụm (figure 3):

1. Đặc điểm phân phối:
   Ma trận hiệp phương sai Sigma_2 = [[10, 0], [0, 1]] khiến cụm thứ 3 có phương sai theo trục x lớn gấp 10 lần trục y, 
   làm biến dạng cụm từ hình tròn thành hình elip kéo dài sang hai bên (cụm màu xanh ngọc).

2. Bản chất toán học của K-Means:
   Ở bước E-step, K-Means gán nhãn dựa trên khoảng cách Euclidean thông thường (||x_n - mu_k||^2), vốn coi mọi hướng 
   trong không gian là như nhau. Thuật toán ngầm giả định tất cả các cụm đều có dạng hình cầu.

3. Hậu quả quan sát được trên đồ thị:
   Các điểm ở phần đuôi bên phải của cụm elip (x ~ 6 đến 10) bị đẩy ra xa tâm cụm xanh ngọc. Khoảng cách Euclidean 
   từ các điểm đó đến tâm cụm vàng lại nhỏ hơn, khiến K-Means gán nhầm toàn bộ phần đuôi sang cụm vàng.

4. Kết luận:
   K-Means hoạt động rất kém với dữ liệu có phương sai không đồng nhất hoặc phân phối bất đối xứng.
'''

if __name__ == "__main__":
    main()