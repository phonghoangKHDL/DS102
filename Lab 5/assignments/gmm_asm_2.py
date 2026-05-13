import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gmm import GaussianMixtureModel

def main():
    image_path = "d:/DS102/Lab 5/data/cow.jpg"
    if not os.path.exists(image_path):
        print(f"Không tìm thấy file {image_path}")
        return 
    img = mpimg.imread(image_path)
    
    if img.max() > 1:
        img = img / 255.0
        
    h, w, c = img.shape
    print(f"Kích thước ảnh gốc: {h}x{w} pixels, {c} kênh màu.")
    
    # CHUYỂN ĐỔI ẢNH THÀNH DỮ LIỆU BẢNG
    X = img.reshape(-1, 3)
    
    # HUẤN LUYỆN GMM
    np.random.seed(42)
    sample_indices = np.random.choice(X.shape[0], size=10000, replace=False)
    X_train = X[sample_indices]
    print("Đang huấn luyện GMM...")
    gmm = GaussianMixtureModel(k=2, max_iters=50, random_state=42)
    gmm.fit(X_train)
    
    # PHÂN CỤM & TÁCH NỀN
    labels = gmm.predict(X)
    counts = np.bincount(labels)
    bg_label = np.argmax(counts)
    img_filtered = X.copy()
    img_filtered[labels == bg_label] = [0, 0, 0] 
    img_filtered = img_filtered.reshape(h, w, c)
    
    # VẼ KẾT QUẢ
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Ảnh gốc")
    plt.imshow(img)
    plt.axis('off') 
    plt.subplot(1, 2, 2)
    plt.title("Ảnh tách nền")
    plt.imshow(img_filtered)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()