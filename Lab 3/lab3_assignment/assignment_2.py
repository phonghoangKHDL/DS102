import sys
import os
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import load_data
from src.metrics import evaluate_svm

def main():
    print("Đang load dữ liệu")
    
    train_path = "Lab 3/data/train"
    test_path = "Lab 3/data/test"
    
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)

    indices = np.random.permutation(len(X_train))
    X_train, y_train = X_train[indices], y_train[indices]

    print("Đang huấn luyện mô hình LinearSVC")
    model = LinearSVC(C=0.5, max_iter=2000, random_state=42)
    model.fit(X_train, y_train)

    print("Đang dự đoán trên tập Test")
    y_pred = model.predict(X_test)
    results = evaluate_svm(y_test, y_pred)

    print("KẾT QUẢ ASSIGNMENT 2: ")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()