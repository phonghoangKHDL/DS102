import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_wine_data
from src.models import DecisionTree
from src.metrics import compute_metrics, print_report

def main():
    # data_path = "Lab 4/data/winequality-red.csv" 
    data_path = "Lab 4/data/winequality-white.csv"
    X_train, X_test, y_train, y_test = load_wine_data(data_path)

    print(f"Dữ liệu huấn luyện: {X_train.shape}")
    print(f"Dữ liệu kiểm thử: {X_test.shape}")

    print("\nHuấn luyện Decision Tree: ")
    model = DecisionTree(max_depth=10, min_samples_split=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results = compute_metrics(y_test, y_pred)
    print_report(results, "Decision Tree")

if __name__ == "__main__":
    main()