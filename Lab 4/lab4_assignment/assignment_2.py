import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_wine_data
from src.models import RandomForest
from src.metrics import compute_metrics, print_report

def main():
    data_path = "Lab 4/data/winequality-white.csv"
    # data_path = "Lab 4/data/winequality-red.csv" 
    X_train, X_test, y_train, y_test = load_wine_data(data_path)

    if X_train is None:
        return

    print(f"Kích thước tập huấn luyện: {X_train.shape}")
    print(f"Kích thước tập kiểm thử: {X_test.shape}")

    print("\nHuấn luyện Random Forest: ")
    
    n_features_sqrt = int(np.sqrt(X_train.shape[1]))
    model = RandomForest(
        n_trees=15,               
        max_depth=10,             
        min_samples_split=5,      
        n_features=n_features_sqrt 
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results = compute_metrics(y_test, y_pred)
    print_report(results, "Random Forest")

if __name__ == "__main__":
    main()