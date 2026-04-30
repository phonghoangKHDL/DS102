import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_wine_data
from src.metrics import compute_metrics, print_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def main():
    data_path = "Lab 4/data/winequality-white.csv"
    # data_path = "Lab 4/data/winequality-red.csv" 
    X_train, X_test, y_train, y_test = load_wine_data(data_path)

    if X_train is None:
        return

    print("\nHuấn luyệnDecision Tree (Sklearn): ")
    dt_model = DecisionTreeClassifier(
        criterion='entropy', 
        max_depth=10, 
        min_samples_split=5,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    
    dt_results = compute_metrics(y_test, dt_preds)
    print_report(dt_results, "Sklearn - Decision Tree")


    print("\nHuấn luyện Random Forest (Sklearn): ")
    rf_model = RandomForestClassifier(
        n_estimators=15,      
        criterion='entropy',
        max_depth=10,
        min_samples_split=5,
        max_features='sqrt', 
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    rf_results = compute_metrics(y_test, rf_preds)
    print_report(rf_results, "Sklearn - Random Forest")

if __name__ == "__main__":
    main()