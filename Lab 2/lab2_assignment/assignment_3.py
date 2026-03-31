import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_mnist_full, load_mnist_ubyte
from src.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

print("--- 1. Logistic Regression (Binary: 0 vs 1) ---")
X_bin, y_bin = load_mnist_ubyte(0, 1)

log_reg = LogisticRegression(solver='lbfgs', max_iter=200)
log_reg.fit(X_bin, y_bin)

y_pred_bin = log_reg.predict(X_bin)
print(f"Accuracy (Binary 0 vs 1): {accuracy_score(y_bin, y_pred_bin)*100:.2f}%")

print("\n--- 2. Softmax Regression (Full 10 classes) ---")
X_all, y_all = load_mnist_full()

softmax_reg = LogisticRegression(solver='lbfgs', max_iter=500)
softmax_reg.fit(X_all, y_all)

y_pred_all = softmax_reg.predict(X_all)
print(f"Accuracy (Full 10 classes): {accuracy_score(y_all, y_pred_all)*100:.2f}%")

print("\nChi tiết đánh giá Softmax:")
print(classification_report(y_all, y_pred_all))