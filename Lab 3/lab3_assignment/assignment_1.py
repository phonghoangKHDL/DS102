import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import load_data
from src.models import SoftMarginSVM
from src.metrics import evaluate_svm

print("Đang load dữ liệu...")
X_train, y_train = load_data('Lab 3/data/train')
X_test, y_test = load_data('Lab 3/data/test')

indices = np.random.permutation(len(X_train))
X_train, y_train = X_train[indices], y_train[indices]

print("Đang huấn luyện SVM")
model = SoftMarginSVM(C=1, lr=0.01, epochs=10) 
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
results = evaluate_svm(y_test, y_pred)

print("KẾT QUẢ ASSIGNMENT 1: ")
for k, v in results.items():
    print(f"{k}: {v}")
