import numpy as np

def accuracy_score(y_true, y_pred):
    # Trả về tỉ lệ đoán đúng (ví dụ: 0.95 nghĩa là đúng 95%)
    return np.mean(y_true == y_pred)