import numpy as np

def evaluate_svm(y_true, y_pred):
    # TP: Dự đoán 1 (Bệnh), thực tế 1
    tp = np.sum((y_true == 1) & (y_pred == 1))
    # TN: Dự đoán -1 (Khỏe), thực tế -1
    tn = np.sum((y_true == -1) & (y_pred == -1))
    # FP: Dự đoán 1 (Bệnh), thực tế -1 (Khỏe) 
    fp = np.sum((y_true == -1) & (y_pred == 1))
    # FN: Dự đoán -1 (Khỏe), thực tế 1 (Bệnh)
    fn = np.sum((y_true == 1) & (y_pred == -1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }