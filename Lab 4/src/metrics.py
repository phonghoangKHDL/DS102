import numpy as np

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1)) # true postives
    fp = np.sum((y_true == 0) & (y_pred == 1)) # false positives
    fn = np.sum((y_true == 1) & (y_pred == 0)) # false negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

def print_report(metrics_dict, model_name="Model"):
    print(f"\nEvaluation Report of {model_name}: ")
    print(f"Precision: {metrics_dict['precision']:.4f}")
    print(f"Recall:    {metrics_dict['recall']:.4f}")
    print(f"F1-Score:  {metrics_dict['f1_score']:.4f}")