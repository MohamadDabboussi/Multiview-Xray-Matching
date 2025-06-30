import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
)

def evaluate(ground_truth, predictions, threshold=0.5):
    ground_truth_np = np.array(ground_truth)
    predictions_np = np.array(predictions)
    ground_truth_binary = (ground_truth_np >= threshold).astype(int)
    pred_binary = (predictions_np >= threshold).astype(int)
    precision = precision_score(ground_truth_binary, pred_binary)
    recall = recall_score(ground_truth_binary, pred_binary)
    ap_score = average_precision_score(ground_truth_binary, predictions_np)
    roc_auc = roc_auc_score(ground_truth_binary, predictions_np)
    return precision, recall, ap_score, roc_auc
