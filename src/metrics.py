from typing import Dict
from sklearn.metrics import accuracy_score, roc_auc_score

def compute_metrics(y_true, y_pred_proba) -> Dict[str, float]:
    """
    y_pred_proba: predicted probability of positive class.
    """
    # assume positive class is ">30" or "YES" etc. We’ll threshold at 0.5
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # If labels are strings in the raw data, we’ll handle that in training script.
    metrics = {}
    try:
        metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics["auc"] = float("nan")
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    return metrics
