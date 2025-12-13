import json
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import pandas as pd

from .models.baseline import build_model_from_df
from .metrics import compute_metrics
from .strategies import StrategyConfig
from sklearn.preprocessing import OneHotEncoder
from .datasets import get_dataset


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"


def encode_labels(y: pd.Series):
    return y.apply(lambda v: 0 if v == "NO" else 1)


def run_experiment(config: StrategyConfig):
    # --------------------
    # 1. Load data & labels
    # --------------------
    ds = get_dataset()
    X_train, y_train, X_id_test, y_id_test, X_ood, y_ood = ds.make_splits()

    # Keep metadata columns for grouping before any feature dropping
    meta_id_test = X_id_test[["sex", "er_flag"]]
    meta_ood_test = X_ood[["sex", "er_flag"]]

    y_train_enc = encode_labels(y_train)
    y_id_enc = encode_labels(y_id_test)
    y_ood_enc = encode_labels(y_ood)

    # --------------------
    # 2. Optional subsampling of train
    # --------------------
    if config.sample_frac < 1.0:
        n = int(len(X_train) * config.sample_frac)
        idx = np.random.choice(len(X_train), size=n, replace=False)
        X_train = X_train.iloc[idx]
        y_train_enc = y_train_enc.iloc[idx]

    # --------------------
    # 3. Optional undersampling of majority class
    # --------------------
    if getattr(config, "undersample_majority", False):
        # majority = label 0 (no readmission)
        maj_idx = y_train_enc[y_train_enc == 0].index
        min_idx = y_train_enc[y_train_enc == 1].index

        n_min = len(min_idx)
        if n_min > 0 and len(maj_idx) > n_min:
            undersampled_maj = np.random.choice(maj_idx, size=n_min, replace=False)
            keep_idx = np.concatenate([undersampled_maj, min_idx])

            X_train = X_train.loc[keep_idx]
            y_train_enc = y_train_enc.loc[keep_idx]

    # --------------------
    # 4. Encode categorical columns
    # --------------------
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    num_cols = X_train.select_dtypes(include=["number"]).columns

    if len(cat_cols) > 0:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        # Fit encoder on TRAIN only
        X_cat_train = encoder.fit_transform(X_train[cat_cols])
        X_num_train = X_train[num_cols].to_numpy()
        X_train_encoded = np.hstack([X_num_train, X_cat_train])

        # Transform ID test and OOD with same encoder & same columns
        X_cat_id = encoder.transform(X_id_test[cat_cols])
        X_num_id = X_id_test[num_cols].to_numpy()
        X_id_encoded = np.hstack([X_num_id, X_cat_id])

        X_cat_ood = encoder.transform(X_ood[cat_cols])
        X_num_ood = X_ood[num_cols].to_numpy()
        X_ood_encoded = np.hstack([X_num_ood, X_cat_ood])
    else:
        # No categoricals, just use numeric columns
        X_train_encoded = X_train[num_cols].to_numpy()
        X_id_encoded = X_id_test[num_cols].to_numpy()
        X_ood_encoded = X_ood[num_cols].to_numpy()

        # --------------------
    # 5. Build & train model on encoded data
    # --------------------
    model = build_model_from_df(pd.DataFrame(X_train_encoded), config)

    # Group-aware sample_weight for group_dro strategies
    sample_weight = None
    if getattr(config, "use_group_dro", False):
        # Use raw X_train columns to compute group ids
        group_ids = ds.compute_group_id(X_train)
        group_ids = np.array(group_ids)
        unique_groups, counts = np.unique(group_ids, return_counts=True)
        freq = dict(zip(unique_groups, counts))
        sample_weight = np.array([1.0 / freq[g] for g in group_ids], dtype=float)

    if sample_weight is not None:
        model.fit(X_train_encoded, y_train_enc, sample_weight=sample_weight)
    else:
        model.fit(X_train_encoded, y_train_enc)


    # --------------------
    # 6. Predict & compute metrics
    # --------------------
    id_proba = model.predict_proba(X_id_encoded)[:, 1]
    ood_proba = model.predict_proba(X_ood_encoded)[:, 1]

    id_metrics = compute_metrics(y_id_enc, id_proba)
    ood_metrics = compute_metrics(y_ood_enc, ood_proba)

    # Compute Sex × ER group accuracies for OOD
    # Compute group accuracies for OOD using the same group function
    y_pred_ood = (ood_proba > 0.5).astype(int)

    groups_ood = ds.compute_group_id(X_ood)
    groups_ood = list(groups_ood)

    correct = defaultdict(int)
    total = defaultdict(int)

    for g, y_true, y_hat in zip(groups_ood, y_ood_enc, y_pred_ood):
        total[g] += 1
        if y_true == y_hat:
            correct[g] += 1

    group_acc = {g: correct[g] / total[g] for g in total}
    # Filter out Unknown/Invalid groups for WGA calculation
    valid_groups = {g: acc for g, acc in group_acc.items()
                    if "Unknown" not in g and "Invalid" not in g}
    worst_group_acc = min(valid_groups.values()) if valid_groups else None


    # --------------------
    # 7. Save results
    # --------------------
    result = {
        "config": asdict(config),
        "id": {
            "auc": id_metrics["auc"],
            "accuracy": id_metrics["accuracy"],
        },
        "ood": {
            "auc": ood_metrics["auc"],
            "accuracy": ood_metrics["accuracy"],
            "group_accuracy": group_acc,
            "worst_group_accuracy": worst_group_acc,
        },
        "meta_id": meta_id_test.to_dict("records"),
        "meta_ood": meta_ood_test.to_dict("records"),
    }

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    out_path = EXPERIMENTS_DIR / f"run_{config.name}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print("Saved", out_path)
    print(result)
    return result
