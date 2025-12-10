import json
from pathlib import Path
import numpy as np
import pandas as pd

from .splits import make_splits
from .models.baseline import build_model_from_df
from .metrics import compute_metrics
from .strategies import StrategyConfig

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"

def encode_labels(y: pd.Series):
    return y.apply(lambda v: 0 if v == "NO" else 1)

def run_experiment(config: StrategyConfig):
    X_train, y_train, X_id_test, y_id_test, X_ood, y_ood = make_splits()

    y_train_enc = encode_labels(y_train)
    y_id_enc = encode_labels(y_id_test)
    y_ood_enc = encode_labels(y_ood)

    # Optional subsampling of whole train
    if config.sample_frac < 1.0:
        n = int(len(X_train) * config.sample_frac)
        idx = np.random.choice(len(X_train), size=n, replace=False)
        X_train = X_train.iloc[idx]
        y_train_enc = y_train_enc.iloc[idx]

    # NEW: undersample majority class in training
    if config.undersample_majority:
        # majority = label 0 (no readmission)
        maj_idx = y_train_enc[y_train_enc == 0].index
        min_idx = y_train_enc[y_train_enc == 1].index

        n_min = len(min_idx)
        if n_min > 0 and len(maj_idx) > n_min:
            undersampled_maj = np.random.choice(maj_idx, size=n_min, replace=False)
            keep_idx = np.concatenate([undersampled_maj, min_idx])
            X_train = X_train.loc[keep_idx]
            y_train_enc = y_train_enc.loc[keep_idx]

    model = build_model_from_df(X_train, config)
    model.fit(X_train, y_train_enc)

    id_proba = model.predict_proba(X_id_test)[:, 1]
    ood_proba = model.predict_proba(X_ood)[:, 1]

    id_metrics = compute_metrics(y_id_enc, id_proba)
    ood_metrics = compute_metrics(y_ood_enc, ood_proba)

    result = {
        "config": config.__dict__,
        "id": id_metrics,
        "ood": ood_metrics,
    }

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    out_path = EXPERIMENTS_DIR / f"run_{config.name}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print("Saved", out_path)
    print(result)
    return result
