import json
from pathlib import Path

import pandas as pd

from .splits import make_splits
from .models.baseline import build_baseline_model
from .metrics import compute_metrics

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"

def encode_labels(y: pd.Series):
    # UCI readmitted labels: "NO", "<30", ">30" – treat "NO" as 0, rest as 1
    return y.apply(lambda v: 0 if v == "NO" else 1)

def main():
    X_train, y_train, X_id_test, y_id_test, X_ood, y_ood = make_splits()

    y_train_enc = encode_labels(y_train)
    y_id_enc = encode_labels(y_id_test)
    y_ood_enc = encode_labels(y_ood)

    make_model = build_baseline_model()
    model = make_model(X_train)

    model.fit(X_train, y_train_enc)

    id_proba = model.predict_proba(X_id_test)[:, 1]
    ood_proba = model.predict_proba(X_ood)[:, 1]

    id_metrics = compute_metrics(y_id_enc, id_proba)
    ood_metrics = compute_metrics(y_ood_enc, ood_proba)

    results = {
        "id": id_metrics,
        "ood": ood_metrics,
    }

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    out_path = EXPERIMENTS_DIR / "baseline_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print("Saved baseline results to", out_path)
    print(results)

if __name__ == "__main__":
    main()
