import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

from .data_loading import load_diabetes_readmission

def make_splits(test_size: float = 0.2, random_state: int = 42
               ) -> Tuple[pd.DataFrame, pd.Series,
                          pd.DataFrame, pd.Series,
                          pd.DataFrame, pd.Series]:
    """
    Train on non-ER admission sources, test OOD on ER only.
    Returns:
      X_train, y_train,
      X_id_test, y_id_test,
      X_ood_test, y_ood_test
    """
    df, label_col, domain_col = load_diabetes_readmission()

    # ER source id from UCI docs is 1 (adjust if needed)
    er_source_id = 1

    # OOD = ER only
    ood_df = df[df[domain_col] == er_source_id].copy()
    id_df = df[df[domain_col] != er_source_id].copy()

    X_id = id_df.drop(columns=[label_col])
    y_id = id_df[label_col]

    X_ood = ood_df.drop(columns=[label_col])
    y_ood = ood_df[label_col]

    X_train, X_id_test, y_train, y_id_test = train_test_split(
        X_id, y_id, test_size=test_size, random_state=random_state, stratify=y_id
    )

    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood

if __name__ == "__main__":
    X_train, y_train, X_id_test, y_id_test, X_ood, y_ood = make_splits()
    print("Train shape:", X_train.shape)
    print("ID test shape:", X_id_test.shape)
    print("OOD test shape:", X_ood.shape)
