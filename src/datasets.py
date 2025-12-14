from dataclasses import dataclass
from typing import Callable
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from . import compas_grouping
from . import compas_splits
from . import grouping  # diabetes grouping
from . import splits    # diabetes splits


class AutoDataset:
    def __init__(self, config: dict):
        self.config = config

    def load_data(self):
        df = pd.read_csv(self.config['df_path'])
        df[self.config['target_col']] = (df[self.config['target_col']] == 1).astype(int)  # Ensure binary
        return df

    def make_splits(self):
        df = self.load_data()
        
        if self.config['split_proxy']:
            # Time-based split using split_proxy
            df['split_proxy'] = pd.to_datetime(df[self.config['split_proxy']])
            df = df.sort_values('split_proxy')
            split_idx = int(len(df) * 0.8)
            
            id_pool = df.iloc[:split_idx]
            ood_pool = df.iloc[split_idx:]
        else:
            # Random split
            id_pool = df
            ood_pool = df.sample(frac=0.2, random_state=42)
            id_pool = id_pool.drop(ood_pool.index)
        
        X_train, X_id_test, y_train, y_id_test = train_test_split(
            id_pool.drop(self.config['target_col'], axis=1),
            id_pool[self.config['target_col']],
            test_size=0.3, random_state=42, stratify=id_pool[self.config['target_col']]
        )
        
        X_ood = ood_pool.drop(self.config['target_col'], axis=1)
        y_ood = ood_pool[self.config['target_col']]
        
        print(f"Train: {X_train.shape[0]}, ID test: {X_id_test.shape[0]}, OOD: {X_ood.shape[0]}")
        return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood

    def compute_group_id(self, df: pd.DataFrame) -> pd.Series:
        attrs = []
        for attr in self.config['protected_attrs']:
            attrs.append(df[attr].fillna('Unknown').astype(str))
        return "_".join(attrs)


@dataclass
class DatasetSpec:
    name: str
    make_splits: Callable  # returns X_train, y_train, X_id, y_id, X_ood, y_ood
    compute_group_id: Callable[[pd.DataFrame], pd.Series]
    protected_attrs: list[str]

# Add to existing DATASETS dict:


# Switch dataset:
CURRENT_DATASET = "compas"


DATASETS = {
    "diabetes": DatasetSpec(
        name="diabetes",
        make_splits=splits.make_splits,
        compute_group_id=grouping.compute_group_id,
        protected_attrs=["sex", "number_emergency"],
    ),
    # later: add "loan_default", "mortality", etc.
    "compas": DatasetSpec(
        name="compas",
        make_splits=compas_splits.make_splits,
        compute_group_id=compas_grouping.compute_group_id,
        protected_attrs=["race", "sex"],
    )
}

CURRENT_DATASET = "compas"

def get_dataset() -> DatasetSpec:
    return DATASETS[CURRENT_DATASET]
