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

        # Handle different target column formats
        target_col = self.config['target_col']
        target_values = df[target_col].unique()

        # If already binary numeric (0, 1)
        if set(target_values).issubset({0, 1}):
            df[target_col] = df[target_col].astype(int)
        # If numeric but needs to be made binary
        elif all(isinstance(v, (int, float)) for v in target_values if pd.notna(v)):
            df[target_col] = (df[target_col] == 1).astype(int)
        # If string values, handle common patterns
        else:
            target_values_clean = [str(v).upper() for v in target_values if pd.notna(v)]
            if 'NO' in target_values_clean and any(x in target_values_clean for x in ['>30', '<30', 'YES']):
                # Readmission case: NO -> 0, any readmission (<30 or >30) -> 1
                df[target_col] = (~(df[target_col].str.upper() == 'NO')).astype(int)
            elif set(target_values_clean).issubset({'YES', 'NO'}):
                # YES/NO case
                df[target_col] = (df[target_col].str.upper() == 'YES').astype(int)
            elif set(target_values_clean).issubset({'TRUE', 'FALSE'}):
                # TRUE/FALSE case
                df[target_col] = (df[target_col].str.upper() == 'TRUE').astype(int)
            else:
                # Default: convert to binary by checking if equal to the last unique value
                positive_class = sorted(target_values)[-1]
                df[target_col] = (df[target_col] == positive_class).astype(int)

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
        # Convert each protected attribute to string and combine element-wise
        attr_series = []
        for attr in self.config['protected_attrs']:
            attr_series.append(df[attr].fillna('Unknown').astype(str))

        # Combine all attributes element-wise with "_" separator
        if len(attr_series) == 1:
            return attr_series[0]
        else:
            # Use pandas string concatenation for element-wise joining
            result = attr_series[0]
            for series in attr_series[1:]:
                result = result + "_" + series
            return result


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
