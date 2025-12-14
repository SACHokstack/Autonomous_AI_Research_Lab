from pathlib import Path
from typing import Dict

from . import datasets
from .datasets import DATASETS, DatasetSpec, AutoDataset

def generate_dataset_spec(df_path: str, analysis: Dict):
    """Create AutoDataset instance from analysis."""

    dataset_name = Path(df_path).stem

    config = {
        'df_path': df_path,
        'target_col': analysis['target_col'],
        'split_proxy': analysis['split_proxy'],
        'protected_attrs': analysis['protected_attrs']
    }

    auto_ds = AutoDataset(config)

    DATASETS[dataset_name] = DatasetSpec(
        name=dataset_name,
        make_splits=auto_ds.make_splits,
        compute_group_id=auto_ds.compute_group_id,
        protected_attrs=analysis['protected_attrs']
    )

    datasets.CURRENT_DATASET = dataset_name
    print("✅ Auto-registered dataset!")
