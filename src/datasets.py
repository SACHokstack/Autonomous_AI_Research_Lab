from dataclasses import dataclass
from typing import Callable
import pandas as pd

from . import grouping  # diabetes grouping
from . import splits    # diabetes splits


@dataclass
class DatasetSpec:
    name: str
    make_splits: Callable  # returns X_train, y_train, X_id, y_id, X_ood, y_ood
    compute_group_id: Callable[[pd.DataFrame], pd.Series]


DATASETS = {
    "diabetes": DatasetSpec(
        name="diabetes",
        make_splits=splits.make_splits,
        compute_group_id=grouping.compute_group_id,
    ),
    # later: add "loan_default", "mortality", etc.
}

CURRENT_DATASET = "diabetes"

def get_dataset() -> DatasetSpec:
    return DATASETS[CURRENT_DATASET]
