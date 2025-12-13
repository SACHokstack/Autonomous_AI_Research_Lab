from dataclasses import dataclass
from typing import Callable
import pandas as pd

from . import compas_grouping
from . import compas_splits
from . import grouping  # diabetes grouping
from . import splits    # diabetes splits


@dataclass
class DatasetSpec:
    name: str
    make_splits: Callable  # returns X_train, y_train, X_id, y_id, X_ood, y_ood
    compute_group_id: Callable[[pd.DataFrame], pd.Series]

# Add to existing DATASETS dict:


# Switch dataset:
CURRENT_DATASET = "compas"


DATASETS = {
    "diabetes": DatasetSpec(
        name="diabetes",
        make_splits=splits.make_splits,
        compute_group_id=grouping.compute_group_id,
    ),
    # later: add "loan_default", "mortality", etc.
    "compas": DatasetSpec(
        name="compas",
        make_splits=compas_splits.make_splits,
        compute_group_id=compas_grouping.compute_group_id,
    )
}

CURRENT_DATASET = "compas"

def get_dataset() -> DatasetSpec:
    return DATASETS[CURRENT_DATASET]
