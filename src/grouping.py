import pandas as pd


def compute_group_id(df: pd.DataFrame) -> pd.Series:
    """
    Dataset-specific group definition.
    Here: Sex × ER, using gender + number_emergency.
    """
    er_flag = (df["number_emergency"] > 0).astype(int)
    sex = df["sex"].replace({"Male": "Male", "Female": "Female"})
    return sex + "_" + er_flag.map({0: "NON_ER", 1: "ER"})
