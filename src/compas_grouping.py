import pandas as pd

def compute_group_id(df: pd.DataFrame) -> pd.Series:
    """
    COMPAS groups: Race × Sex (standard fairness groups).
    E.g. "African-American_Male", "Caucasian_Female".
    """
    race = df["race"].fillna("Unknown").astype(str)
    sex = df["sex"].fillna("Unknown").astype(str)
    return race + "_" + sex
