
import pandas as pd

def compute_group_id(df: pd.DataFrame) -> pd.Series:
    """COMPAS groups: use standardized column names."""
    race = df['race'].fillna('Unknown').astype(str)
    sex = df['sex'].fillna('Unknown').astype(str)
    return race + "_" + sex
