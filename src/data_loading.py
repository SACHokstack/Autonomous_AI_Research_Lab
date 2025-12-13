import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "diabetes_readmission.csv"

def load_diabetes_readmission():
    """
    Load the Diabetes Readmission dataset from a local CSV.
    Returns: df, label_col, domain_col
    """
    df = pd.read_csv(DATA_PATH)

    # UCI diabetes readmission dataset columns
    label_col = "readmitted"        # adjust if your CSV uses a different name
    domain_col = "admission_source_id"  # or "admission_source" depending on file

    # Add derived columns for grouping
    df["sex"] = df["gender"]
    df["er_flag"] = (df["number_emergency"] > 0).astype(int)  # Had any ER visit in prior year

    return df, label_col, domain_col

if __name__ == "__main__":
    df, label, domain = load_diabetes_readmission()
    print(df.head())
    print("Label column:", label)
    print("Domain column:", domain)
