import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "compas-scores-two-years.csv"

def load_compas():
    """Load and apply standard COMPAS preprocessing (ProPublica style)."""
    df = pd.read_csv(DATA_PATH)
    
    # Standard filters (days_since_decision > 30, is_recid <=1, etc.)
    df = df[
        (df.days_b_screening_arrest <= 30) &
        (df.days_b_screening_arrest >= -30) &
        (df.is_recid != -1) &
        (df.c_charge_degree != "0") &
        (df.score_text != 'N/A')
    ].copy()
    
    # Label: two_year_recid
    df['two_year_recid'] = (df.is_recid == 1).astype(int)
    
    # Keep key features + metadata
    cols = [
        'sex', 'age', 'race', 'juv_fel_count', 'decile_score',
        'juv_misd_count', 'juv_other_count', 'priors_count',
        'c_days_from_compas', 'c_charge_degree', 'c_charge_desc',
        'two_year_recid'  # label
    ]
    return df[cols]

def make_splits():
    """Time-based split: early dates = ID, late dates = OOD."""
    df = load_compas()
    
    # Sort by screening date, split 80/20 time-wise
    df = df.sort_values('c_days_from_compas')
    split_idx = int(len(df) * 0.8)
    
    id_pool = df.iloc[:split_idx]
    ood_pool = df.iloc[split_idx:]
    
    # From ID pool: train (70%) + ID test (30%)
    X_id, _, y_id, _ = train_test_split(
        id_pool.drop('two_year_recid', axis=1),
        id_pool['two_year_recid'],
        test_size=0.3, random_state=42, stratify=id_pool['two_year_recid']
    )
    
    X_train, X_id_test, y_train, y_id_test = train_test_split(
        X_id, y_id, test_size=0.3, random_state=42, stratify=y_id
    )
    
    X_ood = ood_pool.drop('two_year_recid', axis=1)
    y_ood = ood_pool['two_year_recid']
    
    print(f"Train: {len(X_train)}, ID test: {len(X_id_test)}, OOD: {len(X_ood)}")
    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood
