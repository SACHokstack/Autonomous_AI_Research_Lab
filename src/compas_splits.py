import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "compas-scores-two-years.csv"

def load_compas():
    """Load YOUR COMPAS dataset with minimal preprocessing."""
    df = pd.read_csv(DATA_PATH)
    
    # Use your actual columns
    # Create label: assume 'DecileScore' > 4 = high risk (recidivism proxy)
    df['two_year_recid'] = (df['decile_score'] > 4).astype(int)
    
    # Map your columns to standard names
    df['sex'] = df['sex'].fillna('Unknown')
    df['race'] = df['race'].fillna('Unknown')
    
    # Keep key features + metadata
    cols = [
        'sex', 'race', 'age', 'decile_score', 'compas_screening_date',
        'priors_count' if 'priors_count' in df else 'Scale_ID',
        'two_year_recid'
    ]
    available_cols = [c for c in cols if c in df.columns or c in df.columns]
    return df[available_cols]

def make_splits():
    """Time-based split using Screening_Date."""
    df = load_compas()
    
    # Sort by Screening_Date, split 80/20 time-wise
    df['screening_date'] = pd.to_datetime(df['compas_screening_date'])
    df = df.sort_values('screening_date')
    split_idx = int(len(df) * 0.8)
    
    id_pool = df.iloc[:split_idx]
    ood_pool = df.iloc[split_idx:]
    
    # From ID pool: train (70%) + ID test (30%)
    X_id, _, y_id, _ = train_test_split(
        id_pool.drop('two_year_recid', axis=1, errors='ignore'),
        id_pool['two_year_recid'],
        test_size=0.3, random_state=42, stratify=id_pool['two_year_recid']
    )
    
    X_train, X_id_test, y_train, y_id_test = train_test_split(
        X_id, y_id, test_size=0.3, random_state=42, stratify=y_id
    )
    
    X_ood = ood_pool.drop('two_year_recid', axis=1, errors='ignore')
    y_ood = ood_pool['two_year_recid']
    
    print(f"Train: {len(X_train)}, ID test: {len(X_id_test)}, OOD: {len(X_ood)}")
    print("Group columns:", ['sex', 'race'] if 'sex' in X_ood else "sex/race missing")
    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood
