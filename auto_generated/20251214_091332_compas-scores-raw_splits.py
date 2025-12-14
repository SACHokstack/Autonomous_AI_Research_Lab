
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_PATH = Path("uploads/20251214_091332_compas-scores-raw.csv")

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Agency_Text'] = (df['Agency_Text'] == 1).astype(int)  # Ensure binary
    return df

def make_splits():
    df = load_data()
    
    
    # Time-based split using DateOfBirth
    df['split_proxy'] = pd.to_datetime(df['DateOfBirth'])
    df = df.sort_values('split_proxy')
    split_idx = int(len(df) * 0.8)
    
    id_pool = df.iloc[:split_idx]
    ood_pool = df.iloc[split_idx:]
    
    
    X_train, X_id_test, y_train, y_id_test = train_test_split(
        id_pool.drop('Agency_Text', axis=1),
        id_pool['Agency_Text'],
        test_size=0.3, random_state=42, stratify=id_pool['Agency_Text']
    )
    
    X_ood = ood_pool.drop('Agency_Text', axis=1)
    y_ood = ood_pool['Agency_Text']
    
    print(f"Train: {X_train.shape[0]}, ID test: {X_id_test.shape[0]}, OOD: {X_ood.shape[0]}")
    return X_train, y_train, X_id_test, y_id_test, X_ood, y_ood