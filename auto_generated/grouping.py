
import pandas as pd

def compute_group_id(df: pd.DataFrame) -> pd.Series:
    
    Agency_Text = df['Agency_Text'].fillna('Unknown').astype(str)
    
    Sex_Code_Text = df['Sex_Code_Text'].fillna('Unknown').astype(str)
    
    
    return Agency_Text + "_" + Sex_Code_Text