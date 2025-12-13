
import pandas as pd
import sys
import os

# Add src to path to import modules
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.splits import make_splits
from src.data_loading import load_diabetes_readmission

print("Checking load_diabetes_readmission...")
df, label, domain = load_diabetes_readmission()
print("df shape:", df.shape)
print("er_flag value counts in raw df:")
print(df["er_flag"].value_counts())

print("\nChecking make_splits...")
X_train, y_train, X_id_test, y_id_test, X_ood, y_ood = make_splits()

print("X_train er_flag value counts:")
if "er_flag" in X_train.columns:
    print(X_train["er_flag"].value_counts())
else:
    print("er_flag missing in X_train")

print("X_ood er_flag value counts:")
if "er_flag" in X_ood.columns:
    print(X_ood["er_flag"].value_counts())
else:
    print("er_flag missing in X_ood")
