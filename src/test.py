import pandas as pd

df = pd.read_csv("/home/sach/Prometheus/data/diabetes_readmission.csv")

# Clean up categorical columns
df["gender"] = df["gender"].replace("Unknown/Invalid", "Unknown")
df["race"] = df["race"].fillna("Unknown")
df["er_flag"] = (df["number_emergency"] > 0).astype(int)


# Build the group feature: gender × race
df["group"] = df["gender"] + "_" + df["er_flag"].map({0:"NON_ER", 1:"ER"})


# Count group sizes
counts = df.groupby("group").size().reset_index(name="n")

print(counts)
