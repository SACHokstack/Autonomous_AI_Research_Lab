import pandas as pd
import numpy as np
from typing import Dict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

ANALYZER_LLM = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)

def detect_binary_target(df: pd.DataFrame) -> str:
    """Find binary target column (0/1, Yes/No, True/False, etc.)."""
    candidates = []

    print(f"🔍 Analyzing {len(df.columns)} columns for binary targets:")

    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        unique_count = len(unique_vals)

        print(f"  📊 {col}: {unique_count} unique values, dtype={df[col].dtype}")

        # Check if it's binary (exactly 2 unique values)
        if unique_count == 2:
            # Convert to numeric if possible
            try:
                # Handle various binary formats
                val_map = {}
                numeric_vals = []

                # Try to map common binary values
                for val in unique_vals:
                    val_str = str(val).lower()
                    if val_str in ['1', '1.0', 'true', 'yes', 'positive', 'high', 'success']:
                        val_map[val] = 1
                        numeric_vals.append(1)
                    elif val_str in ['0', '0.0', 'false', 'no', 'negative', 'low', 'failure']:
                        val_map[val] = 0
                        numeric_vals.append(0)
                    elif pd.api.types.is_numeric_dtype(type(val)) and val in [0, 1]:
                        val_map[val] = int(val)
                        numeric_vals.append(int(val))

                # If we successfully mapped both values to 0/1
                if len(val_map) == 2 and set(numeric_vals) == {0, 1}:
                    # Convert column to binary numeric
                    binary_series = df[col].map(val_map)
                    pos_ratio = binary_series.mean()

                    # More flexible ratio check - accept any binary target
                    if 0.01 <= pos_ratio <= 0.99:  # Very permissive
                        candidates.append((col, pos_ratio, unique_vals))
                        print(f"    ✅ Binary target candidate: {col} (ratio: {pos_ratio:.3f})")
                    else:
                        print(f"    ⚠️  Too extreme ratio: {pos_ratio:.3f}")
                else:
                    print(f"    ❌ Couldn't map to binary: {unique_vals}")

            except Exception as e:
                print(f"    ❌ Error processing {col}: {e}")
                continue
        else:
            print(f"    ⏭️  Not binary ({unique_count} unique values)")

    if candidates:
        # Sort by how close to balanced (0.5) the ratio is
        best_col = sorted(candidates, key=lambda x: abs(x[1] - 0.5))[0]
        print(f"🎯 Selected target: {best_col[0]} (values: {best_col[2]}, ratio: {best_col[1]:.3f})")
        return best_col[0]

    # Fallback: try to find any column that looks like a target
    print("🔄 No clear binary targets found. Checking for target-like column names...")
    target_keywords = ['target', 'label', 'outcome', 'result', 'class', 'prediction', 'y']
    for col in df.columns:
        if any(kw in col.lower() for kw in target_keywords):
            print(f"🎯 Found target-like column name: {col}")
            return col

    # Final fallback: suggest the last column (common ML convention)
    if len(df.columns) > 0:
        last_col = df.columns[-1]
        print(f"📍 Fallback: using last column as target: {last_col}")
        return last_col

    raise ValueError(f"No binary target found. Available columns: {list(df.columns)}")

def detect_protected_attrs(df: pd.DataFrame, target_col: str = None) -> List[str]:
    """Detect sex/gender, race/ethnicity, age columns."""
    keywords = {
        'sex': ['sex', 'gender', 'male', 'female', 'woman', 'man'],
        'race': ['race', 'ethnic', 'black', 'white', 'asian', 'hispanic', 'african', 'caucasian'],
        'age': ['age', 'year', 'dob', 'birth', 'old', 'young']
    }

    candidates = []
    print(f"🔍 Searching for protected attributes in {len(df.columns)} columns:")

    for col in df.columns:
        if target_col and col == target_col:
            print(f"  ⏭️  {col}: skipping target column")
            continue
        col_lower = col.lower()
        for attr, kws in keywords.items():
            if any(kw in col_lower for kw in kws):
                candidates.append(col)
                print(f"  ✅ Found {attr} attribute: {col}")
                break
        else:
            print(f"  ⏭️  {col}: no protected keywords found")

    if len(candidates) < 2:
        print(f"⚠️  Only found {len(candidates)} protected attributes. Adding fallbacks...")
        # Add some fallback candidates - columns that might be categorical
        for col in df.columns:
            if target_col and col == target_col:
                continue
            if col not in candidates and len(candidates) < 2:
                if df[col].dtype == 'object' or df[col].nunique() < 10:
                    candidates.append(col)
                    print(f"  📝 Added fallback: {col} (categorical with {df[col].nunique()} values)")

    result = candidates[:2]  # Top 2 protected attributes
    print(f"🎯 Selected protected attributes: {result}")
    return result

def detect_split_proxy(df: pd.DataFrame, target_col: str = None) -> str:
    """Find date/time column or numeric proxy for ID/OOD split."""
    print(f"🔍 Looking for time-based split proxy in {len(df.columns)} columns:")

    # Look for date/time columns
    date_keywords = ['date', 'time', 'year', 'month', 'day', 'created', 'modified', 'timestamp']
    date_cols = []

    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in date_keywords):
            date_cols.append(col)
            print(f"  ✅ Found date-like column: {col}")

    if date_cols:
        print(f"🎯 Selected date proxy: {date_cols[0]}")
        return date_cols[0]

    # Fallback: any numeric column
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_col and target_col in num_cols:
        num_cols.remove(target_col)  # Don't use target as split proxy

    if len(num_cols) > 0:
        print(f"📊 Using numeric fallback: {num_cols[0]}")
        return num_cols[0]

    print("⚠️  No suitable split proxy found - will use random split")
    return None

def analyze_dataset(df_path: str) -> Dict:
    """Full automated analysis."""
    df = pd.read_csv(df_path)
    print(f"📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"📋 Columns: {list(df.columns)}")

    # Detect components with detailed logging
    print("\n" + "="*50)
    target_col = detect_binary_target(df)

    print("\n" + "="*50)
    protected_attrs = detect_protected_attrs(df, target_col)

    print("\n" + "="*50)
    split_proxy = detect_split_proxy(df, target_col)

    analysis = {
        "target_col": target_col,
        "protected_attrs": protected_attrs,
        "split_proxy": split_proxy,
        "n_rows": df.shape[0],
        "n_cols": df.shape[1]
    }

    print(f"\n🎯 Final Analysis Summary:")
    print(f"  Target: {target_col}")
    print(f"  Protected: {protected_attrs}")
    print(f"  Split Proxy: {split_proxy}")
    print(f"  Shape: {df.shape}")
    print("="*50)
    
    # LLM confirmation
    prompt = f"""
Analyze this dataset summary and confirm analysis:
Shape: {analysis['n_rows']} rows, {analysis['n_cols']} columns
Target: {analysis['target_col']}
Protected: {analysis['protected_attrs']}
Split Proxy: {analysis['split_proxy']}
Sample columns: {list(df.columns[:10])}

Is this reasonable? Suggest improvements."""
    
    response = ANALYZER_LLM.invoke(prompt).content
    analysis["llm_notes"] = response
    
    print("Auto-detected:", analysis)
    return analysis
