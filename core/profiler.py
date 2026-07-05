import pandas as pd
import json

def profile_dataset(df: pd.DataFrame) -> dict:
    profile = {}

    # Basic shape
    profile["rows"] = len(df)
    profile["columns"] = len(df.columns)

    # Column-level analysis
    profile["column_details"] = {}
    for col in df.columns:
        col_info = {}
        col_info["dtype"] = str(df[col].dtype)
        col_info["missing_count"] = int(df[col].isnull().sum())
        col_info["missing_pct"] = round(df[col].isnull().mean() * 100, 2)
        col_info["unique_values"] = int(df[col].nunique())

        if df[col].dtype in ["int64", "float64"]:
            col_info["type"] = "numeric"
            col_info["mean"] = round(df[col].mean(), 2)
            col_info["median"] = round(df[col].median(), 2)
            col_info["std"] = round(df[col].std(), 2)
            col_info["min"] = round(df[col].min(), 2)
            col_info["max"] = round(df[col].max(), 2)
        elif df[col].dtype == "object":
            # Check if it might be a date
            sample = df[col].dropna().head(10).tolist()
            col_info["type"] = "categorical"
            col_info["top_values"] = df[col].value_counts().head(5).to_dict()
            col_info["sample"] = sample
        elif "datetime" in str(df[col].dtype):
            col_info["type"] = "datetime"
            col_info["min_date"] = str(df[col].min())
            col_info["max_date"] = str(df[col].max())
        else:
            col_info["type"] = "other"

        profile["column_details"][col] = col_info

    # Dataset-level summary
    profile["duplicates"] = int(df.duplicated().sum())
    profile["numeric_cols"] = [c for c in df.columns if df[c].dtype in ["int64", "float64"]]
    profile["categorical_cols"] = [c for c in df.columns if df[c].dtype == "object"]
    profile["datetime_cols"] = [c for c in df.columns if "datetime" in str(df[c].dtype)]
    profile["high_missing_cols"] = [
        c for c in df.columns if df[c].isnull().mean() > 0.3
    ]

    # Guess target column (last column or column named label/target/class/attack)
    target_keywords = ["label", "target", "class", "attack", "outcome", "churn", "status"]
    guessed_target = None
    for col in df.columns:
        if col.lower() in target_keywords:
            guessed_target = col
            break
    if not guessed_target:
        guessed_target = df.columns[-1]
    profile["guessed_target"] = guessed_target

    return profile