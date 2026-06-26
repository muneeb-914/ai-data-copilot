import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def get_basic_stats(df: pd.DataFrame) -> dict:
    stats = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    return stats

def get_numeric_summary(df: pd.DataFrame) -> str:
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        return "No numeric columns found."
    return numeric_df.describe().to_string()

def detect_anomalies(df: pd.DataFrame) -> list:
    anomalies = []
    numeric_df = df.select_dtypes(include='number')
    for col in numeric_df.columns:
        mean = numeric_df[col].mean()
        std = numeric_df[col].std()
        outliers = numeric_df[(numeric_df[col] < mean - 3*std) | (numeric_df[col] > mean + 3*std)]
        if not outliers.empty:
            anomalies.append(f"{col}: {len(outliers)} outliers detected")
    return anomalies if anomalies else ["No anomalies detected"]