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
        if numeric_df[col].nunique() <= 2:
            continue
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = numeric_df[(numeric_df[col] < lower) | (numeric_df[col] > upper)]
        if not outliers.empty:
            anomalies.append(f"{col}: {len(outliers)} outliers detected (IQR method) — range [{round(lower,2)}, {round(upper,2)}]")
    return anomalies if anomalies else ["No anomalies detected"]