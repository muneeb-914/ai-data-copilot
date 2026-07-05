import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    report = []
    
    # 1. Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        for col, count in missing_cols.items():
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
                report.append(f"✅ '{col}': filled {count} missing values with median")
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
                report.append(f"✅ '{col}': filled {count} missing values with mode")
    else:
        report.append("✅ No missing values found")
    
    # 2. Duplicate rows
    dupes = df.duplicated().sum()
    if dupes > 0:
        df.drop_duplicates(inplace=True)
        report.append(f"✅ Removed {dupes} duplicate rows")
    else:
        report.append("✅ No duplicate rows found")
    
    # 3. Strip whitespace from string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    report.append("✅ Stripped whitespace from text columns")
    
    return df, report

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