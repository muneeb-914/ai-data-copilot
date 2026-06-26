from utils.groq_client import ask_groq
from utils.analyzer import load_data, get_basic_stats, get_numeric_summary, detect_anomalies

# Test with a sample CSV
import pandas as pd

# Create a quick sample dataset
df = pd.DataFrame({
    "sales": [100, 200, 150, 9999, 130, 170],
    "region": ["North", "South", "East", "West", "North", "South"],
    "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
})
df.to_csv("data/sample.csv", index=False)

# Run analysis
df = load_data("data/sample.csv")
stats = get_basic_stats(df)
summary = get_numeric_summary(df)
anomalies = detect_anomalies(df)

# Ask Groq to interpret
prompt = f"""
Dataset overview:
- Rows: {stats['rows']}, Columns: {stats['columns']}
- Columns: {stats['column_names']}

Numeric summary:
{summary}

Anomalies found: {anomalies}

Give a short business interpretation of this data in 3-4 sentences.
"""

print("=== ANALYSIS ===")
print(f"Stats: {stats}")
print(f"\nAnomalies: {anomalies}")
print("\n=== AI INTERPRETATION ===")
print(ask_groq(prompt))