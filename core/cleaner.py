import pandas as pd

def execute_cleaning_plan(df: pd.DataFrame, cleaning_plan: dict) -> tuple[pd.DataFrame, list]:
    report = []

    # Handle duplicates
    if cleaning_plan.get("duplicates", {}).get("action") == "remove_duplicates":
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        report.append(f"✅ Removed {removed} duplicate rows")

    # Handle columns
    columns_plan = cleaning_plan.get("columns", {})
    for col, plan in columns_plan.items():
        if col not in df.columns:
            continue
        action = plan.get("action")
        reason = plan.get("reason", "")

        if action == "fill_median":
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            report.append(f"✅ '{col}': filled missing with median ({median_val:.2f}) — {reason}")

        elif action == "fill_mode":
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            report.append(f"✅ '{col}': filled missing with mode ('{mode_val}') — {reason}")

        elif action == "drop_column":
            df.drop(columns=[col], inplace=True)
            report.append(f"🗑️ '{col}': dropped — {reason}")

        elif action == "no_action":
            report.append(f"⏭️ '{col}': no action needed — {reason}")

    return df, report