import json
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.groq_client import ask_groq

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def run_insight_agent(profile: dict, stats_summary: str, profile_analysis: str = "", cleaning_report: list = []) -> str:
    insight_input = {
        "rows": profile["rows"],
        "columns": profile["columns"],
        "target_column": profile["guessed_target"],
        "numeric_cols": profile["numeric_cols"],
        "categorical_cols": profile["categorical_cols"],
        "high_missing_cols": profile["high_missing_cols"],
        "duplicates": profile["duplicates"],
        "column_stats": {}
    }

    for col, info in profile["column_details"].items():
        if info["type"] == "numeric":
            insight_input["column_stats"][col] = {
                "mean": info.get("mean"),
                "median": info.get("median"),
                "std": info.get("std"),
                "min": info.get("min"),
                "max": info.get("max")
            }
        elif info["type"] == "categorical":
            insight_input["column_stats"][col] = {
                "top_values": info.get("top_values")
            }

    input_json = json.dumps(insight_input, indent=2, cls=NumpyEncoder)

    system_prompt = """You are a senior business analyst presenting findings to a non-technical executive.

Your job is NOT to describe statistics.
Your job is to tell the business story hidden in the data.

Structure your response in exactly 3 sections:

## What the Data Shows
2-3 sentences. What is actually happening in this dataset in plain business language.

## Key Problems Found
3-5 bullet points. Specific issues, anomalies, or patterns that need attention.
Each point must reference actual column names and numbers from the profile.
No generic observations.

## Recommendations
3-5 bullet points. Specific actions the business should take.
Each recommendation must directly address a problem you found.
Start each with an action verb: Investigate, Monitor, Flag, Prioritize, Review.

Be direct. Be specific. No filler sentences."""

    prompt = f"""Here is the dataset profile and statistics:

{input_json}

Additional stats:
{stats_summary}

Senior analyst's initial findings:
{profile_analysis}

Cleaning actions taken:
{json.dumps(cleaning_report)}

Give your business insight analysis building on the analyst's findings above."""

    return ask_groq(prompt, system_prompt=system_prompt)