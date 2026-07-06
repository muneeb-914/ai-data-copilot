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

def run_visualization_agent(profile: dict, profile_analysis: str = "") -> list:
    viz_input = {
        "target_column": profile["guessed_target"],
        "numeric_cols": profile["numeric_cols"],
        "categorical_cols": profile["categorical_cols"],
        "datetime_cols": profile["datetime_cols"],
        "columns": {}
    }

    for col, info in profile["column_details"].items():
        if info["type"] == "numeric":
            viz_input["columns"][col] = {
                "type": "numeric",
                "mean": info.get("mean"),
                "std": info.get("std"),
                "unique_values": info.get("unique_values")
            }
        elif info["type"] == "categorical":
            viz_input["columns"][col] = {
                "type": "categorical",
                "unique_values": info.get("unique_values"),
                "top_values": info.get("top_values")
            }

    input_json = json.dumps(viz_input, indent=2, cls=NumpyEncoder)

    system_prompt = """You are a data visualization expert.
You receive a dataset profile and must return a JSON list of charts to render.

Each chart must have:
- "chart_type": one of: histogram, bar, boxplot, scatter, heatmap, pie
- "x": column name for x axis
- "y": column name for y axis (null if not needed)
- "color": column name to color by (null if not needed)
- "title": descriptive chart title explaining what insight it shows
- "reason": one sentence why this chart is valuable for this dataset

Rules:
- Maximum 6 charts
- Always include target column distribution if target exists
- For numeric columns with high std, include boxplot
- For categorical vs target, include bar chart
- For numeric vs numeric relationships, include scatter
- Always include correlation heatmap if 3+ numeric columns exist
- Pick charts that reveal the most business insight, not just random columns

Return ONLY a valid JSON array. No explanation. No markdown. No backticks.

Example:
[
  {
    "chart_type": "histogram",
    "x": "session_duration",
    "y": null,
    "color": null,
    "title": "Distribution of Session Duration",
    "reason": "High std suggests outliers worth investigating"
  }
]"""

    prompt = f"""Here is the dataset profile:
{input_json}

A senior analyst already reviewed this dataset and said:
{profile_analysis}

Use their analysis to prioritize which charts are most valuable.
Return the chart recommendations as a JSON array."""
    response = ask_groq(prompt, system_prompt=system_prompt)

    try:
        clean_response = response.strip().strip("```json").strip("```").strip()
        charts = json.loads(clean_response)
    except json.JSONDecodeError:
        charts = []

    return charts