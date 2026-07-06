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

def run_cleaning_agent(profile: dict) -> dict:
    # Only send relevant info, not the full profile
    cleaning_input = {
        "rows": profile["rows"],
        "duplicates": profile["duplicates"],
        "high_missing_cols": profile["high_missing_cols"],
        "columns": {}
    }

    for col, info in profile["column_details"].items():
        cleaning_input["columns"][col] = {
            "type": info["type"],
            "missing_pct": info["missing_pct"],
            "unique_values": info["unique_values"]
        }

    input_json = json.dumps(cleaning_input, indent=2, cls=NumpyEncoder)

    system_prompt = """You are a data cleaning expert.
You receive a dataset profile and must return a JSON cleaning plan.
For each column with missing values, decide:
- "fill_median": numeric columns with low missing % (under 30%)
- "fill_mode": categorical columns with low missing % (under 30%)
- "drop_column": any column with missing % above 30%
- "no_action": columns with no missing values

Also decide on duplicates:
- "remove_duplicates": true or false

Return ONLY a valid JSON object. No explanation. No markdown. No backticks.
Example format:
{
  "duplicates": {"action": "remove_duplicates"},
  "columns": {
    "Age": {"action": "fill_median", "reason": "numeric, 5% missing"},
    "Salary": {"action": "drop_column", "reason": "45% missing"},
    "Gender": {"action": "fill_mode", "reason": "categorical, 3% missing"},
    "Name": {"action": "no_action", "reason": "no missing values"}
  }
}"""

    prompt = f"""Here is the dataset profile:
{input_json}

Return the cleaning plan as JSON."""

    response = ask_groq(prompt, system_prompt=system_prompt)

    # Parse the JSON response
    try:
        # Strip any accidental markdown
        clean_response = response.strip().strip("```json").strip("```").strip()
        cleaning_plan = json.loads(clean_response)
    except json.JSONDecodeError:
        # Fallback: basic plan if Groq returns unexpected format
        cleaning_plan = {"error": "Could not parse cleaning plan", "raw": response}

    return cleaning_plan