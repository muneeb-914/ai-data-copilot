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

def run_cleaning_agent(
    profile: dict,
    use_rag: bool = False,        # NEW: whether to retrieve company cleaning policies
) -> dict:

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

    # ── RAG: Retrieve company cleaning policies if documents exist ────────────
    rag_context = ""
    if use_rag:
        try:
            from rag.pipeline import query_rag_pipeline, get_pipeline_status
            status = get_pipeline_status()
            if status["ready"]:
                chunks = query_rag_pipeline(
                    "data cleaning policy missing values duplicates columns",
                    top_k=3
                )
                if chunks:
                    parts = []
                    for i, chunk in enumerate(chunks):
                        parts.append(
                            f"[Knowledge Source {i + 1}: {chunk['filename']}]\n{chunk['text']}"
                        )
                    rag_context = "\n\n---\n\n".join(parts)
                    print(f"[CleaningAgent] RAG retrieved {len(chunks)} chunk(s) for context.")
            else:
                print("[CleaningAgent] RAG pipeline not ready — skipping retrieval.")
        except Exception as e:
            print(f"[CleaningAgent] RAG retrieval failed — skipping. Error: {e}")

    system_prompt = """You are a data cleaning expert.
You receive a dataset profile and must return a JSON cleaning plan.
For each column with missing values, decide:
- "fill_median": numeric columns with low missing % (under 30%)
- "fill_mode": categorical columns with low missing % (under 30%)
- "drop_column": any column with missing % above 30%
- "no_action": columns with no missing values

Also decide on duplicates:
- "remove_duplicates": true or false

If company cleaning policies are provided in the context below, follow them.
They override the default rules above where they conflict.

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
{input_json}"""

    # ── Append RAG context if available ──────────────────────────────────────
    if rag_context:
        prompt += f"""

Company Cleaning Policies (follow these where they apply):
{rag_context}"""

    prompt += """

Return the cleaning plan as JSON."""

    response = ask_groq(prompt, system_prompt=system_prompt)

    try:
        clean_response = response.strip().strip("```json").strip("```").strip()
        cleaning_plan = json.loads(clean_response)
    except json.JSONDecodeError:
        cleaning_plan = {"error": "Could not parse cleaning plan", "raw": response}

    return cleaning_plan