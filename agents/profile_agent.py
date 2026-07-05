import json
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.groq_client import ask_groq

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer)):
            return int(obj)
        if isinstance(obj, (np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def run_profile_agent(profile: dict) -> str:
    profile_json = json.dumps(profile, indent=2, cls=NumpyEncoder)
    
    system_prompt = """You are a senior data analyst. 
You receive a structured JSON profile of a dataset.
You must respond with a clear, concise analysis covering:
1. What kind of dataset this appears to be
2. Data quality issues that need attention
3. Which columns are most important
4. What type of analysis is recommended
5. Any red flags or interesting patterns

Be direct and specific. No generic advice. Respond in clean markdown."""

    prompt = f"""Here is the dataset profile:

{profile_json}

Give your analysis."""

    return ask_groq(prompt, system_prompt=system_prompt)