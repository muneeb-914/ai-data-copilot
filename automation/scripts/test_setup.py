"""
automation/scripts/test_setup.py

Milestone 1: Verifies the n8n → Python handshake and checks that all V2
agents and the RAG layer are importable from the automation layer.

Called by n8n's Execute Command node:
    python automation/scripts/test_setup.py

Outputs JSON to stdout (n8n captures this as the node's output).
Appends one JSON line per run to automation/logs/test_run.log.
Always exits 0 so n8n doesn't treat import warnings as failures.
"""

import sys
import os
import json
from datetime import datetime

# ── resolve project root ───────────────────────────────────────────────────────
# This script sits at: <root>/automation/scripts/test_setup.py
#                              ^         ^
#                              2 levels up = root
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(ROOT, "automation", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Add project root to path so V2 modules are importable
sys.path.insert(0, ROOT)

# ── load .env if present ───────────────────────────────────────────────────────
env_loaded = False
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    env_loaded = True
except ImportError:
    pass  # dotenv not installed — non-fatal for this test

# ── test 1: V2 agent imports ───────────────────────────────────────────────────
V2_AGENTS = [
    "agents.profile_agent",
    "agents.cleaning_agent",
    "agents.visualization_agent",
    "agents.insight_agent",
    "agents.chat_agent",
]

agent_status = {}
for agent in V2_AGENTS:
    try:
        __import__(agent)
        agent_status[agent] = "OK"
    except Exception as e:
        agent_status[agent] = f"FAIL — {type(e).__name__}: {e}"

# ── test 2: RAG pipeline ───────────────────────────────────────────────────────
try:
    import rag.pipeline  # noqa: F401
    rag_status = "OK"
except Exception as e:
    rag_status = f"FAIL — {type(e).__name__}: {e}"

# ── test 3: GROQ_API_KEY ───────────────────────────────────────────────────────
api_key_set = bool(os.getenv("GROQ_API_KEY"))

# ── test 4: key V2 dependencies ───────────────────────────────────────────────
deps = {
    "pandas": None,
    "groq": None,
    "faiss": None,
    "fastembed": None,
    "plotly": None,
    "streamlit": None,
}
for dep in list(deps.keys()):
    try:
        __import__(dep)
        deps[dep] = "OK"
    except ImportError as e:
        deps[dep] = f"FAIL — {e}"

# ── assemble result ────────────────────────────────────────────────────────────
agents_ok  = all(v == "OK" for v in agent_status.values())
rag_ok     = rag_status == "OK"
deps_ok    = all(v == "OK" for v in deps.values())

if agents_ok and rag_ok and api_key_set and deps_ok:
    overall = "OK"
elif agents_ok and rag_ok:
    overall = "PARTIAL"       # env/dep issues but core is fine
else:
    overall = "NEEDS_FIX"

result = {
    "timestamp":       datetime.utcnow().isoformat() + "Z",
    "test":            "n8n → Python handshake (M1)",
    "python":          sys.version.split()[0],
    "project_root":    ROOT,
    "env_loaded":      env_loaded,
    "groq_api_key_set": api_key_set,
    "agents":          agent_status,
    "rag":             rag_status,
    "dependencies":    deps,
    "status":          overall,
}

# ── append to log (JSON lines) ─────────────────────────────────────────────────
log_path = os.path.join(LOG_DIR, "test_run.log")
with open(log_path, "a") as f:
    f.write(json.dumps(result) + "\n")

# ── print to stdout — n8n's Execute Command node captures this ────────────────
print(json.dumps(result, indent=2))

# Exit 0 always — import warnings are not failures at this stage
sys.exit(0)