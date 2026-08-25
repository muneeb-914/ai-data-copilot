"""
automation/scripts/api.py

V3 — FastAPI wrapper around the V2 AI Data Operations Copilot pipeline.
n8n calls these endpoints via HTTP Request node instead of spawning Python directly.

Start server (run from project root):
    uvicorn automation.scripts.api:app --reload --port 8000

Endpoints:
    GET  /health           → quick health check
    GET  /rag-status       → check RAG index status + list uploaded docs
    POST /upload-docs      → upload PDF/DOCX/TXT/MD → saves to knowledge/ → rebuilds RAG index
    POST /analyze          → upload CSV, runs full V2 pipeline (RAG auto-enabled if docs exist)
    POST /analyze-by-path  → pass local CSV path in JSON body (easier for n8n folder watcher)
"""

import sys
import os
import json
import time
import uuid
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

# ── add project root to sys.path ──────────────────────────────────────────────
# This file lives at: <root>/automation/scripts/api.py
#                              2 levels up = project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── load .env first (Groq key must exist before any agent import) ─────────────
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# ── stdlib / third-party ──────────────────────────────────────────────────────
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# ── output dirs ───────────────────────────────────────────────────────────────
REPORTS_DIR   = ROOT / "automation" / "reports"
LOGS_DIR      = ROOT / "automation" / "logs"
KNOWLEDGE_DIR = ROOT / "knowledge"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_DOC_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Data Operations Copilot — V3 API",
    description="n8n orchestrates the V2 multi-agent pipeline through these endpoints.",
    version="3.0.0",
)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# Calls V2 agents in sequence. Each step is isolated in try/except so one
# failure doesn't kill the whole run — it logs FAIL and continues.
#
# ⚠  ADAPT FUNCTION NAMES to match what your V2 agents actually export.
#    Check each agent file and update the import + function call accordingly.
# ══════════════════════════════════════════════════════════════════════════════

def _safe_run(step_name: str, fn, *args, **kwargs):
    """Runs one pipeline step. Returns (result, None) or (None, error_string)."""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _rag_ready() -> bool:
    """Returns True if the RAG index is built and ready to use."""
    try:
        from rag.pipeline import get_pipeline_status
        return get_pipeline_status().get("ready", False)
    except Exception:
        return False


def run_v2_pipeline(df: pd.DataFrame, filename: str) -> dict:
    steps   = {}
    t_start = time.time()

    # ── 0. RAG auto-detect ────────────────────────────────────────────────────
    # If documents exist in knowledge/ and index is built, enable RAG for all agents.
    # No manual flag needed — pipeline checks automatically every run.
    use_rag = _rag_ready()

    # ── 1. Profiler ───────────────────────────────────────────────────────────
    # Returns profile dict with rows, columns, column_details, numeric_cols, etc.
    # ⚠ VERIFY: open core/profiler.py and confirm the function is named profile_dataset
    from core.profiler import profile_dataset
    profile, err = _safe_run("profiler", profile_dataset, df)
    steps["profiler"] = {"status": "OK" if not err else "FAIL", "error": err}

    # ── 2. Profile Agent ──────────────────────────────────────────────────────
    # Signature: run_profile_agent(profile: dict) -> str
    # Returns: markdown text analysis of the dataset
    profile_analysis = ""
    if profile:
        from agents.profile_agent import run_profile_agent
        profile_analysis, err = _safe_run("profile_agent", run_profile_agent, profile)
        profile_analysis = profile_analysis or ""
        steps["profile_agent"] = {"status": "OK" if not err else "FAIL", "error": err}
    else:
        steps["profile_agent"] = {"status": "SKIP", "error": "profiler failed"}

    # ── 3. Cleaning Agent ─────────────────────────────────────────────────────
    # Signature: run_cleaning_agent(profile: dict, use_rag: bool = False) -> dict
    # Takes ONLY profile — NOT df. Returns per-column cleaning plan dict.
    cleaning_plan = None
    if profile:
        from agents.cleaning_agent import run_cleaning_agent
        cleaning_plan, err = _safe_run(
            "cleaning_agent", run_cleaning_agent, profile, use_rag
        )
        steps["cleaning_agent"] = {"status": "OK" if not err else "FAIL", "error": err}
    else:
        steps["cleaning_agent"] = {"status": "SKIP", "error": "profiler failed"}

    # ── 4. Cleaner — Python executes the cleaning plan ────────────────────────
    # Confirmed: execute_cleaning_plan(df, cleaning_plan) -> tuple[pd.DataFrame, list]
    df_clean = df
    cleaning_report = []
    if cleaning_plan and "error" not in cleaning_plan:
        from core.cleaner import execute_cleaning_plan
        result, err = _safe_run("cleaner", execute_cleaning_plan, df, cleaning_plan)
        if not err and result is not None:
            df_clean, cleaning_report = result[0], list(result[1])
        steps["cleaner"] = {"status": "OK" if not err else "FAIL", "error": err}
    else:
        steps["cleaner"] = {"status": "SKIP"}

    # ── 5. Analyzer — anomaly detection + stats summary ───────────────────────
    # Confirmed:
    #   detect_anomalies(df)    → list of strings  (anomalies for insight_agent)
    #   get_numeric_summary(df) → str              (stats_summary for insight_agent)
    stats_summary = ""
    anomalies     = []
    from core.analyzer import detect_anomalies, get_numeric_summary
    anomalies, err = _safe_run("analyzer", detect_anomalies, df_clean)
    anomalies = anomalies or []
    steps["analyzer"] = {"status": "OK" if not err else "FAIL", "error": err}
    stats_summary, err2 = _safe_run("numeric_summary", get_numeric_summary, df_clean)
    stats_summary = stats_summary or ""

    # ── 6. Visualization Agent ────────────────────────────────────────────────
    # Signature: run_visualization_agent(profile: dict, profile_analysis: str = "") -> list
    # Returns: list of chart spec dicts
    chart_specs = []
    if profile and profile_analysis:
        from agents.visualization_agent import run_visualization_agent
        chart_specs, err = _safe_run(
            "visualization_agent", run_visualization_agent, profile, profile_analysis
        )
        chart_specs = chart_specs or []
        steps["visualization_agent"] = {"status": "OK" if not err else "FAIL", "error": err}
    else:
        steps["visualization_agent"] = {"status": "SKIP"}

    # ── 7. Insight Agent ──────────────────────────────────────────────────────
    # Signature: run_insight_agent(profile, stats_summary, profile_analysis, cleaning_report, use_rag) -> str
    # Returns: 3-section business report as markdown string
    insights = None
    if profile and profile_analysis:
        from agents.insight_agent import run_insight_agent
        insights, err = _safe_run(
            "insight_agent", run_insight_agent,
            profile, stats_summary, profile_analysis, cleaning_report, use_rag
        )
        steps["insight_agent"] = {"status": "OK" if not err else "FAIL", "error": err}
    else:
        steps["insight_agent"] = {"status": "SKIP"}

    # ── assemble final report ─────────────────────────────────────────────────
    all_ok = all(v["status"] in ("OK", "SKIP") for v in steps.values())

    # Use V2's own exporter to build the markdown report (same format as Streamlit)
    markdown_report = None
    if profile and insights:
        from core.exporter import export_insight_report
        markdown_report, _ = _safe_run(
            "exporter", export_insight_report,
            profile, profile_analysis, cleaning_report, anomalies, insights, chart_specs
        )

    return {
        "run_id":           str(uuid.uuid4())[:8],
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "filename":         filename,
        "rows":             int(df.shape[0]),
        "columns":          int(df.shape[1]),
        "duration_seconds": round(time.time() - t_start, 2),
        "rag_used":         use_rag,
        "steps":            steps,
        "insights":         insights,
        "chart_specs":      chart_specs,
        "anomaly_summary":  anomalies,
        "markdown_report":  markdown_report,
        "overall_status":   "OK" if all_ok else "PARTIAL",
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save_and_log(report: dict, filename: str) -> dict:
    """Saves report JSON + markdown to automation/reports/ and logs execution."""
    ts     = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_id = report.get("run_id", "unknown")

    # Save JSON
    json_path = REPORTS_DIR / f"report_{ts}_{run_id}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save markdown if it was generated
    md_path = None
    if report.get("markdown_report"):
        md_path = REPORTS_DIR / f"report_{ts}_{run_id}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report["markdown_report"])

    log_entry = {
        "run_id":       run_id,
        "timestamp":    datetime.utcnow().isoformat() + "Z",
        "filename":     filename,
        "status":       report.get("overall_status", "ERROR"),
        "duration":     report.get("duration_seconds"),
        "json_report":  str(json_path),
        "md_report":    str(md_path) if md_path else None,
    }
    with open(LOGS_DIR / "api_runs.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    report["report_saved_to"] = str(json_path)
    report["markdown_saved_to"] = str(md_path) if md_path else None
    return report


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """
    Health check — n8n HTTP Request node calls GET /health before triggering
    analysis to confirm the API is alive and Groq key is set.
    """
    groq_ok = bool(os.getenv("GROQ_API_KEY"))
    return {
        "status":       "OK" if groq_ok else "WARN",
        "warning":      None if groq_ok else "GROQ_API_KEY not set in .env",
        "timestamp":    datetime.utcnow().isoformat() + "Z",
        "groq_key_set": groq_ok,
        "version":      "V3",
    }


@app.get("/rag-status")
def rag_status():
    """
    Returns whether the RAG index is ready and lists all docs in knowledge/.
    Call this after uploading documents to confirm the index rebuilt correctly.
    """
    docs = [
        f.name for f in KNOWLEDGE_DIR.iterdir()
        if f.is_file() and f.suffix in ALLOWED_DOC_EXTENSIONS
    ]
    ready = _rag_ready()
    return {
        "rag_ready":      ready,
        "document_count": len(docs),
        "documents":      docs,
        "knowledge_dir":  str(KNOWLEDGE_DIR),
        "note": "Upload documents via POST /upload-docs to enable RAG."
                if not docs else
                "RAG index built. Next /analyze call will use these documents."
                if ready else
                "Documents found but index not built yet — try POST /upload-docs again.",
    }


@app.post("/upload-docs")
async def upload_docs(files: List[UploadFile] = File(...)):
    """
    Upload knowledge documents (PDF, DOCX, TXT, MD) to the knowledge/ folder.
    After saving, rebuilds the RAG index automatically.
    All subsequent /analyze calls will use RAG with these documents.

    curl example:
        curl -X POST http://127.0.0.1:8000/upload-docs
             -F "files=@policy.pdf"
             -F "files=@guidelines.md"
    """
    allowed   = []
    rejected  = []
    saved     = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_DOC_EXTENSIONS:
            rejected.append({"file": file.filename, "reason": f"Extension '{ext}' not allowed. Use PDF, DOCX, TXT, or MD."})
            continue
        allowed.append(file)

    # Save allowed files to knowledge/
    for file in allowed:
        try:
            contents  = await file.read()
            save_path = KNOWLEDGE_DIR / file.filename
            with open(save_path, "wb") as f:
                f.write(contents)
            saved.append(file.filename)
        except Exception as e:
            rejected.append({"file": file.filename, "reason": str(e)})

    # Rebuild RAG index if any files were saved
    rag_rebuilt = False
    rag_error   = None
    if saved:
        try:
            from rag.pipeline import build_rag_pipeline
            build_rag_pipeline()
            rag_rebuilt = True
        except Exception as e:
            rag_error = f"{type(e).__name__}: {e}"

    # List all docs now in knowledge/
    all_docs = [
        f.name for f in KNOWLEDGE_DIR.iterdir()
        if f.is_file() and f.suffix in ALLOWED_DOC_EXTENSIONS
    ]

    return {
        "saved":          saved,
        "rejected":       rejected,
        "rag_rebuilt":    rag_rebuilt,
        "rag_error":      rag_error,
        "all_documents":  all_docs,
        "document_count": len(all_docs),
        "status": "OK" if (saved and rag_rebuilt) else "PARTIAL" if saved else "FAIL",
        "note": "RAG index is live. Next /analyze call will use these documents."
                if rag_rebuilt else
                "Files saved but RAG index rebuild failed — check rag_error for details.",
    }


@app.post("/analyze-email")
async def analyze_email(files: List[UploadFile] = File(...)):
    """
    Single endpoint for the Gmail listener workflow.
    Send ALL email attachments here — it sorts them automatically:

    - .csv files       → analyzed by V2 pipeline
    - .pdf/.docx/.txt/.md → uploaded to knowledge/, RAG index rebuilt

    This means if a user emails both a CSV and a company policy PDF,
    the analysis will use the PDF as RAG context automatically.

    n8n HTTP Request node config:
        Method: POST
        URL: http://127.0.0.1:8000/analyze-email
        Body: Form-Data Multipart
        Add one parameter per attachment:
            Name: files | Input Field: attachment_0
            Name: files | Input Field: attachment_1  (if second attachment exists)
    """
    csv_files = []
    doc_files = []

    # Sort attachments by type
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext == ".csv":
            csv_files.append(f)
        elif ext in ALLOWED_DOC_EXTENSIONS:
            doc_files.append(f)

    if not csv_files:
        raise HTTPException(
            status_code=400,
            detail=f"No CSV file found in attachments. "
                   f"Received: {[f.filename for f in files]}. "
                   f"Please attach at least one .csv file."
        )

    # ── Upload documents and rebuild RAG index ────────────────────────────────
    docs_saved = []
    rag_rebuilt = False
    if doc_files:
        for doc in doc_files:
            try:
                contents = await doc.read()
                save_path = KNOWLEDGE_DIR / doc.filename
                with open(save_path, "wb") as f_out:
                    f_out.write(contents)
                docs_saved.append(doc.filename)
            except Exception as e:
                pass  # log but don't stop

        if docs_saved:
            try:
                from rag.pipeline import build_rag_pipeline
                build_rag_pipeline()
                rag_rebuilt = True
            except Exception:
                pass  # RAG failure doesn't stop CSV analysis

    # ── Read and analyze the CSV ──────────────────────────────────────────────
    csv_file = csv_files[0]  # use first CSV found
    try:
        contents = await csv_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        df = pd.read_csv(tmp_path)
        os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read CSV: {e}")

    # RAG is now auto-detected in run_v2_pipeline via _rag_ready()
    # If docs were uploaded above, index is rebuilt and RAG will be used
    try:
        report = run_v2_pipeline(df, csv_file.filename)
    except Exception:
        report = {
            "run_id":         str(uuid.uuid4())[:8],
            "timestamp":      datetime.utcnow().isoformat() + "Z",
            "filename":       csv_file.filename,
            "overall_status": "ERROR",
            "error":          traceback.format_exc(),
        }

    report["docs_uploaded"]  = docs_saved
    report["rag_rebuilt"]    = rag_rebuilt
    report = _save_and_log(report, csv_file.filename)
    return JSONResponse(content=report)


class FailureLog(BaseModel):
    workflow:         str
    failed_node:      str
    error:            str
    status_code:      int   = 0
    retry_count:      int   = 0
    csv_file:         str   = "unknown"
    duration_seconds: float = 0.0


@app.post("/log-failure")
def log_failure(body: FailureLog):
    """
    Called by n8n's failure branch when pipeline fails after all retries.
    Writes to automation/logs/failures.log — separate from api_runs.log.

    n8n HTTP Request node config:
        Method: POST
        URL:    http://127.0.0.1:8000/log-failure
        Body:   JSON
        {
          "workflow":    "AI Copilot Sub-workflow",
          "failed_node": "Analyze CSV",
          "error":       "{{ $json.error }}",
          "status_code": {{ $json.statusCode }},
          "retry_count": 3,
          "csv_file":    "{{ $('When Executed by Another Workflow').item.json.csv_file }}"
        }
    """
    entry = {
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "workflow":         body.workflow,
        "failed_node":      body.failed_node,
        "error":            body.error,
        "status_code":      body.status_code,
        "retry_count":      body.retry_count,
        "csv_file":         body.csv_file,
        "duration_seconds": body.duration_seconds,
        "status":           "FAILED",
    }
    failure_log = LOGS_DIR / "failures.log"
    with open(failure_log, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return {"logged": True, "entry": entry}


def download_report(run_id: str):
    """
    Returns the markdown report file for a given run_id as a file download.
    n8n HTTP Request node fetches this as binary → Email node attaches it.

    n8n HTTP Request node config:
        Method:          GET
        URL:             http://127.0.0.1:8000/download-report/{{ $json.run_id }}
        Response Format: File   ← critical: set this in Options tab
    """
    matches = list(REPORTS_DIR.glob(f"*_{run_id}.md"))
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"No markdown report found for run_id '{run_id}'. "
                   f"Report may not have generated if insights failed."
        )
    report_file = matches[0]
    return FileResponse(
        path=str(report_file),
        media_type="text/markdown",
        filename=report_file.name,
    )


@app.post("/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    """
    File upload endpoint.
    n8n sends a CSV as multipart/form-data → pipeline runs → returns JSON report.

    n8n HTTP Request node config:
        Method:       POST
        URL:          http://localhost:8000/analyze
        Body Content: Form Data
        Field name:   file  |  value: {{ $binary.data }}
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        df = pd.read_csv(tmp_path)
        os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read CSV: {e}")

    try:
        report = run_v2_pipeline(df, file.filename)
    except Exception:
        report = {
            "run_id":    str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "filename":  file.filename,
            "overall_status": "ERROR",
            "error":     traceback.format_exc(),
        }

    report = _save_and_log(report, file.filename)
    return JSONResponse(content=report)


class PathRequest(BaseModel):
    csv_path: str  # relative to project root, e.g. "data/sales.csv"


@app.post("/analyze-by-path")
def analyze_by_path(body: PathRequest):
    """
    Path-based endpoint — easier for n8n folder-watcher workflows.
    n8n sends: {"csv_path": "data/sales.csv"} as JSON body.

    n8n HTTP Request node config:
        Method:          POST
        URL:             http://localhost:8000/analyze-by-path
        Body Content:    JSON
        Body:            {"csv_path": "{{ $json.csv_path }}"}
    """
    full_path = ROOT / body.csv_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {full_path}")
    if not str(full_path).endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        df = pd.read_csv(full_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read CSV: {e}")

    try:
        report = run_v2_pipeline(df, body.csv_path)
    except Exception:
        report = {
            "run_id":    str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "filename":  body.csv_path,
            "overall_status": "ERROR",
            "error":     traceback.format_exc(),
        }

    report = _save_and_log(report, body.csv_path)
    return JSONResponse(content=report)