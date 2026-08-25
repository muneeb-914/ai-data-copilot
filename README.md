# \# 🤖 AI Data Operations Copilot

# 

# An end-to-end AI-powered data analysis platform built with a multi-agent architecture, a RAG Knowledge Layer, and an enterprise automation layer. Upload any CSV dataset and a pipeline of AI agents automatically profiles, cleans, visualizes, and interprets your data — delivering business-ready insights grounded in both your data and your own company documents. In V3, the entire pipeline runs automatically with no manual steps required.

# 

# > \*\*V3 Complete\*\* — Enterprise Automation Layer. n8n orchestrates the full V2 pipeline automatically. Email a CSV (and optionally company documents) to the system and receive a complete analysis report back. Monitoring, retries, and failure alerts included.

# 

# > \*\*V2 Complete\*\* — Multi-agent pipeline + RAG Knowledge Layer. Agents now retrieve context from uploaded documents (PDF, DOCX, TXT, MD) and combine it with CSV analysis to produce grounded, cited answers.

# 

# > \*\*V1 Complete\*\* — Multi-agent pipeline with Groq LLM, Streamlit UI, IQR anomaly detection, AI-driven charts, and export.

# 

# \---

# 

# \## 🎯 Project Overview

# 

# Most data tools show you \*what\* the data says. This platform tells you \*so what\* and \*now what\* — backed by your own company knowledge, and now delivered automatically without you touching it.

# 

# A chain of specialized AI agents — each with a distinct role — work together to turn a raw CSV into a full business analysis report. In V2, agents retrieve relevant content from uploaded documents and combine it with the dataset to produce answers grounded in both structured data and unstructured knowledge. In V3, the entire pipeline is orchestrated by n8n: send a CSV by email and the report comes back to you automatically.

# 

# \*\*GitHub:\*\* \[github.com/muneeb-914/ai-data-copilot](https://github.com/muneeb-914/ai-data-copilot)

# 

# \---

# 

# \## ✨ Features

# 

# \### V3 — Enterprise Automation Layer

# 

# \#### 🔄 Two-Workflow n8n Architecture

# \- \*\*Gmail Listener Workflow\*\* — watches inbox via IMAP, detects emails with attachments, and hands off to the sub-workflow with binary files and sender metadata

# \- \*\*AI Copilot Sub-workflow\*\* — reusable processing engine triggered by any source (email, schedule, or webhook); calls FastAPI, runs all V2 agents, and emails the report back to the original sender

# 

# \#### 📬 Email-Triggered Analysis

# \- Send a CSV (or CSV + PDF/DOCX/TXT/MD company documents) to the inbox

# \- System detects the email, separates CSV from documents automatically, uploads documents to RAG if present, runs the full pipeline

# \- Analysis report delivered back to the sender as a `.md` attachment — no manual steps

# 

# \#### ⚡ FastAPI Automation Layer

# Seven REST endpoints wrap the V2 pipeline, making it callable by any system:

# 

# | Endpoint | Method | Purpose |

# |---|---|---|

# | `/health` | GET | Health check — confirms API and Groq key are live |

# | `/rag-status` | GET | RAG index readiness + list of uploaded documents |

# | `/upload-docs` | POST | Upload PDF/DOCX/TXT/MD → saves to `knowledge/` → rebuilds RAG index |

# | `/analyze` | POST | Upload CSV file → run full pipeline → return JSON report |

# | `/analyze-by-path` | POST | Pass local CSV path in JSON body → run pipeline |

# | `/analyze-email` | POST | Mixed attachments (CSV + docs) → auto-separates, uploads docs, runs pipeline with RAG |

# | `/download-report/{run\_id}` | GET | Serves `.md` report as file download for email attachment |

# 

# \#### 📊 Automated Reporting

# \- Every run saves both `.json` and `.md` reports to `automation/reports/`

# \- Every execution logged to `automation/logs/api\_runs.log` (status, duration, dataset, report path)

# \- `rag\_used: true/false` field in every response — confirms whether RAG was active

# 

# \#### 🔁 Monitoring, Retries, and Failure Alerts

# \- Health Check retries 2× with 5s wait before failing

# \- Run Pipeline retries 3× with 10s wait before routing to failure branch

# \- On failure: logs structured entry to `automation/logs/failures.log` via `/log-failure` endpoint, and sends failure notification email with error message, node, timestamp, and workflow name

# \- On success: downloads `.md` report and delivers it as email attachment

# 

# \#### 🤖 RAG Auto-Detection

# \- Pipeline checks RAG index on every run — no manual flag needed

# \- If documents exist in `knowledge/` and index is built → `use\_rag=True` passed to agents automatically

# \- Works for both scheduled runs and email-triggered runs

# 

# \---

# 

# \### V2 — RAG Knowledge Layer

# 

# \* Upload PDF, DOCX, TXT, or MD documents alongside your CSV

# \* Documents are chunked, embedded using `fastembed` (BAAI/bge-small-en-v1.5), and indexed in a local FAISS vector store

# \* At query time, the most relevant chunks are retrieved and passed to Groq as context

# \* Chat Agent, Insight Agent, and Cleaning Agent all use retrieved knowledge when documents are available

# \* Groq answers using both the CSV data and document context — hybrid reasoning

# \* Answers include source citations referencing which document was used

# \* Full chat history exportable as a Markdown report with citations per answer

# \* Rebuild Index button in sidebar for re-indexing after document changes

# 

# \### V1 — Multi-Agent Pipeline

# 

# \#### 🔍 Intelligent Dataset Profiling

# \* Automatic detection of column types, missing values, duplicates, and target column

# \* AI agent interprets the profile and identifies red flags, important columns, and recommended analysis

# 

# \#### 🧹 AI-Driven Data Cleaning

# \* Cleaning agent decides the right action per column (fill median, fill mode, drop, no action)

# \* In V2: follows company cleaning policies from uploaded documents when available

# \* Python executes the plan — AI decides, Python acts

# 

# \#### 🚨 Anomaly Detection

# \* IQR-based outlier detection (works on small and large datasets)

# \* Automatically skips binary columns and zero-IQR columns to avoid false positives

# 

# \#### 📉 AI-Recommended Charts

# \* Visualization agent reads the dataset profile and profile analysis

# \* Automatically selects the most insightful chart types

# \* Renders charts in a 2-column grid via Plotly

# 

# \#### 💡 Business Insight Report

# \* Insight agent produces a 3-section report: What the Data Shows, Key Problems Found, Recommendations

# \* In V2: enriched with domain knowledge retrieved from uploaded documents

# \* Every recommendation references actual column names and numbers

# 

# \#### 🤖 Chat with Memory + RAG

# \* Full conversational Q\&A about your dataset

# \* In V2: also answers questions about uploaded documents and combines both sources

# \* Remembers conversation history for follow-up questions

# 

# \#### 📥 Export

# \* Download cleaned dataset as CSV

# \* Download full analysis as a Markdown report

# \* \*\*V2:\*\* Download full chat history as a Markdown report with per-answer source citations

# 

# \---

# 

# \## 🏗️ Architecture

# 

# \### V3 — Enterprise Automation Layer

# 

# ```

# Gmail Inbox

# &#x20;   │

# &#x20;   ▼

# n8n Gmail Listener Workflow         ← IMAP trigger: catches emails with attachments

# &#x20;   │  Checks for attachments (IF node)

# &#x20;   │  Passes binary files + sender email to sub-workflow

# &#x20;   │

# &#x20;   ▼

# n8n AI Copilot Sub-workflow         ← reusable engine; callable by any trigger

# &#x20;   │

# &#x20;   ▼

# FastAPI /analyze-email              ← separates CSV from documents automatically

# &#x20;   │  PDF/DOCX/TXT/MD → saved to knowledge/ → RAG index rebuilt

# &#x20;   │  CSV → run\_v2\_pipeline() → RAG auto-detected

# &#x20;   │

# &#x20;   ▼

# V2 Multi-Agent Pipeline             ← intelligence layer (unchanged from V2)

# &#x20;   │  profiler → profile agent → cleaning agent → cleaner

# &#x20;   │  → anomaly detector → visualization agent → insight agent

# &#x20;   │

# &#x20;   ▼

# Report saved (.json + .md)          automation/reports/

# Logged                              automation/logs/api\_runs.log

# &#x20;   │

# &#x20;   ▼

# n8n Check Success (IF node)

# &#x20;   │

# &#x20;   ├── SUCCESS → GET /download-report/{run\_id} → Email .md to original sender

# &#x20;   │

# &#x20;   └── FAILURE → POST /log-failure → failures.log → Failure Notification Email

# ```

# 

# \### V1 Pipeline

# 

# ```

# CSV Upload

# &#x20;   │

# &#x20;   ▼

# Python Profiler            ← detects types, missing %, stats, target column

# &#x20;   │

# &#x20;   ▼

# Profile Agent (Groq)       ← interprets dataset, flags issues, recommends analysis

# &#x20;   │

# &#x20;   ▼

# Cleaning Agent (Groq)      ← decides fill/drop action per column

# &#x20;   │

# &#x20;   ▼

# Python Cleaner             ← executes cleaning plan on dataframe

# &#x20;   │

# &#x20;   ▼

# Anomaly Detector           ← IQR method, skips binary/constant columns

# &#x20;   │

# &#x20;   ▼

# Visualization Agent (Groq) ← reads profile + profile analysis → picks chart types

# &#x20;   │

# &#x20;   ▼

# Plotly Charts              ← renders AI-recommended visuals

# &#x20;   │

# &#x20;   ▼

# Insight Agent (Groq)       ← reads all agent outputs → business story + recommendations

# &#x20;   │

# &#x20;   ▼

# Chat Agent (Groq)          ← full context from all agents + conversation memory

# &#x20;   │

# &#x20;   ▼

# Export                     ← cleaned CSV + markdown report

# ```

# 

# \### V2 RAG Knowledge Layer

# 

# ```

# Document Upload (PDF / DOCX / TXT / MD)

# &#x20;   │

# &#x20;   ▼

# Loader                     ← reads raw text from each file

# &#x20;   │

# &#x20;   ▼

# Splitter                   ← chunks text into 500-char pieces with 100-char overlap

# &#x20;   │

# &#x20;   ▼

# Embedder (fastembed)       ← generates 384-dim semantic vectors per chunk

# &#x20;   │

# &#x20;   ▼

# FAISS Vector Index         ← stores vectors locally for similarity search

# &#x20;   │

# &#x20;   ▼

# Retriever                  ← embeds user query → searches index → returns top-k chunks

# &#x20;   │

# &#x20;   ▼

# Cleaning Agent  ──┐

# Insight Agent   ──┼── receive retrieved chunks as additional context → Groq reasons over both

# Chat Agent      ──┘

# &#x20;   │

# &#x20;   ▼

# Answer + Source Citations

# &#x20;   │

# &#x20;   ▼

# Export Chat Report with Citations

# ```

# 

# \*\*Core principle:\*\* Retrieve first. Reason second. Agents retrieve relevant knowledge before asking the LLM to reason.

# 

# \*\*Key design principle:\*\* AI decides, Python executes. Agents reason about the data; Python manipulates it.

# 

# \---

# 

# \## 📁 Folder Structure

# 

# ```

# ai-data-copilot/

# │

# ├── agents/

# │   ├── profile\_agent.py        # Dataset profiling and red flag detection

# │   ├── cleaning\_agent.py       # Per-column cleaning + company policy retrieval (V2)

# │   ├── visualization\_agent.py  # Chart type selection

# │   ├── insight\_agent.py        # Business insights + domain knowledge retrieval (V2)

# │   └── chat\_agent.py           # Conversational Q\&A + RAG retrieval (V2)

# │

# ├── core/

# │   ├── profiler.py             # Pure Python dataset profiling

# │   ├── analyzer.py             # Stats, summaries, anomaly detection

# │   ├── cleaner.py              # Cleaning plan execution engine

# │   ├── charts.py               # Plotly chart rendering

# │   └── exporter.py             # CSV, markdown report, chat report with citations (V2)

# │

# ├── rag/                        # V2 — RAG Knowledge Layer

# │   ├── loader.py               # Reads PDF, DOCX, TXT, MD from knowledge/ folder

# │   ├── splitter.py             # Chunks text with configurable size and overlap

# │   ├── embedder.py             # Generates semantic embeddings via fastembed

# │   ├── vectordb.py             # Builds and manages local FAISS vector index

# │   ├── retriever.py            # Embeds query, searches index, returns top-k chunks

# │   └── pipeline.py             # Wires full RAG flow; exposes build/query/answer functions

# │

# ├── automation/                 # V3 — Enterprise Automation Layer

# │   ├── scripts/

# │   │   ├── api.py              # FastAPI wrapper — 7 endpoints

# │   │   └── test\_setup.py       # Pipeline health check script

# │   ├── workflows/

# │   │   ├── m1\_scheduled\_pipeline.json

# │   │   ├── m2\_schedule\_api.json

# │   │   └── m3\_gmail\_listener.json

# │   ├── reports/                # Auto-generated .json + .md reports (not committed)

# │   └── logs/

# │       ├── api\_runs.log        # Execution log — JSON lines (not committed)

# │       └── failures.log        # Failure log — JSON lines (not committed)

# │

# ├── knowledge/                  # V2 — uploaded knowledge documents (not committed)

# │   └── your\_document.pdf

# │

# ├── utils/

# │   └── groq\_client.py          # Groq API connection and ask\_groq()

# │

# ├── data/                       # Local dataset storage

# ├── app.py                      # Streamlit application (V1 + V2)

# ├── .env                        # API keys (not committed)

# └── requirements.txt

# ```

# 

# \---

# 

# \## 🔍 How the RAG Layer Works Internally

# 

# \### 1. Document Loading (`rag/loader.py`)

# Reads raw text from uploaded files. Supports PDF (via pypdf), DOCX (via python-docx), TXT, and MD. Returns a list of dicts with `filename`, `filetype`, and `content`.

# 

# \### 2. Chunking (`rag/splitter.py`)

# Splits each document into overlapping text chunks (default: 500 characters, 100 overlap). Overlap ensures sentences at chunk boundaries are not lost. Every chunk carries its source `filename` so citations remain traceable.

# 

# \### 3. Embedding (`rag/embedder.py`)

# Uses `fastembed` with the `BAAI/bge-small-en-v1.5` model to generate 384-dimensional semantic vectors for each chunk. No PyTorch required — runs on ONNX Runtime. Model loads once per session (singleton pattern).

# 

# \### 4. Indexing (`rag/vectordb.py`)

# Builds a local FAISS `IndexFlatL2` index from the chunk vectors. Saves both the index and a parallel metadata list (filename, chunk index, text) to disk under `rag/faiss\_index/`. Metadata is stored separately from FAISS — vectors in FAISS, text in pickle.

# 

# \### 5. Retrieval (`rag/retriever.py`)

# At query time, embeds the user's question using the same model, searches the FAISS index for the closest vectors (L2 distance), and returns the top-k most relevant chunks with their source filenames and similarity scores. Lower score = more relevant.

# 

# \### 6. Pipeline (`rag/pipeline.py`)

# Wires all steps together as a shared service. `build\_rag\_pipeline()` runs steps 1–4 once and caches the index in memory. `query\_rag\_pipeline()` runs step 5 on any question. `answer\_with\_rag()` retrieves chunks and passes them to Groq as context. Any agent calls these functions — the RAG layer is not an agent itself.

# 

# \### 7. Citations

# Retrieved chunks are labeled `\[Knowledge Source N: filename]` in the Groq prompt. Groq naturally references these labels in its answers. The exporter scans answer text for document filenames and lists them as citations per Q\&A pair in the exported report.

# 

# \---

# 

# \## 🛠️ Tech Stack

# 

# | Layer | Technology |

# |---|---|

# | UI | Streamlit |

# | Automation / Orchestration | n8n |

# | API Layer | FastAPI + Uvicorn |

# | AI / LLM | Groq API (llama-3.3-70b-versatile) |

# | Embeddings | fastembed (BAAI/bge-small-en-v1.5, ONNX) |

# | Vector Search | FAISS (faiss-cpu) |

# | Document Parsing | pypdf, python-docx |

# | Data Processing | Pandas, NumPy |

# | Visualization | Plotly |

# | Environment | Python 3.10+, python-dotenv |

# | Version Control | Git / GitHub |

# 

# \---

# 

# \## ⚙️ Installation

# 

# \*\*1. Clone the repository\*\*

# 

# ```bash

# git clone https://github.com/muneeb-914/ai-data-copilot.git

# cd ai-data-copilot

# ```

# 

# \*\*2. Create and activate virtual environment\*\*

# 

# ```bash

# python -m venv venv

# 

# \# Windows

# venv\\Scripts\\activate

# 

# \# Mac/Linux

# source venv/bin/activate

# ```

# 

# \*\*3. Install dependencies\*\*

# 

# ```bash

# pip install -r requirements.txt

# ```

# 

# \*\*4. Set up environment variables\*\*

# 

# Create a `.env` file in the root:

# 

# ```

# GROQ\_API\_KEY=your\_groq\_api\_key\_here

# ```

# 

# Get a free Groq API key at: \[console.groq.com](https://console.groq.com)

# 

# \*\*5. Run the Streamlit app (V1 + V2)\*\*

# 

# ```bash

# streamlit run app.py

# ```

# 

# \*\*6. Run the automation layer (V3)\*\*

# 

# Open two terminals from the project root:

# 

# ```bash

# \# Terminal 1 — FastAPI pipeline server (port 8000)

# uvicorn automation.scripts.api:app --reload --port 8000

# 

# \# Terminal 2 — n8n workflow engine (port 5678)

# n8n start

# ```

# 

# Then open `http://localhost:5678`, import the workflow JSONs from `automation/workflows/`, and configure your SMTP and IMAP credentials.

# 

# \---

# 

# \## 📦 Requirements

# 

# ```

# streamlit

# pandas

# plotly

# groq

# python-dotenv

# numpy

# pypdf

# python-docx

# fastembed

# faiss-cpu

# fastapi

# uvicorn

# python-multipart

# ```

# 

# \---

# 

# \## 🗂️ Example Datasets

# 

# The platform works with any CSV. Tested on:

# 

# | Dataset | Source | Rows | Use Case |

# |---|---|---|---|

# | Cybersecurity Intrusion Detection | \[Kaggle](https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset) | 9,537 | Attack pattern analysis + RAG policy testing |

# 

# \---

# 

# \## 🗺️ Roadmap

# 

# \### ✅ V1 — Multi-Agent Pipeline

# Five specialized AI agents, Streamlit UI, anomaly detection, AI charts, export.

# 

# \### ✅ V2 — RAG Knowledge Layer

# Document upload, semantic search, hybrid CSV + document reasoning, citations, chat export.

# 

# \### ✅ V3 — Enterprise Automation Layer

# FastAPI automation layer with 7 endpoints. n8n two-workflow design: Gmail Listener catches emails and hands off to a reusable AI Copilot Sub-workflow. Pipeline runs automatically on email trigger or schedule. RAG auto-detected on every run. Monitoring with retries, failure logging, and failure notification emails. Reports saved and delivered as `.md` attachments.

# 

# \### ⬜ V4 — Authentication
# 

# \---

# 

# \## 👤 Author

# 

# \*\*Muneeb Ur Rehman\*\*

# BS Information Technology — University of Sargodha

# Data Analyst | AI Automation Enthusiast

# 

# \* GitHub: \[github.com/muneeb-914](https://github.com/muneeb-914)

# \* LinkedIn: \[linkedin.com/in/muneeb-ur-rehman-994223322](https://www.linkedin.com/in/muneeb-ur-rehman-994223322)

# \* Email: mu181842@gmail.com

# 

# \---

# 

# \*Built as a portfolio centerpiece targeting Data Analyst and Data Science roles — 2026.\*

