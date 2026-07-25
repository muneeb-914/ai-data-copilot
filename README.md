# 🤖 AI Data Operations Copilot

An end-to-end AI-powered data analysis platform built with a multi-agent architecture and a RAG Knowledge Layer. Upload any CSV dataset and a pipeline of AI agents automatically profiles, cleans, visualizes, and interprets your data — delivering business-ready insights grounded in both your data and your own company documents.

> **V2 Complete** — Multi-agent pipeline + RAG Knowledge Layer. Agents now retrieve context from uploaded documents (PDF, DOCX, TXT, MD) and combine it with CSV analysis to produce grounded, cited answers.

> **V1 Complete** — Multi-agent pipeline with Groq LLM, Streamlit UI, IQR anomaly detection, AI-driven charts, and export.

---

## 🎯 Project Overview

Most data tools show you *what* the data says. This platform tells you *so what* and *now what* — and now backs every answer with your own company knowledge.

A chain of specialized AI agents — each with a distinct role — work together to turn a raw CSV into a full business analysis report. In V2, agents can also retrieve relevant content from uploaded documents and combine it with the dataset to produce answers that are grounded in both structured data and unstructured knowledge.

**GitHub:** [github.com/muneeb-914/ai-data-copilot](https://github.com/muneeb-914/ai-data-copilot)

---

## ✨ Features

### V2 — RAG Knowledge Layer

* Upload PDF, DOCX, TXT, or MD documents alongside your CSV
* Documents are chunked, embedded using `fastembed` (BAAI/bge-small-en-v1.5), and indexed in a local FAISS vector store
* At query time, the most relevant chunks are retrieved and passed to Groq as context
* Chat Agent, Insight Agent, and Cleaning Agent all use retrieved knowledge when documents are available
* Groq answers using both the CSV data and document context — hybrid reasoning
* Answers include source citations referencing which document was used
* Full chat history exportable as a Markdown report with citations per answer
* Rebuild Index button in sidebar for re-indexing after document changes

### V1 — Multi-Agent Pipeline

#### 🔍 Intelligent Dataset Profiling
* Automatic detection of column types, missing values, duplicates, and target column
* AI agent interprets the profile and identifies red flags, important columns, and recommended analysis

#### 🧹 AI-Driven Data Cleaning
* Cleaning agent decides the right action per column (fill median, fill mode, drop, no action)
* In V2: follows company cleaning policies from uploaded documents when available
* Python executes the plan — AI decides, Python acts

#### 🚨 Anomaly Detection
* IQR-based outlier detection (works on small and large datasets)
* Automatically skips binary columns and zero-IQR columns to avoid false positives

#### 📉 AI-Recommended Charts
* Visualization agent reads the dataset profile and profile analysis
* Automatically selects the most insightful chart types
* Renders charts in a 2-column grid via Plotly

#### 💡 Business Insight Report
* Insight agent produces a 3-section report: What the Data Shows, Key Problems Found, Recommendations
* In V2: enriched with domain knowledge retrieved from uploaded documents
* Every recommendation references actual column names and numbers

#### 🤖 Chat with Memory + RAG
* Full conversational Q&A about your dataset
* In V2: also answers questions about uploaded documents and combines both sources
* Remembers conversation history for follow-up questions

#### 📥 Export
* Download cleaned dataset as CSV
* Download full analysis as a Markdown report
* **V2:** Download full chat history as a Markdown report with per-answer source citations

---

## 🏗️ Architecture

### V1 Pipeline

```
CSV Upload
    │
    ▼
Python Profiler            ← detects types, missing %, stats, target column
    │
    ▼
Profile Agent (Groq)       ← interprets dataset, flags issues, recommends analysis
    │
    ▼
Cleaning Agent (Groq)      ← decides fill/drop action per column
    │
    ▼
Python Cleaner             ← executes cleaning plan on dataframe
    │
    ▼
Anomaly Detector           ← IQR method, skips binary/constant columns
    │
    ▼
Visualization Agent (Groq) ← reads profile + profile analysis → picks chart types
    │
    ▼
Plotly Charts              ← renders AI-recommended visuals
    │
    ▼
Insight Agent (Groq)       ← reads all agent outputs → business story + recommendations
    │
    ▼
Chat Agent (Groq)          ← full context from all agents + conversation memory
    │
    ▼
Export                     ← cleaned CSV + markdown report
```

### V2 RAG Knowledge Layer

```
Document Upload (PDF / DOCX / TXT / MD)
    │
    ▼
Loader                     ← reads raw text from each file
    │
    ▼
Splitter                   ← chunks text into 500-char pieces with 100-char overlap
    │
    ▼
Embedder (fastembed)       ← generates 384-dim semantic vectors per chunk
    │
    ▼
FAISS Vector Index         ← stores vectors locally for similarity search
    │
    ▼
Retriever                  ← embeds user query → searches index → returns top-k chunks
    │
    ▼
Cleaning Agent  ──┐
Insight Agent   ──┼── receive retrieved chunks as additional context → Groq reasons over both
Chat Agent      ──┘
    │
    ▼
Answer + Source Citations
    │
    ▼
Export Chat Report with Citations
```

**Core principle:** Retrieve first. Reason second. Agents retrieve relevant knowledge before asking the LLM to reason.

**Key design principle:** AI decides, Python executes. Agents reason about the data; Python manipulates it.

---

## 📁 Folder Structure

```
ai-data-copilot/
│
├── agents/
│   ├── profile_agent.py        # Dataset profiling and red flag detection
│   ├── cleaning_agent.py       # Per-column cleaning + company policy retrieval (V2)
│   ├── visualization_agent.py  # Chart type selection
│   ├── insight_agent.py        # Business insights + domain knowledge retrieval (V2)
│   └── chat_agent.py           # Conversational Q&A + RAG retrieval (V2)
│
├── core/
│   ├── profiler.py             # Pure Python dataset profiling
│   ├── analyzer.py             # Stats, summaries, anomaly detection
│   ├── cleaner.py              # Cleaning plan execution engine
│   ├── charts.py               # Plotly chart rendering
│   └── exporter.py             # CSV, markdown report, chat report with citations (V2)
│
├── rag/                        # V2 — RAG Knowledge Layer
│   ├── loader.py               # Reads PDF, DOCX, TXT, MD from knowledge/ folder
│   ├── splitter.py             # Chunks text with configurable size and overlap
│   ├── embedder.py             # Generates semantic embeddings via fastembed
│   ├── vectordb.py             # Builds and manages local FAISS vector index
│   ├── retriever.py            # Embeds query, searches index, returns top-k chunks
│   └── pipeline.py             # Wires full RAG flow; exposes build/query/answer functions
│
├── knowledge/                  # V2 — uploaded knowledge documents (not committed)
│   └── your_document.pdf
│
├── utils/
│   └── groq_client.py          # Groq API connection and ask_groq()
│
├── data/                       # Local dataset storage
├── app.py                      # Streamlit application
├── .env                        # API keys (not committed)
└── requirements.txt
```

---

## 🔍 How the RAG Layer Works Internally

### 1. Document Loading (`rag/loader.py`)
Reads raw text from uploaded files. Supports PDF (via pypdf), DOCX (via python-docx), TXT, and MD. Returns a list of dicts with `filename`, `filetype`, and `content`.

### 2. Chunking (`rag/splitter.py`)
Splits each document into overlapping text chunks (default: 500 characters, 100 overlap). Overlap ensures sentences at chunk boundaries are not lost. Every chunk carries its source `filename` so citations remain traceable.

### 3. Embedding (`rag/embedder.py`)
Uses `fastembed` with the `BAAI/bge-small-en-v1.5` model to generate 384-dimensional semantic vectors for each chunk. No PyTorch required — runs on ONNX Runtime. Model loads once per session (singleton pattern).

### 4. Indexing (`rag/vectordb.py`)
Builds a local FAISS `IndexFlatL2` index from the chunk vectors. Saves both the index and a parallel metadata list (filename, chunk index, text) to disk under `rag/faiss_index/`. Metadata is stored separately from FAISS — vectors in FAISS, text in pickle.

### 5. Retrieval (`rag/retriever.py`)
At query time, embeds the user's question using the same model, searches the FAISS index for the closest vectors (L2 distance), and returns the top-k most relevant chunks with their source filenames and similarity scores. Lower score = more relevant.

### 6. Pipeline (`rag/pipeline.py`)
Wires all steps together as a shared service. `build_rag_pipeline()` runs steps 1–4 once and caches the index in memory. `query_rag_pipeline()` runs step 5 on any question. `answer_with_rag()` retrieves chunks and passes them to Groq as context. Any agent calls these functions — the RAG layer is not an agent itself.

### 7. Citations
Retrieved chunks are labeled `[Knowledge Source N: filename]` in the Groq prompt. Groq naturally references these labels in its answers. The exporter scans answer text for document filenames and lists them as citations per Q&A pair in the exported report.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| AI / LLM | Groq API (llama-3.3-70b-versatile) |
| Embeddings | fastembed (BAAI/bge-small-en-v1.5, ONNX) |
| Vector Search | FAISS (faiss-cpu) |
| Document Parsing | pypdf, python-docx |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Environment | Python 3.10+, python-dotenv |
| Version Control | Git / GitHub |

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/muneeb-914/ai-data-copilot.git
cd ai-data-copilot
```

**2. Create and activate virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at: [console.groq.com](https://console.groq.com)

**5. Run the app**

```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
pandas
plotly
groq
python-dotenv
numpy
pypdf
python-docx
fastembed
faiss-cpu
```

---

## 🗂️ Example Datasets

The platform works with any CSV. Tested on:

| Dataset | Source | Rows | Use Case |
|---|---|---|---|
| Cybersecurity Intrusion Detection | [Kaggle](https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset) | 9,537 | Attack pattern analysis + RAG policy testing |

---

## 🗺️ Roadmap

### ✅ V1 — Multi-Agent Pipeline
Five specialized AI agents, Streamlit UI, anomaly detection, AI charts, export.

### ✅ V2 — RAG Knowledge Layer
Document upload, semantic search, hybrid CSV + document reasoning, citations, chat export.

### ⬜ V3 — n8n Orchestration
Replace manual Python agent chaining with n8n visual workflows. Each agent becomes an n8n node. Trigger analysis automatically on file upload via webhook. WhatsApp / Slack integration for insight delivery.

### ⬜ V4 — MLOps Layer
Train ML models on uploaded datasets. Track experiments with MLflow. Version and deploy models via FastAPI. Monitor model drift over time.

---

## 👤 Author

**Muneeb Ur Rehman**
BS Information Technology — University of Sargodha
Data Analyst | AI Automation Enthusiast

* GitHub: [github.com/muneeb-914](https://github.com/muneeb-914)
* LinkedIn: [linkedin.com/in/muneeb-ur-rehman-994223322](https://www.linkedin.com/in/muneeb-ur-rehman-994223322)
* Email: mu181842@gmail.com

---

*Built as a portfolio centerpiece targeting Data Analyst and Data Science roles — 2026.*
