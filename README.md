# 🤖 AI Data Operations Copilot

An end-to-end AI-powered data analysis platform built with a multi-agent architecture. Upload any CSV dataset and a pipeline of AI agents automatically profiles, cleans, visualizes, and interprets your data — delivering business-ready insights and recommendations.

> \*\*V1 Complete\*\* — Multi-agent pipeline with Groq LLM, Streamlit UI, IQR anomaly detection, AI-driven charts, and export.

\---

## 🎯 Project Overview

Most data tools show you *what* the data says. This platform tells you *so what* and *now what*.

A chain of specialized AI agents — each with a distinct role — work together to turn a raw CSV into a full business analysis report. The user never has to write a single line of code or choose a single chart manually.

**Live demo:** \[Coming soon after deployment]  
**GitHub:** [github.com/muneeb-914/ai-data-copilot](https://github.com/muneeb-914/ai-data-copilot)

\---

## ✨ Features

### 🔍 Intelligent Dataset Profiling

* Automatic detection of column types, missing values, duplicates, and target column
* AI agent interprets the profile and identifies red flags, important columns, and recommended analysis

### 🧹 AI-Driven Data Cleaning

* Cleaning agent decides the right action per column (fill median, fill mode, drop, no action)
* Python executes the plan — AI decides, Python acts
* Supports pandas 2.0+ safe `fillna` patterns

### 🚨 Anomaly Detection

* IQR-based outlier detection (works on small and large datasets)
* Automatically skips binary columns and zero-IQR columns to avoid false positives

### 📉 AI-Recommended Charts

* Visualization agent reads the dataset profile and profile analysis
* Automatically selects the most insightful chart types (histogram, bar, boxplot, scatter, heatmap, pie)
* Renders charts in a 2-column grid via Plotly

### 💡 Business Insight Report

* Insight agent produces a 3-section report: What the Data Shows, Key Problems Found, Recommendations
* Every recommendation references actual column names and numbers
* Reads from all previous agents — not just raw data

### 🤖 Chat with Memory

* Full conversational Q\&A about your dataset
* Chat agent has context from every other agent: profile, cleaning report, anomalies, charts, insights
* Remembers conversation history for follow-up questions

### 📥 Export

* Download cleaned dataset as CSV
* Download full analysis as a Markdown report

\---

## 🏗️ Architecture

```
CSV Upload
    │
    ▼
Python Profiler          ← detects types, missing %, stats, target column
    │
    ▼
Profile Agent (Groq)     ← interprets dataset, flags issues, recommends analysis
    │
    ▼
Cleaning Agent (Groq)    ← decides fill/drop action per column
    │
    ▼
Python Cleaner           ← executes cleaning plan on dataframe
    │
    ▼
Anomaly Detector         ← IQR method, skips binary/constant columns
    │
    ▼
Visualization Agent (Groq) ← reads profile + profile analysis → picks chart types
    │
    ▼
Plotly Charts            ← renders AI-recommended visuals
    │
    ▼
Insight Agent (Groq)     ← reads all agent outputs → business story + recommendations
    │
    ▼
Chat Agent (Groq)        ← full context from all agents + conversation memory
    │
    ▼
Export                   ← cleaned CSV + markdown report
```

**Key design principle:** AI decides, Python executes. Agents reason about the data; Python manipulates it.

\---

## 📁 Folder Structure

```
ai-data-copilot/
│
├── agents/
│   ├── profile\_agent.py        # Dataset profiling and red flag detection
│   ├── cleaning\_agent.py       # Per-column cleaning recommendations
│   ├── visualization\_agent.py  # Chart type selection
│   ├── insight\_agent.py        # Business insights and recommendations
│   └── chat\_agent.py           # Conversational Q\&A with memory
│
├── core/
│   ├── profiler.py             # Pure Python dataset profiling
│   ├── analyzer.py             # Stats, summaries, anomaly detection
│   ├── cleaner.py              # Cleaning plan execution engine
│   ├── charts.py               # Plotly chart rendering
│   └── exporter.py             # CSV and markdown report export
│
├── utils/
│   └── groq\_client.py          # Groq API connection and ask\_groq()
│
├── data/                       # Local dataset storage
├── app.py                      # Streamlit application
├── .env                        # API keys (not committed)
└── requirements.txt
```

\---

## 🛠️ Tech Stack

|Layer|Technology|
|-|-|
|UI|Streamlit|
|AI / LLM|Groq API (llama-3.3-70b-versatile)|
|Data Processing|Pandas|
|Visualization|Plotly|
|Environment|Python 3.10+, python-dotenv|
|Version Control|Git / GitHub|
|Deployment (V1)|Localhost → Hostinger VPS (planned)|

\---

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
venv\\Scripts\\activate

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
GROQ\_API\_KEY=your\_groq\_api\_key\_here
```

Get a free Groq API key at: [console.groq.com](https://console.groq.com)

**5. Run the app**

```bash
streamlit run app.py
```

\---

## 📦 Requirements

Create `requirements.txt` with:

```
streamlit
pandas
plotly
groq
python-dotenv
numpy
```

\---

## 🗂️ Example Datasets

The platform works with any CSV. Tested on:

|Dataset|Source|Rows|Use Case|
|-|-|-|-|
|Cybersecurity Intrusion Detection|[Kaggle](https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset)|9,537|Attack pattern analysis|
|||||

Any sales, HR, financial, or operational CSV will work.

\---



## 🗺️ Future Roadmap

### V2 — RAG Layer *(Next)*

* Upload company documents, SOPs, historical reports
* Semantic search using ChromaDB + embeddings
* Chat answers questions using both data AND documents
* Smarter EDA engine that acts on profile agent recommendations

### V3 — n8n Orchestration

* Replace manual Python agent chaining with n8n visual workflows
* Each agent becomes an n8n node
* Trigger analysis automatically on file upload via webhook
* WhatsApp / Slack integration for insight delivery

### V4 — MLOps Layer

* Train ML models on uploaded datasets
* Track experiments with MLflow
* Version and deploy models via FastAPI
* Monitor model drift over time

\---

## 👤 Author

**Muneeb Ur Rehman**  
BS Information Technology — University of Sargodha  
Data Analyst | AI Automation Enthusiast

* GitHub: [github.com/muneeb-914](https://github.com/muneeb-914)
* LinkedIn: [www.linkedin.com/in/muneeb-ur-rehman-994223322](www.linkedin.com/in/muneeb-ur-rehman-994223322)
* Email: mu181842@gmail.com

\---

*Built as a portfolio centerpiece targeting Data Analyst and Data Scientist roles — 2026 job market.*

