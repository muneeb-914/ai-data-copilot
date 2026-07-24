import streamlit as st
import pandas as pd
import plotly.express as px
from core.profiler import profile_dataset
from agents.profile_agent import run_profile_agent
from core.analyzer import get_basic_stats, get_numeric_summary, detect_anomalies
from utils.groq_client import ask_groq

st.set_page_config(page_title="AI Data Copilot", page_icon="🤖", layout="wide")

st.title("🤖 AI Data Operations Copilot")
st.markdown("Upload your data and let AI analyze it for you.")

# ── SIDEBAR: Knowledge Document Upload (Milestone 4) ─────────────────────────
with st.sidebar:
    st.markdown("## 📚 Knowledge Documents")
    st.markdown("Upload documents to give AI additional context.")

    uploaded_docs = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key="knowledge_uploader"
    )

    if uploaded_docs:
        import os
        from pathlib import Path
        from rag.pipeline import build_rag_pipeline, get_pipeline_status, rebuild_rag_pipeline

        knowledge_dir = Path("knowledge")
        knowledge_dir.mkdir(exist_ok=True)

        # Track which docs are currently uploaded
        uploaded_names = [f.name for f in uploaded_docs]

        # Save new uploads to knowledge/ folder
        newly_saved = []
        for doc in uploaded_docs:
            doc_path = knowledge_dir / doc.name
            if not doc_path.exists():
                with open(doc_path, "wb") as f:
                    f.write(doc.read())
                newly_saved.append(doc.name)

        # If new documents were saved, rebuild the index
        if newly_saved:
            st.info(f"📄 New document(s) detected: {', '.join(newly_saved)}")
            with st.spinner("Indexing documents..."):
                result = rebuild_rag_pipeline()
            if result["status"] == "built":
                st.success(
                    f"✅ Indexed {result['documents']} document(s) → "
                    f"{result['chunks']} chunks → "
                    f"{result['index_total']} vectors"
                )
                st.session_state.rag_ready = True
            elif result["status"] == "empty":
                st.warning("No supported documents found in knowledge/ folder.")
                st.session_state.rag_ready = False
        else:
            # Documents already saved — load existing index
            status = get_pipeline_status()
            if status["ready"]:
                st.session_state.rag_ready = True
            else:
                with st.spinner("Loading knowledge index..."):
                    build_rag_pipeline()
                st.session_state.rag_ready = True

        # Show current index status
        st.markdown("---")
        st.markdown("#### 📊 Knowledge Index Status")
        status = get_pipeline_status()
        if status["ready"] and status["in_memory"]:
            st.success(f"🟢 Active — {status['index_total']} vectors loaded")
        elif status["ready"] and not status["in_memory"]:
            st.info("🟡 Index on disk — will load on first query")
        else:
            st.error("🔴 No index found")

        # Show uploaded documents list
        st.markdown("#### 📄 Uploaded Documents")
        for doc in uploaded_docs:
            st.markdown(f"- `{doc.name}`")

        # Manual rebuild button
        st.markdown("---")
        if st.button("🔄 Rebuild Index"):
            with st.spinner("Rebuilding knowledge index..."):
                result = rebuild_rag_pipeline()
            st.success(f"✅ Rebuilt — {result['index_total']} vectors")
            st.session_state.rag_ready = True

    else:
        st.info("No documents uploaded yet.")
        st.session_state.rag_ready = False

    # Show RAG status badge at bottom of sidebar
    st.markdown("---")
    rag_ready = st.session_state.get("rag_ready", False)
    if rag_ready:
        st.markdown("**RAG Status:** 🟢 Active")
    else:
        st.markdown("**RAG Status:** 🔴 Inactive")

# ── MAIN: CSV Upload ──────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load and store in session
    if "raw_df" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        st.session_state.raw_df = pd.read_csv(uploaded_file)
        st.session_state.df = st.session_state.raw_df.copy()
        st.session_state.file_name = uploaded_file.name
        st.session_state.cleaned = False
        st.session_state.profile = profile_dataset(st.session_state.raw_df)
        st.session_state.profile_analysis = None

    df = st.session_state.df
    profile = st.session_state.profile
    rag_ready = st.session_state.get("rag_ready", False)

    st.success(f"✅ Loaded {profile['rows']} rows and {profile['columns']} columns")
    if st.session_state.cleaned:
        st.info("🧹 Currently analyzing **cleaned** version of the data")
    if rag_ready:
        st.info("📚 Knowledge documents are active — AI agents will use them for context")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🔍 Profile", "🧹 Clean", "📈 Stats", "🚨 Anomalies", "📉 Charts", "💡 Insights", "🤖 AI Q&A", "📥 Export"
    ])

    # TAB 1 - PROFILE
    with tab1:
        st.subheader("Dataset Profile")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", profile["rows"])
        col2.metric("Columns", profile["columns"])
        col3.metric("Duplicates", profile["duplicates"])
        col4.metric("High Missing Cols", len(profile["high_missing_cols"]))

        st.markdown("#### Column Details")
        col_data = []
        for col_name, info in profile["column_details"].items():
            col_data.append({
                "Column": col_name,
                "Type": info["type"],
                "Missing %": info["missing_pct"],
                "Unique Values": info["unique_values"],
                "DType": info["dtype"]
            })
        st.dataframe(pd.DataFrame(col_data), use_container_width=True)

        st.markdown(f"**Guessed Target Column:** `{profile['guessed_target']}`")

        st.markdown("---")
        if st.button("🤖 Run AI Profile Analysis"):
            with st.spinner("Profiling dataset..."):
                st.session_state.profile_analysis = run_profile_agent(profile)

        if st.session_state.profile_analysis:
            st.markdown("#### AI Analysis")
            st.markdown(st.session_state.profile_analysis)

    # TAB 2 - CLEAN
    with tab2:
        st.subheader("Data Quality & Cleaning")

        raw = st.session_state.raw_df
        missing = raw.isnull().sum()
        missing_cols = missing[missing > 0]

        st.markdown("#### Missing Values")
        if not missing_cols.empty:
            st.dataframe(missing_cols.rename("Missing Count"), use_container_width=True)
        else:
            st.success("No missing values")

        st.markdown("#### Duplicate Rows")
        if profile["duplicates"] > 0:
            st.warning(f"{profile['duplicates']} duplicate rows found")
        else:
            st.success("No duplicates")

        st.markdown("---")
        if st.button("🤖 Generate AI Cleaning Plan"):
            with st.spinner("Analyzing what needs cleaning..."):
                from agents.cleaning_agent import run_cleaning_agent
                st.session_state.cleaning_plan = run_cleaning_agent(
                    profile,
                    use_rag=rag_ready       # uses company cleaning policies if available
                )

        if "cleaning_plan" in st.session_state:
            plan = st.session_state.cleaning_plan
            if "error" in plan:
                st.error(plan["error"])
                st.code(plan["raw"])
            else:
                st.markdown("#### AI Cleaning Plan")
                plan_rows = []
                for col, details in plan.get("columns", {}).items():
                    plan_rows.append({
                        "Column": col,
                        "Action": details.get("action"),
                        "Reason": details.get("reason")
                    })
                st.dataframe(pd.DataFrame(plan_rows), use_container_width=True)

                st.markdown("---")
                if st.button("✅ Execute Cleaning Plan"):
                    from core.cleaner import execute_cleaning_plan
                    cleaned_df, report = execute_cleaning_plan(
                        st.session_state.raw_df.copy(),
                        st.session_state.cleaning_plan
                    )
                    st.session_state.df = cleaned_df
                    st.session_state.cleaned = True
                    st.session_state.cleaning_report = report
                    for item in report:
                        st.write(item)
                    st.success("Done! All tabs now use cleaned data.")

    # TAB 3 - STATS
    with tab3:
        st.subheader("Numeric Summary")
        numeric_df = df.select_dtypes(include='number')
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe(), use_container_width=True)

    # TAB 4 - ANOMALIES
    with tab4:
        st.subheader("Anomaly Detection")
        anomalies = detect_anomalies(df)
        st.session_state.anomalies = anomalies
        for a in anomalies:
            if "No anomalies" in a:
                st.success(a)
            else:
                st.warning(a)

    # TAB 5 - CHARTS
    with tab5:
        st.subheader("AI-Recommended Charts")

        if st.button("📊 Generate Smart Charts"):
            with st.spinner("AI deciding what to visualize..."):
                from agents.visualization_agent import run_visualization_agent
                st.session_state.chart_specs = run_visualization_agent(
                    profile,
                    profile_analysis=st.session_state.profile_analysis or ""
                )

        if "chart_specs" in st.session_state and st.session_state.chart_specs:
            from core.charts import render_chart
            specs = st.session_state.chart_specs

            for i in range(0, len(specs), 2):
                col1, col2 = st.columns(2)
                with col1:
                    spec = specs[i]
                    st.caption(f"💡 {spec.get('reason', '')}")
                    fig = render_chart(df, spec)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                if i + 1 < len(specs):
                    with col2:
                        spec = specs[i+1]
                        st.caption(f"💡 {spec.get('reason', '')}")
                        fig = render_chart(df, spec)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
        elif "chart_specs" in st.session_state and not st.session_state.chart_specs:
            st.warning("Could not generate chart recommendations. Try again.")

    # TAB 6 - INSIGHTS
    with tab6:
        st.subheader("Business Insights")
        if rag_ready:
            st.info("📚 Knowledge documents will be used to enrich insights")

        if st.button("💡 Generate Business Insights"):
            with st.spinner("Analyzing business story..."):
                from agents.insight_agent import run_insight_agent
                stats_summary = df.select_dtypes(include='number').describe().to_string()
                st.session_state.insights = run_insight_agent(
                    profile,
                    stats_summary,
                    profile_analysis=st.session_state.get("profile_analysis") or "",
                    cleaning_report=st.session_state.get("cleaning_report") or [],
                    use_rag=rag_ready       # uses domain knowledge if available
                )

        if "insights" in st.session_state:
            st.markdown(st.session_state.insights)

    # TAB 7 - Q&A
    with tab7:
        st.subheader("💬 Chat with your Data")
        if rag_ready:
            st.info("📚 Knowledge documents are active — ask questions about both your data and uploaded documents")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            else:
                st.chat_message("assistant").markdown(msg["content"])

        question = st.chat_input("Ask anything about your data...")
        if question:
            st.chat_message("user").markdown(question)

            with st.spinner("Thinking..."):
                from agents.chat_agent import run_chat_agent
                df_sample = df.head(50).to_string()
                insights = st.session_state.get("insights") or ""

                answer = run_chat_agent(
                    question=question,
                    chat_history=st.session_state.chat_history,
                    profile=profile,
                    df_sample=df_sample,
                    insights=insights,
                    chart_specs=st.session_state.get("chart_specs") or [],
                    profile_analysis=st.session_state.get("profile_analysis") or "",
                    cleaning_report=st.session_state.get("cleaning_report") or [],
                    anomalies=st.session_state.get("anomalies") or [],
                    use_rag=rag_ready       # uses knowledge documents if available
                )

            st.chat_message("assistant").markdown(answer)

            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # TAB 8 - EXPORT
    with tab8:
        st.subheader("Export Results")
        from core.exporter import export_cleaned_csv, export_insight_report

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📄 Cleaned Dataset")
            if st.session_state.cleaned:
                csv_bytes = export_cleaned_csv(st.session_state.df)
                st.download_button(
                    label="⬇️ Download Cleaned CSV",
                    data=csv_bytes,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Clean your data first in the Clean tab.")

        with col2:
            st.markdown("#### 📊 Insight Report")
            report_md = export_insight_report(
                profile=profile,
                profile_analysis=st.session_state.get("profile_analysis") or "",
                cleaning_report=st.session_state.get("cleaning_report") or [],
                anomalies=st.session_state.get("anomalies") or [],
                insights=st.session_state.get("insights") or "",
                chart_specs=st.session_state.get("chart_specs") or []
            )
            st.download_button(
                label="⬇️ Download Report (.md)",
                data=report_md,
                file_name="analysis_report.md",
                mime="text/markdown"
            )
            st.markdown("#### Preview")
            st.markdown(report_md)

else:
    st.info("👆 Upload a CSV file to get started")