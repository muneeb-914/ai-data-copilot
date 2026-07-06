import streamlit as st
import pandas as pd
import plotly.express as px
from core.profiler import profile_dataset
from agents.profile_agent import run_profile_agent
from utils.analyzer import get_basic_stats, get_numeric_summary, detect_anomalies, clean_data
from utils.groq_client import ask_groq

st.set_page_config(page_title="AI Data Copilot", page_icon="🤖", layout="wide")

st.title("🤖 AI Data Operations Copilot")
st.markdown("Upload your data and let AI analyze it for you.")

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

    st.success(f"✅ Loaded {profile['rows']} rows and {profile['columns']} columns")
    if st.session_state.cleaned:
        st.info("🧹 Currently analyzing **cleaned** version of the data")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔍 Profile", "🧹 Clean", "📈 Stats", "🚨 Anomalies", "📉 Charts", "💡 Insights", "🤖 AI Q&A"
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
                st.session_state.cleaning_plan = run_cleaning_agent(profile)

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
        for a in anomalies:
            if "No anomalies" in a:
                st.success(a)
            else:
                st.warning(a)

    # TAB 5 - CHARTS (data-driven based on profile)
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

            # Render in 2 column grid
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

    with tab6:
        st.subheader("Business Insights")
        if st.button("💡 Generate Business Insights"):
            with st.spinner("Analyzing business story..."):
                from agents.insight_agent import run_insight_agent
                stats_summary = df.select_dtypes(include='number').describe().to_string()
                st.session_state.insights = run_insight_agent(profile, stats_summary)

        if "insights" in st.session_state:
            st.markdown(st.session_state.insights)

    # TAB 6 - Q&A
    with tab7:
        st.subheader("Ask anything about your data")
        question = st.text_input("Your question", placeholder="e.g. Which protocol has the most anomalies?")
        if st.button("Ask AI") and question:
            with st.spinner("Thinking..."):
                sample = df.head(50).to_string()
                prompt = f"""
You are a data analyst. Here is a dataset sample:
{sample}

Dataset info:
- Rows: {profile['rows']}, Columns: {profile['columns']}
- Column names: {list(df.columns)}
- Target column: {profile['guessed_target']}

Answer this question: {question}

Be specific and give a business-oriented answer.
"""
                answer = ask_groq(prompt)
                st.markdown(answer)

else:
    st.info("👆 Upload a CSV file to get started")