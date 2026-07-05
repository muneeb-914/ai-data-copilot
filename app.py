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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Profile", "🧹 Clean", "📈 Stats", "🚨 Anomalies", "📉 Charts", "🤖 AI Q&A"
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
        if st.button("🧹 Clean Data Now"):
            cleaned_df, report = clean_data(st.session_state.raw_df.copy())
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
        st.subheader("Charts")
        numeric_cols = profile["numeric_cols"]
        categorical_cols = profile["categorical_cols"]

        if numeric_cols:
            st.markdown("#### Distribution")
            selected_col = st.selectbox("Select column", numeric_cols)
            fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)

            if len(numeric_cols) >= 2:
                st.markdown("#### Correlation Heatmap")
                corr = df[numeric_cols].corr()
                fig2 = px.imshow(corr, text_auto=True, title="Correlation Matrix")
                st.plotly_chart(fig2, use_container_width=True)

        if categorical_cols and numeric_cols:
            st.markdown("#### Category vs Numeric")
            cat_col = st.selectbox("Select category column", categorical_cols)
            num_col = st.selectbox("Select numeric column", numeric_cols, key="num")
            fig3 = px.bar(
                df.groupby(cat_col)[num_col].mean().reset_index(),
                x=cat_col, y=num_col,
                title=f"Average {num_col} by {cat_col}"
            )
            st.plotly_chart(fig3, use_container_width=True)

        if profile["datetime_cols"]:
            st.markdown("#### Time Series")
            date_col = st.selectbox("Select date column", profile["datetime_cols"])
            num_col_ts = st.selectbox("Select numeric column", numeric_cols, key="ts")
            fig4 = px.line(df, x=date_col, y=num_col_ts, title=f"{num_col_ts} over time")
            st.plotly_chart(fig4, use_container_width=True)

    # TAB 6 - Q&A
    with tab6:
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