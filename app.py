import streamlit as st
import pandas as pd
from utils.analyzer import load_data, get_basic_stats, get_numeric_summary, detect_anomalies
from utils.groq_client import ask_groq

st.set_page_config(page_title="AI Data Copilot", page_icon="🤖", layout="wide")

st.title("🤖 AI Data Operations Copilot")
st.markdown("Upload your data and let AI analyze it for you.")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "📈 Stats", "🚨 Anomalies", "🤖 AI Insight"])
    
    with tab1:
        st.subheader("Raw Data")
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.subheader("Numeric Summary")
        numeric_df = df.select_dtypes(include='number')
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe(), use_container_width=True)
        
        st.subheader("Column Info")
        stats = get_basic_stats(df)
        st.json(stats)
    
    with tab3:
        st.subheader("Anomaly Detection")
        anomalies = detect_anomalies(df)
        for a in anomalies:
            if "No anomalies" in a:
                st.success(a)
            else:
                st.warning(a)
    
    with tab4:
        st.subheader("AI Business Interpretation")
        if st.button("🤖 Generate AI Insight"):
            with st.spinner("Analyzing..."):
                stats = get_basic_stats(df)
                summary = get_numeric_summary(df)
                anomalies = detect_anomalies(df)
                
                prompt = f"""
                Dataset overview:
                - Rows: {stats['rows']}, Columns: {stats['columns']}
                - Columns: {stats['column_names']}
                
                Numeric summary:
                {summary}
                
                Anomalies: {anomalies}
                
                Give a concise business interpretation with:
                1. What the data shows
                2. Key issues or anomalies
                3. Actionable recommendations
                """
                
                result = ask_groq(prompt)
                st.markdown(result)

else:
    st.info("👆 Upload a CSV file to get started")