import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from agent import load_data,ask

st.set_page_config(page_title="📊 Data Analyst Assistant", layout="wide")
st.title("📊 Local Data Analysis Assistant")
st.caption("Powered by LangChain + Ollama (llama3) — 100% local")

# --- Upload or use default CSV ---
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    csv_path = "uploaded_data.csv"
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.read())
else:
    csv_path = "superstore.csv"

# --- Load data ---
try:
    df = pd.read_csv(csv_path)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

except FileNotFoundError:
    st.error("No CSV found. Please upload a file or add superstore.csv to the project folder.")
    st.stop()

# --- Auto EDA ---
with st.expander("🔍 Auto Summary (EDA)"):
    st.write(df.describe())

# --- Chat Interface ---
st.subheader("💬 Ask a Question About Your Data")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if prompt := st.chat_input("e.g. Which region has the highest profit?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                df_agent = load_data(csv_path)
                answer = ask(df_agent, prompt)
            except Exception as e:
                answer = f"Error: {str(e)}"
        
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})