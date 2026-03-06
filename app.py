import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from agent import load_data, ask

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
    df = pd.read_csv(csv_path, encoding="latin1")

    # Clean column names
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")

    # Auto-convert date columns
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

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

# ================================
#   FLEXIBLE GRAPH SECTION
# ================================
st.subheader("📈 Plot a Graph")

# Separate columns by type
numeric_cols = df.select_dtypes(include="number").columns.tolist()
all_cols = df.columns.tolist()

col_left, col_right = st.columns(2)

with col_left:
    x_col = st.selectbox("X-axis (group by)", options=all_cols,
                         index=all_cols.index("Category") if "Category" in all_cols else 0)
    plot_type = st.selectbox("Chart type", options=["bar", "line", "scatter"])

with col_right:
    y_col = st.selectbox("Y-axis (measure)", options=numeric_cols,
                         index=numeric_cols.index("Sales") if "Sales" in numeric_cols else 0)
    agg_func = st.selectbox("Aggregation", options=["sum", "mean", "count", "max", "min"],
                            disabled=(plot_type == "scatter"))

if st.button("🎨 Generate Plot"):
    try:
        fig, ax = plt.subplots(figsize=(12, 5))

        if plot_type == "scatter":
            # Scatter: plot raw x vs y directly
            if pd.api.types.is_numeric_dtype(df[x_col]):
                ax.scatter(df[x_col], df[y_col], alpha=0.5, color="#2196F3",
                           edgecolors="#333", linewidth=0.3)
                ax.set_xlabel(x_col, fontsize=12)
            else:
                st.warning("⚠️ Scatter plot requires a numeric X-axis column.")
                st.stop()
        else:
            # Group and aggregate
            if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                x_data = df[x_col].dt.to_period("M").astype(str)
                x_label = f"{x_col} (Month)"
            else:
                x_data = df[x_col]
                x_label = x_col

            grouped = df.groupby(x_data)[y_col].agg(agg_func).reset_index()
            grouped.columns = ["x", "y"]
            grouped = grouped.sort_values("x")

            if plot_type == "bar":
                bars = ax.bar(grouped["x"], grouped["y"],
                              color="#2196F3", edgecolor="white")
                # Value labels on bars
                for bar in bars:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.01,
                            f"{bar.get_height():,.0f}",
                            ha="center", va="bottom", fontsize=8)

            elif plot_type == "line":
                ax.plot(grouped["x"], grouped["y"], marker="o", linewidth=2,
                        color="#2196F3", markersize=7, markerfacecolor="#FF5722")
                # Value labels on points
                for _, row in grouped.iterrows():
                    ax.annotate(f"{row['y']:,.0f}", (row["x"], row["y"]),
                                textcoords="offset points", xytext=(0, 10),
                                ha="center", fontsize=8)

            ax.set_xlabel(x_label, fontsize=12)

        ax.set_title(f"{agg_func.title()} of {y_col} by {x_col}", fontsize=14,
                     fontweight="bold", pad=12)
        ax.set_ylabel(f"{agg_func.title()} of {y_col}", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Could not generate plot: {e}")


# ================================
#   CHAT INTERFACE
# ================================
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
