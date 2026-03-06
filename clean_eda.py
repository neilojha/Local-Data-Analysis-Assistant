import pandas as pd
import matplotlib.pyplot as plt
import os

print("=" * 50)
print("       SUPERSTORE - EDA & CLEANING REPORT")
print("=" * 50)

# --- LOAD DATA ---
df = pd.read_csv("superstore.csv", encoding="latin1")
print(f"\n✅ Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# --- STEP 1: BASIC INFO ---
print("\n📋 COLUMN NAMES & DATA TYPES:")
print("-" * 40)
print(df.dtypes)

# --- STEP 2: MISSING VALUES ---
print("\n❓ MISSING VALUES:")
print("-" * 40)
missing = df.isnull().sum()
missing = missing[missing > 0]
if missing.empty:
    print("No missing values found ✅")
else:
    print(missing)

# --- STEP 3: DUPLICATES ---
print("\n🔁 DUPLICATE ROWS:")
print("-" * 40)
dupes = df.duplicated().sum()
print(f"Found {dupes} duplicate rows")

# --- STEP 4: BASIC STATS ---
print("\n📊 BASIC STATISTICS (Numeric Columns):")
print("-" * 40)
print(df.describe())

# --- STEP 5: UNIQUE VALUES IN KEY COLUMNS ---
print("\n🗂️ UNIQUE VALUES IN KEY COLUMNS:")
print("-" * 40)
for col in ["Region", "Category", "Sub-Category", "Segment", "Ship Mode"]:
    if col in df.columns:
        print(f"{col}: {df[col].unique()}")

# =====================
#   CLEANING STEPS
# =====================
print("\n")
print("=" * 50)
print("            CLEANING THE DATA")
print("=" * 50)

# Fix date columns
date_cols = [col for col in df.columns if "date" in col.lower() or "Date" in col]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    print(f"✅ Converted '{col}' to datetime")

# Drop duplicates
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"✅ Removed {before - after} duplicate rows")

# Strip whitespace from string columns
str_cols = df.select_dtypes(include=["object", "string"]).columns
df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())
print("✅ Stripped whitespace from all text columns")

# Fix column names (remove spaces, lowercase)
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")
print("✅ Cleaned column names (no spaces or dashes)")

# Drop rows where Sales or Profit is missing (critical columns)
critical_cols = [c for c in ["Sales", "Profit"] if c in df.columns]
before = len(df)
df = df.dropna(subset=critical_cols)
after = len(df)
print(f"✅ Dropped {before - after} rows with missing Sales/Profit")

# --- FINAL SUMMARY ---
print("\n")
print("=" * 50)
print("             FINAL CLEANED DATA SUMMARY")
print("=" * 50)
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
if "Order_Date" in df.columns:
    print(f"Date range: {df['Order_Date'].min()} → {df['Order_Date'].max()}")
if "Sales" in df.columns:
    print(f"Total Sales: ${df['Sales'].sum():,.2f}")
if "Profit" in df.columns:
    print(f"Total Profit: ${df['Profit'].sum():,.2f}")

# --- SAVE CLEANED DATA ---
output_path = "cleaned_superstore.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Cleaned data saved to '{output_path}'")


# =====================
#   FLEXIBLE PLOT FUNCTION
# =====================

def plot_graph(df, x_col, y_col, agg="sum", plot_type="line", save=True):
    """
    Plot a graph between any two columns.

    Parameters:
        df        : cleaned DataFrame
        x_col     : column for x-axis (categorical, date, or numeric)
        y_col     : column for y-axis (must be numeric)
        agg       : aggregation for y — 'sum', 'mean', 'count', 'max', 'min'
        plot_type : 'line', 'bar', 'scatter'
        save      : if True, saves the graph as a PNG
    """

    print(f"\n📈 Plotting: {y_col} vs {x_col}  |  agg={agg}  |  type={plot_type}")

    if x_col not in df.columns:
        print(f"❌ Column '{x_col}' not found. Available: {list(df.columns)}")
        return
    if y_col not in df.columns:
        print(f"❌ Column '{y_col}' not found. Available: {list(df.columns)}")
        return

    # --- Prepare x-axis ---
    # If x is a datetime column, extract month by default
    if pd.api.types.is_datetime64_any_dtype(df[x_col]):
        x_data = df[x_col].dt.to_period("M").astype(str)
        x_label = f"{x_col} (Month)"
    else:
        x_data = df[x_col]
        x_label = x_col

    # --- Aggregate ---
    if plot_type != "scatter":
        agg_func = {"sum": "sum", "mean": "mean", "count": "count",
                    "max": "max", "min": "min"}.get(agg, "sum")
        grouped = df.groupby(x_data)[y_col].agg(agg_func).reset_index()
        grouped.columns = ["x", "y"]
        grouped = grouped.sort_values("x")
        x_vals = grouped["x"]
        y_vals = grouped["y"]
    else:
        x_vals = df[x_col]
        y_vals = df[y_col]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(13, 6))

    if plot_type == "line":
        ax.plot(x_vals, y_vals, marker="o", linewidth=2,
                color="#2196F3", markersize=7, markerfacecolor="#FF5722")
        if len(x_vals) <= 20:
            for xv, yv in zip(x_vals, y_vals):
                ax.annotate(f"{yv:,.0f}", (xv, yv),
                            textcoords="offset points", xytext=(0, 10),
                            ha="center", fontsize=8, color="#333")

    elif plot_type == "bar":
        bars = ax.bar(x_vals, y_vals, color="#2196F3", edgecolor="white", linewidth=0.5)
        if len(x_vals) <= 20:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{bar.get_height():,.0f}",
                        ha="center", va="bottom", fontsize=8, color="#333")

    elif plot_type == "scatter":
        ax.scatter(x_vals, y_vals, color="#2196F3", alpha=0.6,
                   edgecolors="#333", linewidth=0.3)

    else:
        print(f"❌ Unknown plot_type '{plot_type}'. Use 'line', 'bar', or 'scatter'.")
        return

    ax.set_title(f"{agg.title()} of {y_col} by {x_label}", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(f"{agg.title()} of {y_col}", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save:
        fname = f"plot_{y_col}_vs_{x_col}_{plot_type}.png"
        plt.savefig(fname, dpi=150)
        print(f"✅ Graph saved as '{fname}'")

    plt.show()


# =====================
#   PLOT EXAMPLES
#   Add or edit these calls to plot whatever you need
# =====================

print("\n")
print("=" * 50)
print("              GENERATING PLOTS")
print("=" * 50)

# Sales by Month (line)
plot_graph(df, x_col="Order_Date", y_col="Sales", agg="sum", plot_type="line")

# Sales by Region (bar)
plot_graph(df, x_col="Region", y_col="Sales", agg="sum", plot_type="bar")

# Profit by Category (bar)
plot_graph(df, x_col="Category", y_col="Profit", agg="sum", plot_type="bar")

# Sales vs Profit (scatter)
plot_graph(df, x_col="Sales", y_col="Profit", plot_type="scatter")

# Average Discount by Sub-Category (bar)
plot_graph(df, x_col="Sub_Category", y_col="Discount", agg="mean", plot_type="bar")


print("\n🎉 EDA Complete! You can now run: streamlit run app.py")
print("=" * 50)
