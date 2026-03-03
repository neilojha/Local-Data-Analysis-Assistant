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
str_cols = df.select_dtypes(include=["object","string"]).columns
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
print(f"Date range: {df['Order_Date'].min()} → {df['Order_Date'].max()}" if "Order_Date" in df.columns else "")
print(f"Total Sales: ${df['Sales'].sum():,.2f}" if "Sales" in df.columns else "")
print(f"Total Profit: ${df['Profit'].sum():,.2f}" if "Profit" in df.columns else "")

# --- SAVE CLEANED DATA ---
output_path = "cleaned_superstore.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Cleaned data saved to '{output_path}'")
print("\n🎉 EDA Complete! You can now run: streamlit run app.py")
print("=" * 50)