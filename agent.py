from langchain_ollama import OllamaLLM
import pandas as pd

# Load LLM
llm = OllamaLLM(model="llama3", temperature=0)

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin1")
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    return df

def ask(df: pd.DataFrame, question: str) -> str:
    # Step 1: Ask LLM to generate pandas code
    prompt = f"""You are a Python data analyst. You have a pandas DataFrame called `df` with these columns:
{list(df.columns)}

Sample data:
{df.head(3).to_string()}

Write ONLY a single Python expression using pandas to answer this question:
"{question}"

Rules:
- Output ONLY the Python code, nothing else
- No markdown, no backticks, no explanation
- Must be a single expression that returns a value
- pandas is imported as pd, datetime as dt
- Date columns are already in datetime format

Code:"""

    code = llm.invoke(prompt).strip()
    # Clean up any accidental markdown
    code = code.replace("```python", "").replace("```", "").strip()

    # Step 2: Execute the generated code
    try:
        local_vars = {"df": df, "pd": pd}
        result = eval(code, {"__builtins__": {}}, local_vars)

        # Step 3: Ask LLM to explain the result in plain English
        explain_prompt = f"""A user asked: "{question}"
The pandas code `{code}` returned: {result}

Give a short, clear answer in 1-2 sentences explaining what this means."""

        explanation = llm.invoke(explain_prompt).strip()
        return f"**Result:** `{result}`\n\n**Insight:** {explanation}\n\n*Code used:* `{code}`"

    except Exception as e:
        return f"⚠️ Could not execute code: `{code}`\n\nError: `{str(e)}`"