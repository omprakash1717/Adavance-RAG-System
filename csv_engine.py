import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def _call_llm(prompt: str) -> str:
    """Call LLM with automatic fallback: DeepSeek -> Gemini OpenAI-compat."""
    providers = []

    if GEMINI_API_KEY:
        providers.append(("Gemini", GEMINI_API_KEY, "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.0-flash-lite"))
        providers.append(("Gemini", GEMINI_API_KEY, "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.0-flash"))
        providers.append(("Gemini", GEMINI_API_KEY, "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash"))
    if DEEPSEEK_API_KEY:
        providers.append(("DeepSeek", DEEPSEEK_API_KEY, "https://api.deepseek.com", "deepseek-chat"))

    last_error = None
    for name, key, base_url, model in providers:
        try:
            client = OpenAI(api_key=key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
                timeout=30,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            print(f"  {name}/{model} failed: {str(e)[:80]}... trying next")
            continue

    raise Exception(
        f"All AI providers are currently unavailable. "
        f"Please wait a minute and try again. Last error: {str(last_error)[:150]}"
    )


def process_csv(filepath: str) -> dict:
    """Loads a CSV or Excel file, returning basic info so the UI knows it's ready."""
    import pandas as pd
    print(f"Loading '{filepath}'...")

    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "columns_list": list(df.columns)
        }
    except Exception as e:
        raise Exception(f"Failed to process structured file: {str(e)}")


def query_csv(user_query: str, filepath: str) -> str:
    """Reads the dataset and passes it to LLM along with the user's query."""
    import pandas as pd
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        max_rows = 100
        truncated = False
        if len(df) > max_rows:
            df_subset = df.head(max_rows)
            truncated = True
        else:
            df_subset = df

        dataset_md = df_subset.to_markdown(index=False)
        truncation_note = f"\n[NOTE: Only the first {max_rows} of {len(df)} rows are shown.]" if truncated else ""

        prompt = f"""You are an expert Data Analyst and Assistant.
Analyze the data carefully and answer the user's question accurately.
Provide your answer in a clean, professional tone using Markdown for readability.
Keep your response concise.
{truncation_note}

DATASET:
{dataset_md}

USER QUESTION:
{user_query}"""

        return _call_llm(prompt)

    except Exception as e:
        raise Exception(f"Failed to query CSV: {str(e)}")


def get_csv_recommendations(filepath: str):
    """Generates analytical questions based on CSV columns and sample data."""
    import pandas as pd
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        cols = list(df.columns)
        sample = df.head(5).to_string()

        prompt = f"""You are a Data Analyst. Here is a snapshot of a dataset:
Columns: {cols}
Sample Data:
{sample}

Based on this, suggest 3 smart analytical questions the user should ask.
Keep them concise and professional.
Format: Return ONLY the questions as a JSON list of strings."""

        import json
        text = _call_llm(prompt)
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except:
        return ["Show some basic statistics", "What are the top 5 rows?", "Give me a summary of results"]
