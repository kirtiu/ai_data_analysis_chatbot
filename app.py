import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import re

# ==============================
# 1 CONFIGURATION
# ==============================

st.set_page_config(page_title="Dataset Insight", layout="wide")
st.title("📊 AI Dataset Insight Chatbot ")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key
)

# ==============================
# 2 SESSION STATE
# ==============================

if "history" not in st.session_state:
    st.session_state.history = []

if "df" not in st.session_state:
    st.session_state.df = None

if "last_code" not in st.session_state:
    st.session_state.last_code = None


# ==============================
# 3 UTILITY FUNCTIONS
# ==============================

def is_safe_code(code: str) -> bool:
    dangerous = [
        "import", "os.", "sys.", "open(", "exec", "eval",
        "__", "subprocess", "shutil", "pickle", "write", "remove"
    ]
    return not any(word in code for word in dangerous)


def generate_code(question: str,df: pd.DataFrame) -> str:
    columns = ", ".join(df.columns)
    schema_info = df.dtypes.to_string()

    prompt = f"""
You are a senior Python data analyst.

The dataframe is called df.
Available columns (case sensitive):
{columns}
Column Data Types:
{schema_info}
Rules:
- Only generate pure Python pandas or matplotlib code.
- Do NOT import anything.
- Do NOT print().
- Store final answer in a variable named result.
- If visualization is needed, create a matplotlib plot.
- Never access files or OS.
- Never explain anything.
- Return ONLY executable Python code.

User Question:
{question}
"""
    response = llm.invoke(prompt)
    return response.content.strip()


def execute_code(code: str, df: pd.DataFrame):
    local_vars = {"df": df}

    try:
        exec(code, {}, local_vars)
        result = local_vars.get("result", None)
        return result, None
    except Exception as e:
        return None, str(e)


def explain_result(question: str, result):
    explanation_prompt = f"""
User asked:
{question}

The computed result is:
{result}

Explain this clearly in simple business language.
"""
    response = llm.invoke(explanation_prompt)
    return response.content


# ==============================
# 4 FILE UPLOAD
# ==============================

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file, engine="openpyxl")
    else:
        return None


uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.session_state.df = df

    st.subheader("Preview of Data")
    st.dataframe(df.head())

# ==============================
# 5 QUESTION PIPELINE
# ==============================

if st.session_state.df is not None:

    question = st.text_input("Ask a question about your data:")

    col1, col2 = st.columns([1,1])

    with col1:
        analyze_button = st.button("Analyze")

    with col2:
        regenerate_button = st.button("Regenerate Code")

    if analyze_button and question:

        with st.spinner("Generating analysis..."):
            def clean_code(code: str) -> str:
    # If markdown code block exists, extract inside it
                match = re.search(r"```(?:python)?\s*(.*?)```", code, re.DOTALL)
                if match:
                    return match.group(1).strip()

    # Otherwise just return cleaned string
                return code.strip()
            # Step 1: Generate Code
            raw_code = generate_code(question, st.session_state.df)
            code = clean_code(raw_code)
            st.session_state.last_code = code

            # Step 2: Safety Check
            if not is_safe_code(code):
                st.error("Unsafe code detected.")
                st.stop()

            # Step 3: Execute Code
            result, error = execute_code(code, st.session_state.df)

            if error:
                st.error(f"Execution Error: {error}")
                st.stop()

            # Step 4: Display Generated Code
            with st.expander("🧠 Generated Code"):
                st.code(code, language="python")

            # Step 5: Auto Plot Render
            if plt.get_fignums():
                st.pyplot(plt.gcf())
                plt.clf()

            # Step 6: Show Result
            if result is not None:
                st.subheader("📊 Result")
                st.write(result)

                # Download option
                if isinstance(result, pd.DataFrame):
                    csv = result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Result CSV",
                        csv,
                        "analysis_result.csv",
                        "text/csv"
                    )

            # Step 7: Explain Result
            explanation = explain_result(question, result)
            st.subheader("🤖 Explanation")
            st.write(explanation)

            # Step 8: Store Memory
            st.session_state.history.append({
                "question": question,
                "result": str(result),
                "explanation": explanation
            })


    # Regenerate Feature
    if regenerate_button and st.session_state.last_code:
        st.rerun()

# ==============================
# 6 MEMORY DISPLAY
# ==============================

if st.session_state.history:
    with st.expander("📜 Conversation History"):
        for item in st.session_state.history:
            st.write("**Q:**", item["question"])
            st.write("**Result:**", item["result"])
            st.write("**Explanation:**", item["explanation"])
            st.write("---")