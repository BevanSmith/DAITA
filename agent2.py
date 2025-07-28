import os
import pandas as pd
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import StateGraph, END

load_dotenv()  # Loads GROQ_API_KEY from .env

# Define the agent state to hold your DataFrame and filename
class AgentState(TypedDict):
    df: Optional[pd.DataFrame]
    file_name: Optional[str]

# Load CSV function (same as before)
def load_csv(file_path: str) -> AgentState:
    try:
        df = pd.read_csv(file_path)
        print(f"[✅] Loaded dataset: {file_path} with shape {df.shape}")
        return {"df": df, "file_name": file_path}
    except Exception as e:
        print(f"[❌] Failed to load dataset: {e}")
        return {"df": None, "file_name": None}

def main():
    # Step 1: Load your dataset
    file_path = input("Enter path to your CSV file (e.g., data/titanic.csv): ").strip()
    state = load_csv(file_path)
    if state["df"] is None:
        print("Exiting due to load failure.")
        return

    # Step 2: Initialize the Groq chat model
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in environment.")
        return

    llm = ChatGroq(api_key=groq_api_key,  model_name="llama-3.1-8b-instant")  # or your preferred Groq model

    # Step 3: Initialize PythonREPL tool
    python_repl = PythonREPL()
    python_repl.globals["df"] = state["df"]
    python_repl.globals["plt"] = plt # Make matplotlib available to the REPL's execution environment


    # Step 4: Build LangGraph graph with the initial state
    graph = StateGraph(AgentState)

    # We’ll keep a simple loop for user input and responses
    print("\n[🧠 AI Data Assistant Ready! Ask your questions about the dataset. Type 'exit' to quit.]\n")

    while True:
        user_prompt = input("You: ")
        if user_prompt.lower() in {"exit", "quit"}:  #.lower means to force everything to lower case 
            print("Goodbye!")
            break

        # Step 5: Agent flow - LLM generates Python code, PythonREPL executes it

        # For simplicity, we manually call the LLM + execute code here.
        # LangGraph lets you build more complex flows later.

        # Ask the LLM to generate Python code to analyze the current df based on user prompt
        prompt_for_code = f"""
YYou are a helpful data science assistant with access to a pandas DataFrame called df and matplotlib as plt.

Your job is to generate Python code in response to user questions.

Follow these rules:
- Always make sure the final line of your code is an expression or variable that will be displayed in the terminal (e.g., 'summary' or 'df.isnull().sum()').
- Do NOT use print().
- If the user asks to save something, use .to_csv('filename.csv'), but still return the summary object so it is printed.
- Do NOT use .to_pdf(), .to_html(), or any other save formats.
- If the user asks to plot something, use plt.savefig("plot.png") and plt.close(). Never use plt.show().
- Never include import statements. Assume df and plt are already available.
- Only output valid Python code. Do not add explanations or comments.
User request: \"{user_prompt}\"
"""

        # Generate code from LLM
        code_response = llm.invoke(prompt_for_code)
        code_to_run = code_response.content.strip()

        print(f"\n[Generated code]:\n{code_to_run}\n")

        try:
            # Execute the code on the current environment where df exists
            # PythonREPL expects a dict with 'code' key
            execution_result = python_repl.run(code_to_run)
            # Print the output (could be text or plot info)
            print(f"Agent:\n{execution_result}\n")
        except Exception as e:
            print(f"Error running Python code: {e}")

if __name__ == "__main__":
    main()
