import pandas as pd
from typing import TypedDict, Optional
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.utilities import PythonREPL
from langgraph import Graph

load_dotenv()  # Loads GROQ_API_KEY from .env

# 1. Define the state
class AgentState(TypedDict):  #this is never initialized as an instance; it is a guide as to how the states should look
    df: Optional[pd.DataFrame]
    file_name: Optional[str]

# 2. Load CSV into state
def load_csv(file_path: str) -> AgentState: #takes in file path and it must be a string. -> type hint
    # the function returns an object of type AgentState.  The type hint simply tells us that a string
    # must come in and get converted to the AgentState
    try:
        df = pd.read_csv(file_path)
        print(f"[✅] Loaded dataset: {file_path} with shape {df.shape}")
        return {"df": df, "file_name": file_path}  #dictionary with 2 keys:  df, file_name
    except Exception as e:
        print(f"[❌] Failed to load dataset: {e}")
        return {"df": None, "file_name": None}

# 3. Main loop (for now, just loads file)
if __name__ == "__main__":
    file_path = input("Enter path to your CSV file (e.g., data/titanic.csv): ").strip()
    state = load_csv(file_path)  #creating an instance of load_csv

    if state["df"] is not None:
        print(state["df"].head())
