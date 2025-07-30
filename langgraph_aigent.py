import os
import pandas as pd
import re
from typing import TypedDict, Optional, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.utilities import PythonREPL
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage # New imports for messages

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# --- Step 1: Define the Agent State (Memory Blueprint) ---
class AgentState(TypedDict):
    df: Optional[pd.DataFrame]
    original_df: Optional[pd.DataFrame] # To preserve the raw original dataset
    file_name: Optional[str]
    messages: List[BaseMessage] # Conversation history for the LLM
    generated_code: Optional[str] # The Python code produced by the LLM
    execution_result: Optional[str] # The textual output from the Python REPL
    plot_saved_path: Optional[str] # Path to the last saved plot image
    csv_saved_paths: List[str] # List of paths to any saved CSV files
    final_response: Optional[str] # The message to display to the user

# --- Step 2: Initialize Core Tools and LLM (Global for simplicity initially) ---
# Load GROQ API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in a .env file.")

# Initialize LLM
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

# Initialize PythonREPL tool (globals will be set within nodes)
python_repl = PythonREPL()

# --- Helper function (from your original code, will be used by a node) ---
def generate_plot_filename(user_prompt: str) -> str:
    # Sanitize the prompt for use in a filename
    sanitized_prompt = re.sub(r'[^\w\s]', '', user_prompt).strip().replace(' ', '_')

    # Try to extract plot type (histogram, boxplot)
    plot_type_match = re.search(r'(histogram|boxplot|scatter|line|bar)', user_prompt, re.IGNORECASE)
    plot_type = plot_type_match.group(1).lower() if plot_type_match else "plot"

    # Try to extract column names (simple heuristic: look for words after 'of' or 'for')
    column_match = re.search(r'(of|for)\s+([a-zA-Z0-9_,\s]+)', user_prompt, re.IGNORECASE)
    column_name = ""
    if column_match:
        # Take the first word after 'of' or 'for' as a potential column name
        cols = [c.strip() for c in column_match.group(2).split(',') if c.strip()]
        if cols:
            column_name = cols[0]
            column_name = re.sub(r'[^\w]', '', column_name) # Remove non-alphanumeric
            column_name = column_name.lower()

    if column_name and plot_type:
        filename = f"{column_name}_{plot_type}.png"
    elif plot_type:
        filename = f"{plot_type}.png"
    elif sanitized_prompt:
        filename = f"{sanitized_prompt[:50]}.png" # Truncate long prompts
    else:
        filename = "unnamed_plot.png"

    # Add a timestamp to ensure absolute uniqueness, especially if parsing fails
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return f"{os.path.splitext(filename)[0]}_{timestamp}.png"


# --- Placeholder for the graph definition and main execution ---
if __name__ == "__main__":
    print("Milestone 1: Setup complete. AgentState and core tools initialized.")
    print("Ready to proceed to defining the first node in Milestone 2.")
    # We will build the graph and its execution here in later milestones.