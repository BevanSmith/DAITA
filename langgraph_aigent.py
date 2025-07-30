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
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# --- Step 1: Define the Agent State (Memory Blueprint) ---
class AgentState(TypedDict):
    df: Optional[pd.DataFrame]   #optional means can be whatever is inside the bracket or None.
    original_df: Optional[pd.DataFrame] # To preserve the raw original dataset
    file_name: Optional[str]
    messages: List[BaseMessage] # Conversation history for the LLM
    generated_code: Optional[str] # The Python code produced by the LLM
    execution_result: Optional[str] # The textual output from the Python REPL
    plot_saved_path: Optional[str] # Path to the last saved plot image
    csv_saved_paths: List[str] # List of paths to any saved CSV files
    final_response: Optional[str] # The message to display to the user

# --- Step 2: Initialize Core Tools and LLM (Global for simplicity initially) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in a .env file.")

llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
python_repl = PythonREPL()

# --- Helper function (from your original code, will be used by a node) ---
def generate_plot_filename(user_prompt: str) -> str:
    sanitized_prompt = re.sub(r'[^\w\s]', '', user_prompt).strip().replace(' ', '_')
    plot_type_match = re.search(r'(histogram|boxplot|scatter|line|bar)', user_prompt, re.IGNORECASE)
    plot_type = plot_type_match.group(1).lower() if plot_type_match else "plot"
    column_match = re.search(r'(of|for)\s+([a-zA-Z0-9_,\s]+)', user_prompt, re.IGNORECASE)
    column_name = ""
    if column_match:
        cols = [c.strip() for c in column_match.group(2).split(',') if c.strip()]
        if cols:
            column_name = cols[0]
            column_name = re.sub(r'[^\w]', '', column_name)
            column_name = column_name.lower()

    if column_name and plot_type:
        filename = f"{column_name}_{plot_type}.png"
    elif plot_type:
        filename = f"{plot_type}.png"
    elif sanitized_prompt:
        filename = f"{sanitized_prompt[:50]}.png"
    else:
        filename = "unnamed_plot.png"

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return f"{os.path.splitext(filename)[0]}_{timestamp}.png"

# --- NEW: Milestone 2: Data Loading Node ---
def load_data_node(state: AgentState) -> AgentState:
    """
    Loads a CSV file into a DataFrame and updates the agent state.
    The file path is expected to be in the latest HumanMessage content.
    """
    messages = state.get("messages", [])
    file_path = None  ## Ensures 'file_path' exists in scope with a default 'None' value.
    if messages and isinstance(messages[-1], HumanMessage):   #if messages exists and it is a HumanMessag; or rather, if the last message was human.
        # Assuming the file path is the content of the initial HumanMessage
        file_path = messages[-1].content   #.content is the actual message; -1 is to extract the last message from the messages list.
    else:
        print("[❌] Error: No file path provided in the initial HumanMessage.")
        messages.append(AIMessage(content="Error: Please provide a file path to load."))
        return {"df": None, "file_name": None, "messages": messages}  #the returns update the state; None because no file path to load.

# the first if else statement checks if there is a message and if its human.  
# the second if statement (below) says yes there was a message but checks if its empty
# if not file_path means falsiness: if X is empty or None or zero or False. 
    if not file_path:
        print("[❌] Error: File path is empty.")
        messages.append(AIMessage(content="Error: File path cannot be empty."))
        return {"df": None, "file_name": None, "messages": messages}

    try:
        df = pd.read_csv(file_path)
        print(f"[✅] Loaded dataset: {file_path} with shape {df.shape}")
        messages.append(AIMessage(content=f"Dataset '{file_path}' loaded successfully with shape {df.shape}."))
        return {"df": df.copy(), "original_df": df.copy(), "file_name": file_path, "messages": messages}
    except Exception as e:
        print(f"[❌] Failed to load dataset: {e}")
        messages.append(AIMessage(content=f"Failed to load dataset from '{file_path}': {e}"))
        return {"df": None, "original_df": None, "file_name": None, "messages": messages}



#so essentially, we load a user input filename in and it gets stored as humanmessages and is saved in the initial state.  
# This initial state is fed into the app graph which feeds into the node (load data) that gets the message and checks if its message and human; 
# if so it sets it as file_path; if not, it jumps out and returns None to the state.  If empty it jumps out and returns None to the state.  Then if we are
#  still around in the code (so to speak) it tries to
#  read csv the file path and proved a basic shape and append the AI message to the list and then return a bunch of df info.  If not then return None and exit.



# --- Main execution block ---
if __name__ == "__main__":
    print("\n--- LangGraph Data Assistant (Milestone 2) ---")

    # Step 3: Build LangGraph graph with the initial state
    graph = StateGraph(AgentState)

    # Add the data loading node
    graph.add_node("load_data", load_data_node)

    # Set the entry point of the graph
    graph.set_entry_point("load_data")

    # For Milestone 2, the graph simply ends after loading data
    graph.add_edge("load_data", END)

    # Compile the graph
    app = graph.compile()

    # Get the file path from the user
    user_file_path = input("Enter path to your CSV file (e.g., data/titanic.csv): ").strip()

    # Initial state for the graph invocation
    # The file path is passed as the content of the first HumanMessage

    #bevan:  the user feeds int the file path and it gets added to the state or initial_state
    # below.  then it is fed into the app (.invoke) which is the graph.  Then it goes to the 
    #first node which is load data node and looks for "messages" in the state.  Does the
    #state have messages?  and then is the last one (-1) a human message?  
    initial_state = {
        "messages": [HumanMessage(content=user_file_path)],
        "csv_saved_paths": [], # Initialize lists to empty
        "plot_saved_path": None,
        "generated_code": None,
        "execution_result": None,
        "final_response": None,
    }

    print("\n[Running the graph to load data...]")
    final_state = app.invoke(initial_state)

    print("\n--- Final State after Data Loading ---")
    if final_state["df"] is not None:
        print(f"DataFrame loaded. Shape: {final_state['df'].shape}")
        print(f"File Name: {final_state['file_name']}")
        print("\nFirst 5 rows of loaded DataFrame:")
        print(final_state['df'].head().to_string())
    else:
        print("Data loading failed.")

    print("\nAgent Messages:")
    for msg in final_state["messages"]:
        print(f"- {msg.type.upper()}: {msg.content}")

    print("\nMilestone 2 complete. Data loading integrated into LangGraph.")