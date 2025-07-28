import os
import pandas as pd
import re # Import regex for parsing user prompt
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.utilities import PythonREPL
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline



from langgraph.graph import StateGraph, END

# Import matplotlib and set the 'Agg' backend for non-GUI environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# Helper function to generate a descriptive filename for plots
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

    llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.1-8b-instant")  # or your preferred Groq model

    # Step 3: Initialize PythonREPL tool
    python_repl = PythonREPL()
    python_repl.globals["df"] = state["df"]
    python_repl.globals["plt"] = plt # Make matplotlib available to the REPL's execution environment
    python_repl.globals["train_test_split"] = train_test_split
    python_repl.globals["StandardScaler"] = StandardScaler
    python_repl.globals["Pipeline"] = Pipeline


    # Step 4: Build LangGraph graph with the initial state (though not fully utilized in this simple loop)
    graph = StateGraph(AgentState)

    # We’ll keep a simple loop for user input and responses
    print("\n[🧠 AI Data Assistant Ready! Ask your questions about the dataset. Type 'exit' to quit.]\n")

    while True:
        user_prompt = input("You: ")
        if user_prompt.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Ask the LLM to generate Python code to analyze the current df based on user prompt
        # IMPORTANT: Instruct the LLM to ALWAYS save to "plot.png" and return a specific string.
        prompt_for_code = f"""
You are a helpful data science assistant with access to a pandas DataFrame called df and matplotlib as plt.

Your job is to generate Python code in response to user questions.

Follow these strict rules:
1.  Always ensure the final line of your code is an expression or variable that will be displayed in the terminal.
2.  If the final output is a pandas DataFrame or Series, convert it to a string using `.to_string()` for display (e.g., `df.describe().to_string()`).
3.  Do NOT use `print()`.
4.  If the user asks to save something to a CSV, use `.to_csv('filename.csv')`, but ensure the final line of your code still returns a summary or confirmation string.
5.  Do NOT use `.to_pdf()`, `.to_html()`, or any other save formats other than CSV.
6.  **FOR PLOTTING REQUESTS (e.g., histogram, boxplot, scatter plot, line plot, bar chart):**
    a.  **ALWAYS begin your plotting code with `plt.figure()` to create a new, clean plot.**
    b.  Generate the specific plotting function (e.g., `plt.hist(df['column'])`, `plt.boxplot(df['column'])`, `plt.scatter(df['col1'], df['col2'])`, `df['column'].plot()`, etc.).
    c.  **SAVE THE PLOT TO `plot.png` ONLY: `plt.savefig("plot.png")`.**
    d.  **IMMEDIATELY AFTER `plt.savefig()`, add `plt.close()` to ensure the plot resources are released.**
    e.  **CRITICAL: ABSOLUTELY, POSITIVELY NEVER USE `plt.show()` under any circumstances. It will not work and will cause errors.**
    f.  The final line of your code for ANY plotting request MUST be the exact string: `'Plot saved to plot.png'`.
7.  Never include import statements. Assume `df` and `plt` are already available.
8.  Only output valid Python code. Do not add explanations or comments.

9. If the user asks for preprocessing, model training, or splitting into X/y, you may use `train_test_split`, `Pipeline`, or `StandardScaler`.
These are already imported and available.
- Always make the final line of your code a print() statement. For example, print(df.describe()) or print(df.isnull().sum()).

User request: \"{user_prompt}\"
"""
        # Generate code from LLM
        code_response = llm.invoke(prompt_for_code)
        code_to_run = code_response.content.strip()

        print(f"\n[Generated code]:\n{code_to_run}\n")

        # Define the temporary plot file path that the LLM is instructed to use
        temp_plot_file_name = "plot.png"

        # Remove the temporary plot.png before generating a new one to ensure a fresh plot
        if os.path.exists(temp_plot_file_name):
            os.remove(temp_plot_file_name)

        try:
            # Execute the code on the current environment where df and plt exist
            execution_result = python_repl.run(code_to_run)
            print(f"Agent:\n{execution_result}")


            # Check if the temporary plot file was created
            if os.path.exists(temp_plot_file_name):
                # Generate a unique and descriptive filename
                new_plot_filename = generate_plot_filename(user_prompt)
                os.rename(temp_plot_file_name, new_plot_filename)
                print(f"Agent: Plot saved to {new_plot_filename}\n")
            
            # Print the output from the REPL if there's any direct textual result
            # This will catch the 'Plot saved to plot.png' string from the LLM
            if execution_result:
                print(f"Agent:\n{execution_result}\n")
            elif not os.path.exists(temp_plot_file_name) and not execution_result:
                # Only inform if no plot was saved (temporarily) AND no textual output
                print("Agent: Operation completed, but no direct output or plot was generated. Please refine your request if you expected a specific result.")

        except Exception as e:
            print(f"Error running Python code: {e}")

if __name__ == "__main__":
    main()
