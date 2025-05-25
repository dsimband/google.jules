import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, exceptions as dash_exceptions # Renamed dash.exceptions
from dash.dependencies import Input, Output, State
import os
import sys
from dotenv import load_dotenv

# Imports from the CLI app (app.py)
# Assuming app.py is in the same directory or accessible via PYTHONPATH
try:
    from app import (
        load_documents,
        initialize_vectorizer,
        retrieve_relevant_chunk,
        generate_job_review_llm,
        # OPENAI_API_KEY_VALUE is handled by os.getenv here after load_dotenv
        # PROJECT_DATA_DIR will be defined globally here
    )
except ImportError as e:
    sys.stderr.write(f"Error importing from app.py: {e}\nEnsure app.py is in the same directory or PYTHONPATH is set.\nExiting.\n")
    sys.exit(1)


# --- Global Initializations ---

# Define Project Data Directory (if not easily importable or to override)
PROJECT_DATA_DIR = "project_data"

# Load .env file for environment variables like OPENAI_API_KEY
load_dotenv()

# API Key Setup
API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY_AVAILABLE = bool(API_KEY)
if not API_KEY_AVAILABLE:
    print("WARNING: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or in a .env file. LLM functionality will be disabled.", file=sys.stderr)
    # Note: The app.py's generate_job_review_llm and call_openai_api functions will also
    # check for the API key via its own global OPENAI_API_KEY_VALUE, which is set using os.getenv.
    # So, if load_dotenv() here makes it available to os.getenv, app.py functions should pick it up.

# RAG Data Loading (once at startup)
DOCUMENTS = []
VECTORIZER = None
TFIDF_MATRIX = None

print("Dashboard: Loading RAG documents...")
try:
    DOCUMENTS = load_documents(PROJECT_DATA_DIR)
    if DOCUMENTS:
        print(f"Dashboard: Loaded {len(DOCUMENTS)} documents for RAG.")
        VECTORIZER, TFIDF_MATRIX = initialize_vectorizer(DOCUMENTS)
        if VECTORIZER and TFIDF_MATRIX is not None:
            print("Dashboard: RAG Vectorizer initialized successfully.")
        else:
            print("Dashboard: Failed to initialize RAG vectorizer (VECTORIZER or TFIDF_MATRIX is None).", file=sys.stderr)
            # This might happen if documents are empty or all fail to process
    else:
        print("Dashboard: No documents found for RAG or load_documents returned empty.", file=sys.stderr)
except Exception as e:
    print(f"Dashboard: An error occurred during RAG data loading: {e}", file=sys.stderr)
    # Fallback: RAG components will remain None/empty


# Initialize the Dash app
# Using Dash Bootstrap Components for styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server # Expose server for potential WSGI deployment

# App layout
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("AI Job Review Generator"), width=12), className="mb-3 mt-3 text-center"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Enter Job Details:"),
            dcc.Textarea(
                id='job-query-input',
                placeholder="Paste job description, key responsibilities, desired skills, desired tone for the review (e.g., professional, friendly, urgent), company values to highlight, etc.",
                style={'width': '100%', 'height': 200},
                className="mb-2" # Added margin bottom for spacing
            ),
            dbc.Button("Generate Review", id='submit-button', color="primary", className="mt-2", n_clicks=0) # n_clicks initialized
        ], md=6), # Takes half width on medium screens and up
        dbc.Col([
            dbc.Label("Generated Job Review:"),
            dcc.Loading(
                id="loading-output",
                type="default", # or "circle", "cube", etc.
                children=[
                    # Using dcc.Markdown for better text formatting options like paragraphs and lists
                    dcc.Markdown(id='job-review-output', children="Your generated job review will appear here.", style={'white-space': 'pre-wrap', 'border': '1px solid #ccc', 'padding': '10px', 'min-height': '200px'})
                ]
            )
        ], md=6) # Takes the other half
    ]),
    # Footer example (optional)
    # dbc.Row(dbc.Col(html.Hr(), width=12), className="mt-5"),
    # dbc.Row(dbc.Col(html.P("© 2024 AI Job Review Generator", className="text-muted text-center small"), width=12))
], fluid=False) # fluid=False gives some padding around, fluid=True takes full width


# Callbacks will be defined here later
@app.callback(
    Output('job-review-output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('job-query-input', 'value')],
    prevent_initial_call=True # Good practice for callbacks triggered by n_clicks
)
def update_job_review(n_clicks, job_query):
    # if not n_clicks or n_clicks < 1: # prevent_initial_call=True handles the n_clicks is None case
    #     raise dash_exceptions.PreventUpdate # Or return dash.no_update

    if not API_KEY_AVAILABLE:
        # This check is important if app.py's functions don't robustly handle missing key themselves before API call
        return "Error: OpenAI API key is not configured. Please set it in your .env file or environment."

    if not job_query or job_query.strip() == "":
        return "Please enter job details in the text area."

    # Access global RAG components
    rag_context = ""
    if DOCUMENTS and VECTORIZER and TFIDF_MATRIX is not None: # Ensure TFIDF_MATRIX is also checked
        try:
            print(f"Dashboard: Retrieving RAG context for query: '{job_query[:50]}...'")
            rag_context = retrieve_relevant_chunk(job_query, VECTORIZER, TFIDF_MATRIX, DOCUMENTS, top_n=1)
            if not rag_context or "No relevant document found" in rag_context or "Error occurred" in rag_context:
                print(f"Dashboard: No relevant RAG context found or error: {rag_context}")
                rag_context = "" # Ensure it's an empty string for generate_job_review_llm
            else:
                print(f"Dashboard: RAG context retrieved (length: {len(rag_context)}).")
        except Exception as e:
            print(f"Dashboard: Error during RAG retrieval: {e}", file=sys.stderr)
            rag_context = "" # Proceed without RAG context on error
    else:
        print("Dashboard: RAG components not available. Proceeding without RAG context.")
    
    print(f"Dashboard: Submitting to LLM. Query: '{job_query[:50]}...', RAG context length: {len(rag_context)}")
    
    # generate_job_review_llm from app.py uses its own logic to access the API key,
    # typically via os.getenv() within its call_openai_api, which should work if load_dotenv() here was successful.
    # The app.py was modified in Turn 12 (Subtask 13) to set its global OPENAI_API_KEY_VALUE
    # using os.getenv() after its own load_dotenv() call.
    # If dashboard_app.py is the entry point, its load_dotenv() call is the one that matters for os.getenv().
    
    try:
        review = generate_job_review_llm(job_query, rag_context)
        
        if review:
            # Replace literal \n with Markdown line breaks if review is plain text
            # If generate_job_review_llm already returns Markdown, this isn't strictly needed
            # but can help ensure line breaks are rendered.
            return review.replace("\n", "  \n")
        else:
            # Check if generate_job_review_llm returned an error message it might print itself
            return "Failed to generate job review. The LLM call might have failed or returned empty. Check console for errors from 'app.py'."
    except Exception as e:
        print(f"Dashboard: An unexpected error occurred calling generate_job_review_llm: {e}", file=sys.stderr)
        return f"An unexpected error occurred: {e}"


if __name__ == '__main__':
    # TODO: Add host and port, debug mode from environment variables or config
    # For development, it's fine to run debug=True directly.
    # For production, debug should be False, and host/port might be 0.0.0.0 and an env var.
    app.run_server(debug=True, host='0.0.0.0', port=8050)
