import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State # Will be used for callbacks later
# dash.exceptions will be imported if needed, for now relying on prevent_initial_call

from app import (
    load_documents,
    initialize_vectorizer,
    retrieve_relevant_chunk,
    generate_job_review_llm
)
from dotenv import load_dotenv
import os
import sys # For potential sys.exit or error messages

# Global Initializations
PROJECT_DATA_DIR = "project_data"
load_dotenv() # Load .env file for API key

API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY_AVAILABLE = bool(API_KEY)

if not API_KEY_AVAILABLE:
    print("WARNING: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or in a .env file. LLM functionality will be disabled.", file=sys.stderr)

DOCUMENTS = []
VECTORIZER = None
TFIDF_MATRIX = None

print("Loading RAG documents...")
try:
    DOCUMENTS = load_documents(PROJECT_DATA_DIR)
    if DOCUMENTS:
        print(f"Loaded {len(DOCUMENTS)} documents for RAG.")
        VECTORIZER, TFIDF_MATRIX = initialize_vectorizer(DOCUMENTS)
        if VECTORIZER and TFIDF_MATRIX is not None: # Check both
            print("RAG Vectorizer initialized.")
        else:
            print("Failed to initialize RAG vectorizer.", file=sys.stderr)
            VECTORIZER = None # Ensure consistent state
            TFIDF_MATRIX = None
    else:
        print("No documents found for RAG or failed to load.", file=sys.stderr)
except Exception as e:
    print(f"Error during RAG initialization: {e}", file=sys.stderr)
    # Ensure RAG components are in a consistent state
    DOCUMENTS = []
    VECTORIZER = None
    TFIDF_MATRIX = None

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
                placeholder="Paste job description, key responsibilities, desired skills, tone for the review...",
                style={'width': '100%', 'height': 200},
                className="mb-2" # Added margin bottom
            ),
            dbc.Button("Generate Review", id='submit-button', color="primary", className="mt-2", n_clicks=0) # Added n_clicks
        ], md=6), 
        dbc.Col([
            dbc.Label("Generated Job Review:"),
            dcc.Loading(
                id="loading-output",
                type="default",
                children=[
                    # Using dcc.Markdown for potentially formatted output
                    dcc.Markdown(id='job-review-output', children="Your generated job review will appear here.", style={'white-space': 'pre-wrap', 'border': '1px solid #ccc', 'padding': '10px', 'min-height': '200px'}) 
                ]
            )
        ], md=6) 
    ]),
], fluid=False)

# Callbacks will be defined here later
@app.callback(
    Output('job-review-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('job-query-input', 'value'),
    prevent_initial_call=True
)
def update_job_review(n_clicks, job_query):
    if not API_KEY_AVAILABLE:
        return "Error: OpenAI API key is not configured. Please set it in your .env file or environment."

    if not job_query or not job_query.strip():
        return "Please enter job details in the text area."

    rag_context = ""
    if DOCUMENTS and VECTORIZER and TFIDF_MATRIX: # TFIDF_MATRIX was checked during init
        try:
            print(f"Dashboard Callback: Retrieving RAG context for query: '{job_query[:50]}...'")
            retrieved_context = retrieve_relevant_chunk(job_query, VECTORIZER, TFIDF_MATRIX, DOCUMENTS, top_n=1)
            if retrieved_context and "No relevant document found" not in retrieved_context and "Error occurred" not in retrieved_context :
                rag_context = retrieved_context
                print(f"Dashboard Callback: RAG context retrieved (length: {len(rag_context)}).")
            else:
                print(f"Dashboard Callback: No relevant RAG context found or error during retrieval: {retrieved_context}")
        except Exception as e:
            print(f"Dashboard Callback: Error during RAG retrieval: {e}", file=sys.stderr)
            # Optionally return an error message to UI here:
            # return f"Error during RAG retrieval: {e}"
    else:
        print("Dashboard Callback: RAG components not available. Proceeding without RAG context.")
    
    print(f"Dashboard Callback: Submitting to LLM. Query: '{job_query[:50]}...', RAG context length: {len(rag_context)}")
    
    try:
        review = generate_job_review_llm(job_query, rag_context)
        if review:
            # For dcc.Markdown, newlines need to be double spaces for <br> or actual paragraphs.
            # The generate_job_review_llm might already produce Markdown-friendly text.
            # If it's plain text with single \n, this replacement helps.
            return review.replace("\n", "  \n")
        else:
            return "Failed to generate job review. LLM returned no content. Check console for errors."
    except Exception as e:
        print(f"Dashboard Callback: Error during LLM call: {e}", file=sys.stderr)
        return f"Error during LLM call: {e}. Check console for details."

if __name__ == '__main__':
    # TODO: Add host and port, debug mode from environment variables or config
    app.run_server(debug=True, host='0.0.0.0', port=8050)
```
