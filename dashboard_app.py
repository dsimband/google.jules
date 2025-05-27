import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State # Will be used for callbacks later
import dash # Required for dash.no_update
# dash.exceptions will be imported if needed, for now relying on prevent_initial_call

from video_processor import download_video, process_video_with_pose
import shutil # For cleaning up temporary files/folders
# os is already imported
# base64 is not strictly needed if serving from /assets

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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder='assets')
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

    # Separator and Video Processing Section
    dbc.Row(dbc.Col(html.Hr(), width=12), className="mt-4 mb-4"), # Added margin for separation
    dbc.Row(dbc.Col(html.H2("YouTube Video Pose Analysis"), width=12), className="mb-3 mt-3 text-center"),
    dbc.Row([
        dbc.Col([
            dbc.Label("YouTube Video URL:"),
            dbc.Input(id='youtube-url-input', placeholder="Enter YouTube Video URL", type="url", className="mb-2"),
            dbc.Button("Process Video", id='process-video-button', color="success", className="mt-2", n_clicks=0),
            html.Div(id='video-status-output', className="mt-3", style={'white-space': 'pre-wrap'}), # Added pre-wrap for multi-line messages
            # Error output specifically for video processing, styled differently
            html.Div(id='video-error-output', className="mt-2 text-danger", style={'white-space': 'pre-wrap'})
        ], md=12), 
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-video-output",
                type="default",
                children=[
                    html.Video(id='processed-video-output', controls=True, style={'width': '100%', 'max-height': '500px', 'margin-top': '20px', 'display': 'none'}) # Initially hidden
                ]
            )
        ], md=12)
    ])
], fluid=False)

# Callbacks for Job Review
@app.callback(
    Output('job-review-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('job-query-input', 'value'),
    prevent_initial_call=True
)
def update_job_review(n_clicks_review, job_query): # Renamed n_clicks for clarity
    if not API_KEY_AVAILABLE:
        return "Error: OpenAI API key is not configured. Please set it in your .env file or environment."

    if not job_query or not job_query.strip():
        return "Please enter job details in the text area."

    rag_context = ""
    if DOCUMENTS and VECTORIZER and TFIDF_MATRIX is not None: # TFIDF_MATRIX was checked during init
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
    else:
        print("Dashboard Callback: RAG components not available. Proceeding without RAG context.")
    
    print(f"Dashboard Callback: Submitting to LLM. Query: '{job_query[:50]}...', RAG context length: {len(rag_context)}")
    
    try:
        review = generate_job_review_llm(job_query, rag_context)
        if review:
            return review.replace("\n", "  \n")
        else:
            return "Failed to generate job review. LLM returned no content. Check console for errors."
    except Exception as e:
        print(f"Dashboard Callback: Error during LLM call: {e}", file=sys.stderr)
        return f"Error during LLM call: {e}. Check console for details."

# Callback for Video Processing
@app.callback(
    [Output('processed-video-output', 'src'),
     Output('processed-video-output', 'style'), 
     Output('video-status-output', 'children'),
     Output('video-error-output', 'children')],
    [Input('process-video-button', 'n_clicks')],
    [State('youtube-url-input', 'value')],
    prevent_initial_call=True
)
def update_video_output(n_clicks_video, youtube_url):
    status_messages = []
    error_message = ""
    # Default to hiding video and no source
    video_src = "" 
    video_style = {'width': '100%', 'max-height': '500px', 'margin-top': '20px', 'display': 'none'} 

    if n_clicks_video is None or n_clicks_video == 0:
        # This state can occur if prevent_initial_call=False or on certain reloads.
        # With prevent_initial_call=True, this typically means no click has happened yet.
        return video_src, video_style, "Enter a YouTube URL and click 'Process Video' to begin.", ""

    if not youtube_url or not youtube_url.strip():
        status_messages.append("Please enter a valid YouTube URL.")
        return video_src, video_style, "\n".join(status_messages), "" # Clear any previous error

    # Define directories
    assets_dir = "assets" 
    temp_dir = "temp_video_downloads" 
    
    # Ensure directories exist
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create unique filenames using n_clicks_video to avoid caching issues and allow reprocessing.
    unique_suffix = str(n_clicks_video)
    downloaded_video_filename = f"downloaded_video_{unique_suffix}.mp4"
    processed_video_filename_in_assets = f"processed_video_assets_{unique_suffix}.mp4" # Filename within assets folder
    
    temp_download_path = os.path.join(temp_dir, downloaded_video_filename)
    # This is the actual filesystem path where the processed video will be saved (inside 'assets').
    final_filesystem_processed_path = os.path.join(assets_dir, processed_video_filename_in_assets)
    # This is the URL path that Dash will use to serve the video from the 'assets' directory.
    video_serve_path = f"/{assets_dir}/{processed_video_filename_in_assets}"

    # --- Cleanup potentially pre-existing files from a previous identical click (unlikely with n_clicks but defensive) ---
    if os.path.exists(temp_download_path):
        try: os.remove(temp_download_path)
        except OSError as e: status_messages.append(f"Log: Error removing old temp file {temp_download_path}: {e}")
    if os.path.exists(final_filesystem_processed_path):
        try: os.remove(final_filesystem_processed_path)
        except OSError as e: status_messages.append(f"Log: Error removing old asset file {final_filesystem_processed_path}: {e}")

    # --- Download Step ---
    status_messages.append(f"Attempting to download video from: {youtube_url}...")
    download_path_internal = None # To store the path returned by download_video
    try:
        download_path_internal = download_video(youtube_url, temp_download_path) # download_video saves to temp_download_path
        if not download_path_internal:
            status_messages.append("Error: Failed to download video. The URL might be invalid or the video unavailable.")
            error_message = "Download failed. Please check the YouTube URL or try a different video."
            return video_src, video_style, "\n".join(status_messages), error_message
        status_messages.append(f"Video downloaded successfully to: {download_path_internal}")
    except Exception as e:
        status_messages.append(f"An unexpected error occurred during video download: {str(e)}")
        error_message = f"Download error: {str(e)}"
        if download_path_internal and os.path.exists(download_path_internal): # Cleanup if download_path_internal was set and file exists
            try: os.remove(download_path_internal)
            except OSError as del_e: status_messages.append(f"Log: Failed to cleanup temp file on download error: {del_e}")
        return video_src, video_style, "\n".join(status_messages), error_message

    # --- Processing Step ---
    status_messages.append("Processing video for pose estimation. This may take a moment...")
    processed_output_path_actual = None # To store path from process_video_with_pose
    try:
        # process_video_with_pose will save the processed video to `final_filesystem_processed_path`
        processed_output_path_actual = process_video_with_pose(download_path_internal, final_filesystem_processed_path)
        
        if not processed_output_path_actual:
            status_messages.append("Error: Failed to process video. An issue occurred during pose detection.")
            error_message = "Video processing failed. This could be due to the video format or an internal error."
            if os.path.exists(download_path_internal): # Cleanup original downloaded file
                try: os.remove(download_path_internal)
                except OSError as del_e: status_messages.append(f"Log: Failed to cleanup temp file on processing error: {del_e}")
            return video_src, video_style, "\n".join(status_messages), error_message
        
        status_messages.append(f"Video processed successfully. Output saved to: {processed_output_path_actual}")
        status_messages.append(f"Video will be served from Dash assets path: {video_serve_path}")
        video_src = video_serve_path # Set the source for the video player
        video_style = {'width': '100%', 'max-height': '500px', 'margin-top': '20px', 'display': 'block'} # Make video visible
        error_message = "" # Clear any previous error message on full success

    except Exception as e:
        status_messages.append(f"An unexpected error occurred during video processing: {str(e)}")
        error_message = f"Processing error: {str(e)}"
        if os.path.exists(download_path_internal): # Cleanup original downloaded file
            try: os.remove(download_path_internal)
            except OSError as del_e: status_messages.append(f"Log: Failed to cleanup temp file on processing error: {del_e}")
        if os.path.exists(final_filesystem_processed_path): # Cleanup potentially partially processed file in assets
             try: os.remove(final_filesystem_processed_path)
             except OSError as del_e: status_messages.append(f"Log: Failed to cleanup asset file on processing error: {del_e}")
        return video_src, video_style, "\n".join(status_messages), error_message
    finally:
        # --- Final Cleanup of the original downloaded temporary file from `temp_dir` ---
        # The processed video in `assets` (final_filesystem_processed_path) is kept to be served by Dash.
        if download_path_internal and os.path.exists(download_path_internal):
            try:
                os.remove(download_path_internal)
                status_messages.append(f"Cleaned up temporary input file: {download_path_internal}")
            except OSError as e:
                status_messages.append(f"Log: Could not delete temporary input file {download_path_internal} after processing: {e}")
        # Old processed videos in `assets` (from previous n_clicks) are not cleaned up by this callback.

    return video_src, video_style, "\n".join(status_messages), error_message

if __name__ == '__main__':
    # Ensure assets directory exists at startup, though callback also checks
    if not os.path.exists("assets"):
        os.makedirs("assets")
    app.run_server(debug=True, host='0.0.0.0', port=8050)
```
