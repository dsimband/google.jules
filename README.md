# AI-Powered Job Review Generator

## Description

This application leverages the power of OpenAI's Large Language Models (LLMs), specifically GPT-3.5-turbo and GPT-4o, combined with Retrieval Augmented Generation (RAG) to create compelling and contextually relevant job reviews. It retrieves information from local documents to provide context to the LLMs, enabling them to generate more tailored and informative content.

## Features

*   **Retrieval Augmented Generation (RAG):** Utilizes a local `project_data/` directory as a knowledge base. The system scans for `.txt` and `.md` files, vectorizes their content, and retrieves the most relevant chunks based on the job query.
*   **Two-Pass LLM Generation:**
    1.  **Drafting:** GPT-3.5-turbo generates an initial draft of the job review based on the user's query and the RAG context.
    2.  **Refinement:** GPT-4o reviews and refines the draft, enhancing its clarity, conciseness, grammar, and engagement.
*   **Flexible Document Handling:** Supports `.txt` and `.md` file types within the `project_data/` directory for the RAG system.
*   **Environment-Based Configuration:** Securely manages the OpenAI API key via environment variables.

## Project Structure

```
.
├── app.py                # Main application script containing RAG logic, LLM calls, and execution flow.
├── project_data/         # Directory for storing .txt and .md files for RAG context.
│   └── .gitkeep          # Placeholder to ensure the directory is tracked by git.
├── tests/                # Contains unit and integration tests.
│   ├── test_rag.py       # Unit tests for RAG components (document loading, vectorization, retrieval).
│   ├── test_app.py       # Integration tests for the main application flow (mocking LLM calls).
│   └── .gitkeep          # Placeholder.
├── requirements.txt      # Lists Python dependencies for the project.
└── README.md             # This file.
```

## Setup Instructions

1.  **Clone the Repository (if applicable):**
    ```bash
    # If you have a git repository, clone it. Otherwise, ensure you have the project files.
    # git clone <repository_url>
    # cd <project_directory>
    ```

2.  **Navigate to the Project Directory:**
    If you cloned a repository, you're likely already there. Otherwise, `cd` into the directory where you've saved these files.

3.  **Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    ```

4.  **Activate the Virtual Environment:**
    *   On Linux/macOS:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows (Command Prompt):
        ```bash
        venv\Scripts\activate
        ```
    *   On Windows (PowerShell):
        ```bash
        .\venv\Scripts\Activate.ps1
        ```

5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Set up OpenAI API Key:**
    *   This project uses an `.env` file to manage the OpenAI API key, leveraging the `python-dotenv` library. This is the recommended way to handle your API key securely.
    *   An example file `.env.example` is provided. Copy it to create your own `.env` file:
        ```bash
        cp .env.example .env
        ```
    *   Open the newly created `.env` file (e.g., with a text editor) and add your actual OpenAI API key:
        ```env
        OPENAI_API_KEY="your_actual_api_key_here" 
        ```

        Replace `"your_actual_api_key_here"` with your real API key.
    *   **Important:** The `.env` file contains sensitive information and is included in `.gitignore` to prevent accidental commits. Do not remove it from `.gitignore`.
    *   (Alternative) While using a `.env` file is preferred, the application will still respect the `OPENAI_API_KEY` if it's set directly as an environment variable in your system (e.g., via `export` or `set`). However, the `.env` method takes precedence if the file exists and the variable is set within it, due to the `load_dotenv()` call in `app.py`.


2.  **Ensure API Access (OpenAI Account & Models):**
    Make sure your OpenAI account associated with the API key has sufficient credits and access to the `gpt-3.5-turbo` and `gpt-4o` models.


This configuration applies to both the Command-Line Application and the Web Application.

## Command-Line Application (app.py)

### Usage

1.  **Populate `project_data/` Directory:**
    Create or copy relevant text (`.txt`) or markdown (`.md`) files into the `project_data/` directory. These files will serve as the knowledge base for the RAG system. Examples include:
    *   Company values and mission statements.
    *   Summaries of past projects.
    *   Descriptions of team skills and expertise.
    *   Existing job description templates or role outlines.


2.  **Run the Application:**
    Execute the main script from the project's root directory:
    ```bash
    python app.py
    ```


3.  **Job Query Customization (for CLI):**
    Currently, the CLI application uses a hardcoded job query within the `main()` function in `app.py`. To generate reviews for different roles or requirements via the CLI, you will need to modify this query directly in the `app.py` script:
    ```python
    # In app.py, inside the main() function:
    job_query = "Generate a review for a senior software engineer specializing in Python and cloud technologies, requiring at least 5 years of experience and strong communication skills."
    # Change the string above to your desired query.
    ```


## Web Application (dashboard_app.py)

This project also includes a web interface built with Dash for a more interactive experience.

### How to Run the Web Application

1.  **Ensure Setup and Configuration are Complete:**
    Follow the "Setup Instructions" and "Configuration" sections above. The `pip install -r requirements.txt` step now also installs `dash` and `dash-bootstrap-components` required for the web app.

2.  **Run the Dash Application Server:**
    Execute the `dashboard_app.py` script from the project's root directory:
    ```bash
    python dashboard_app.py
    ```

3.  **Access in Web Browser:**
    Once the server is running (you should see output in your terminal similar to `Dash is running on http://0.0.0.0:8050/`), open your web browser and navigate to:
    `http://127.0.0.1:8050` (or `http://localhost:8050`)

    The application interface allows you to enter job details in a text area and generate the review by clicking a button. The output will be displayed on the page.


## Running Tests

The project includes unit and integration tests to ensure its components function correctly.

1.  **Navigate to the Project Root Directory.**
2.  **Run the Tests:**

    Ensure your virtual environment is activated and all dependencies from `requirements.txt` are installed.

    ```bash
    python -m unittest discover -s tests
    ```
    This command will automatically discover and run all tests within the `tests` directory.
```
