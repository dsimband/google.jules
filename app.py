import os
from pathlib import Path  # Using pathlib for better path manipulation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai # Added for OpenAI API calls
import sys # Added for sys.exit

# --- OpenAI API Key Handling ---
# The API key is checked within main() now.
OPENAI_API_KEY_VALUE = os.getenv("OPENAI_API_KEY")


def load_documents(directory_path: str) -> list[str]:
    """
    Scans the directory for .txt and .md files, reads their content,
    and returns a list of strings.

    Args:
        directory_path: The path to the directory containing the documents.

    Returns:
        A list of strings, where each string is the content of a file.
    """
    documents = []
    # Using pathlib to glob for files for easier cross-platform compatibility
    p = Path(directory_path)
    file_paths = list(p.glob('**/*.txt')) + list(p.glob('**/*.md'))

    if not file_paths:
        print(f"Warning: No .txt or .md files found in {directory_path}")
        return documents

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())
        except FileNotFoundError:
            print(f"Warning: File not found {file_path}, skipping.")
        except IOError as e:
            print(f"Warning: Could not read file {file_path} due to IO error: {e}, skipping.")
        except Exception as e:
            print(f"Warning: An unexpected error occurred with file {file_path}: {e}, skipping.")
    return documents

def initialize_vectorizer(documents: list[str]):
    """
    Initializes and fits a TfidfVectorizer with the given documents.

    Args:
        documents: A list of strings, where each string is the content of a document.

    Returns:
        A tuple containing the fitted TfidfVectorizer and the TF-IDF matrix.
        Returns (None, None) if documents list is empty or an error occurs.
    """
    if not documents:
        print("Warning: No documents provided to initialize vectorizer.")
        return None, None
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        return vectorizer, tfidf_matrix
    except Exception as e:
        print(f"Error initializing TfidfVectorizer: {e}")
        return None, None

# --- OpenAI API Integration ---

def call_openai_api(model_name: str, system_prompt: str, user_prompt: str, context: str = "") -> str:
    """
    Calls the OpenAI ChatCompletion API with the given parameters.

    Args:
        model_name: The name of the OpenAI model to use (e.g., "gpt-3.5-turbo", "gpt-4o").
        system_prompt: The system message to guide the LLM's behavior.
        user_prompt: The user's message or query.
        context: Additional context (e.g., from RAG) to be included.

    Returns:
        The content of the assistant's message from the API response, or an empty string if an error occurs.
    """
    if not OPENAI_API_KEY_VALUE:
        print("Error: OpenAI API key is not configured. Cannot call the API.")
        return ""

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY_VALUE)
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        # Incorporate context effectively. Here, it's added before the main user prompt.
        # Adjust based on how you find it works best.
        full_user_prompt = f"Context from retrieved documents:\n{context}\n\nUser Query/Instruction:\n{user_prompt}"
        messages.append({"role": "user", "content": full_user_prompt})

        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            print("Warning: Received an unexpected response structure from OpenAI API.")
            return ""
    except openai.APIConnectionError as e:
        print(f"OpenAI API Connection Error: {e}")
    except openai.RateLimitError as e:
        print(f"OpenAI API Rate Limit Exceeded: {e}")
    except openai.AuthenticationError as e:
        print(f"OpenAI API Authentication Error: {e}. Check your API key.")
    except openai.APIStatusError as e:
        print(f"OpenAI API Status Error: {e.status_code} - {e.response}")
    except Exception as e:
        print(f"An unexpected error occurred when calling OpenAI API: {e}")
    return ""


def generate_job_review_llm(job_details_query: str, rag_context: str) -> str:
    """
    Generates a job review using a two-pass approach with OpenAI LLMs.
    Pass 1: Draft generation with GPT-3.5-turbo.
    Pass 2: Refinement with GPT-4o.

    Args:
        job_details_query: The user's initial query or details for the job review.
        rag_context: Context retrieved by the RAG system.

    Returns:
        The generated (and potentially refined) job review, or an error message.
    """
    system_prompt_job_review = (
        "You are an expert HR assistant specialized in writing compelling and informative job reviews. "
        "Your goal is to attract suitable candidates. Use the provided context and job details "
        "to create a draft of a job review. Ensure the tone is professional and engaging."
    )

    # --- First Pass (GPT-3.5-turbo for drafting) ---
    user_prompt_draft = (
        f"Based on the following job details and context, please draft a job review.\n\n"
        f"Job Details/Query: {job_details_query}\n\n"
        f"Context from internal documents: {rag_context}"
    )

    print("Generating initial draft with gpt-3.5-turbo...")
    draft_review = call_openai_api(
        model_name="gpt-3.5-turbo",
        system_prompt=system_prompt_job_review,
        user_prompt=user_prompt_draft,
        context=rag_context # Context is also passed directly to call_openai_api
    )

    if not draft_review:
        return "Error: Failed to generate the initial draft for the job review. The RAG context might have been too large or an API error occurred."

    print(f"Draft generated:\n{draft_review[:200]}...\n") # Print snippet of draft

    # --- Second Pass (GPT-4o for refinement) ---
    system_prompt_refinement = ( # Could be the same or slightly adjusted for refinement
        "You are an expert HR editor. Your task is to refine and enhance the provided draft job review. "
        "Ensure it is clear, concise, grammatically perfect, and highly engaging. "
        "Use the original job details and context for accuracy."
    )
    user_prompt_refine = (
        f"Please review and refine the following draft job review. "
        f"Make sure it aligns well with the initial job details and provided context.\n\n"
        f"Original Job Details/Query: {job_details_query}\n\n"
        f"Context from internal documents: {rag_context}\n\n" # Re-provide context for completeness
        f"Draft to Refine:\n{draft_review}"
    )

    print("Refining draft with gpt-4o...")
    refined_review = call_openai_api(
        model_name="gpt-4o", # Using GPT-4o for refinement
        system_prompt=system_prompt_refinement,
        user_prompt=user_prompt_refine,
        # The draft is part of the user_prompt_refine, context is also passed
        context=rag_context
    )

    if not refined_review:
        print("Warning: Failed to refine the job review with GPT-4o. Returning the GPT-3.5-turbo draft.")
        return f"GPT-3.5-turbo Draft (Refinement Failed):\n{draft_review}"

    print("Review refined successfully.")
    return refined_review


def retrieve_relevant_chunk(query: str, vectorizer, tfidf_matrix, documents: list[str], top_n: int = 1) -> str:
    """
    Retrieves the most relevant document chunk(s) based on cosine similarity.

    Args:
        query: The user's query string.
        vectorizer: The fitted TfidfVectorizer.
        tfidf_matrix: The TF-IDF matrix of the documents.
        documents: The original list of document contents.
        top_n: The number of top similar documents to return.

    Returns:
        The content of the top_n most similar document(s).
        Returns an empty string if no relevant chunk is found or an error occurs.
    """
    if not query:
        print("Warning: Empty query provided.")
        return ""
    if vectorizer is None or tfidf_matrix is None:
        print("Warning: Vectorizer or TF-IDF matrix is not initialized.")
        return ""
    if not documents:
        print("Warning: No documents available to retrieve from.")
        return ""
    if top_n <= 0:
        print("Warning: top_n must be a positive integer.")
        return ""

    try:
        # Transform the query using the same vectorizer
        query_vector = vectorizer.transform([query])

        # Calculate cosine similarity between the query vector and document vectors
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Get the indices of the top_n most similar documents
        # Adding check for cases where fewer documents than top_n are available
        num_documents = tfidf_matrix.shape[0]
        actual_top_n = min(top_n, num_documents)

        if actual_top_n == 0:
            return "No documents available for comparison."

        # Get the indices of the top_n most similar documents
        # If cosine_similarities are all zero, argsort might behave unexpectedly,
        # so we handle cases where no similarity is found.
        if np.all(cosine_similarities == 0):
            return "No relevant document found for the query."

        relevant_doc_indices = np.argsort(cosine_similarities)[-actual_top_n:][::-1]


        # Retrieve the content of the most similar documents
        # Concatenate the content if top_n > 1
        relevant_chunks = [documents[i] for i in relevant_doc_indices if cosine_similarities[i] > 0]

        if not relevant_chunks:
            return "No relevant document found for the query."

        return "\n---\n".join(relevant_chunks)
    except Exception as e:
        print(f"Error retrieving relevant chunk: {e}")
        return "Error occurred during retrieval."

# --- Main Application Logic ---
def main():
    """
    Main function to run the RAG and LLM job review generation process.
    """
    # Check for OpenAI API Key
    if not OPENAI_API_KEY_VALUE:
        print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    PROJECT_DATA_DIR = "project_data"
    documents = load_documents(PROJECT_DATA_DIR)

    vectorizer = None
    tfidf_matrix = None
    rag_context = ""

    if not documents:
        print(f"No documents found in {PROJECT_DATA_DIR}. RAG will be skipped.")
    else:
        print(f"Loaded {len(documents)} document(s) from {PROJECT_DATA_DIR}.")
        vectorizer, tfidf_matrix = initialize_vectorizer(documents)
        if not vectorizer or tfidf_matrix is None: # Check if tfidf_matrix is None explicitly
            print("Failed to initialize TfidfVectorizer. RAG will be skipped.")
            # Ensure documents list is not passed to retrieve_relevant_chunk if vectorizer failed
            documents = []


    # Define a sample job query
    job_query = "Generate a review for a senior software engineer specializing in Python and cloud technologies, requiring at least 5 years of experience and strong communication skills."

    if documents and vectorizer and tfidf_matrix is not None: # Check tfidf_matrix again
        print(f"\nRetrieving context for query: \"{job_query[:50]}...\"")
        rag_context = retrieve_relevant_chunk(job_query, vectorizer, tfidf_matrix, documents, top_n=1)
        if not rag_context or "No relevant document found" in rag_context or "Error occurred" in rag_context:
            print(f"No relevant context found for the query, or an error occurred: {rag_context}")
            rag_context = "" # Ensure rag_context is empty if no useful info found
        else:
            print(f"Retrieved RAG context (first 100 chars): {rag_context[:100]}...")
    else:
        if documents : # Only print this if we had documents but vectorizer failed
             print("Skipping RAG due to vectorizer initialization failure or no documents.")


    print("\n--- Generating Job Review ---")
    # Call the LLM to generate the job review
    job_review = generate_job_review_llm(job_query, rag_context)

    if job_review:
        print("\n--- Generated Job Review ---")
        print(job_review)
    else:
        print("\nFailed to generate job review. The function returned no output or an error message was printed during the process.")

if __name__ == "__main__":
    main()
