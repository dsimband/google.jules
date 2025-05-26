import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import io
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Mock RAG loading functions from app.py BEFORE dashboard_app is imported by tests
# These mocks will apply to all tests in this class.
# dashboard_app.py calls load_documents and initialize_vectorizer at its global scope.
# So, these need to be patched broadly.
@patch('app.load_documents', MagicMock(return_value=[("doc1.txt", "Sample document content for RAG.")]))
@patch('app.initialize_vectorizer', MagicMock(return_value=(MagicMock(), MagicMock()))) # (mock_vectorizer, mock_tfidf_matrix)
@patch('dashboard_app.API_KEY_AVAILABLE', True) # Assume API key is available for most tests
class TestDashboardApp(unittest.TestCase):

    original_openai_api_key = None
    
    @classmethod
    def setUpClass(cls):
        # This print is for sanity check during test development/debugging
        # print("setUpClass: Initializing mocks for RAG and API Key availability.", file=sys.stderr)
        
        # Store original API key if it exists, and set a dummy one for dashboard_app global scope
        cls.original_openai_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "dummy_test_key_for_dashboard_app_load"

        # The patches on the class should handle load_documents, initialize_vectorizer, and API_KEY_AVAILABLE.
        # Now, we can import dashboard_app globally within the test module if needed,
        # but it's often safer to import it within test methods or setUp if there are complex
        # module-level interactions. For this case, since patches are class-level,
        # importing dashboard_app module here or in individual tests should be fine.
        # We will import specific items from it within tests.

    @classmethod
    def tearDownClass(cls):
        # Restore original API key
        if cls.original_openai_api_key is None:
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
        else:
            os.environ["OPENAI_API_KEY"] = cls.original_openai_api_key
        # print("tearDownClass: Restored original OPENAI_API_KEY.", file=sys.stderr)


    def setUp(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        # Suppress print statements from dashboard_app during its import/loading phase for cleaner test output
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()


    def tearDown(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        # Remove project root from sys.path if it was added by this test file's import
        # This is a bit tricky as it's added at module level.
        # A robust way is to ensure it's removed only if no other test needs it.
        # For now, given it's standard practice for project structure, leave it.
        # if str(project_root) in sys.path:
        #     sys.path.remove(str(project_root))

    def find_component_by_id(self, component, target_id):
        """Helper to find a Dash component by ID in a layout structure."""
        if hasattr(component, 'id') and component.id == target_id:
            return component
        if hasattr(component, 'children'):
            if isinstance(component.children, list):
                for child in component.children:
                    found = self.find_component_by_id(child, target_id)
                    if found:
                        return found
            else: # Single child
                return self.find_component_by_id(component.children, target_id)
        return None

    def test_app_initialization_and_layout(self, mock_api_key_available, mock_init_vectorizer, mock_load_docs):
        # mock_api_key_available, mock_init_vectorizer, mock_load_docs are passed by class decorators
        from dashboard_app import app as dash_app_instance # Import here
        
        self.assertIsNotNone(dash_app_instance.layout, "Dash app layout should not be None.")
        
        # Check for key components by ID
        self.assertIsNotNone(self.find_component_by_id(dash_app_instance.layout, 'job-query-input'), "job-query-input component not found.")
        self.assertIsNotNone(self.find_component_by_id(dash_app_instance.layout, 'submit-button'), "submit-button component not found.")
        self.assertIsNotNone(self.find_component_by_id(dash_app_instance.layout, 'job-review-output'), "job-review-output component not found.")

    @patch('app.generate_job_review_llm')
    @patch('app.retrieve_relevant_chunk')
    def test_generate_review_callback_success(self, mock_retrieve_chunk, mock_generate_llm, mock_api_key_available, mock_init_vectorizer, mock_load_docs):
        # mock_api_key_available, mock_init_vectorizer, mock_load_docs are from class decorators
        # mock_retrieve_chunk, mock_generate_llm are from method decorators

        from dashboard_app import update_job_review # Import callback here
        import dashboard_app as da # For accessing mocked global RAG components if needed

        mock_generate_llm.return_value = "Mocked LLM Review"
        mock_retrieve_chunk.return_value = "Mocked RAG Context"
        
        # Simulate that RAG components were "loaded" by the global mocks
        # These are what the callback will use from dashboard_app's global scope
        # The class-level mocks ensure these are populated when dashboard_app is imported/used
        
        test_query = "Test job query for senior dev"
        result = update_job_review(n_clicks=1, job_query=test_query)
        
        self.assertEqual(result, "Mocked LLM Review  \n") # Callback adds "  \n"
        
        # Verify retrieve_relevant_chunk was called correctly
        # It uses dashboard_app.VECTORIZER, TFIDF_MATRIX, DOCUMENTS which are set up by class-level mocks
        mock_retrieve_chunk.assert_called_once()
        call_args_retrieve = mock_retrieve_chunk.call_args[0]
        self.assertEqual(call_args_retrieve[0], test_query)
        self.assertIsNotNone(call_args_retrieve[1]) # mock_vectorizer
        self.assertIsNotNone(call_args_retrieve[2]) # mock_tfidf_matrix
        self.assertIsNotNone(call_args_retrieve[3]) # mock_documents (content from mock_load_docs)
        self.assertEqual(call_args_retrieve[4], 1)  # top_n

        mock_generate_llm.assert_called_once_with(test_query, "Mocked RAG Context")

    @patch('dashboard_app.API_KEY_AVAILABLE', False) # Override class-level mock for this test
    def test_generate_review_callback_no_api_key(self, mock_api_key_false, mock_api_key_available, mock_init_vectorizer, mock_load_docs):
        # mock_api_key_false is from method decorator, others from class
        from dashboard_app import update_job_review

        result = update_job_review(n_clicks=1, job_query="Test query")
        self.assertEqual(result, "Error: OpenAI API key is not configured. Please set it in your .env file or environment.")

    def test_generate_review_callback_empty_query(self, mock_api_key_available, mock_init_vectorizer, mock_load_docs):
        # Mocks from class decorators
        from dashboard_app import update_job_review
        
        result = update_job_review(n_clicks=1, job_query="   ") # Empty or whitespace query
        self.assertEqual(result, "Please enter job details in the text area.")

if __name__ == '__main__':
    # Need to explicitly set OPENAI_API_KEY here if tests are run directly
    # and dashboard_app is imported at module level without mocks being active yet.
    # However, with setUpClass mocks, this should be okay.
    unittest.main()
