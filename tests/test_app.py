import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys
import io
import shutil
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Now import from app
# We need to be careful if app.OPENAI_API_KEY_VALUE is used by main at module level
# For tests, it's better if main() itself checks the env var or if we can mock the global.
from app import main
import app as current_app_module # To modify app.OPENAI_API_KEY_VALUE for one test

class TestAppFlow(unittest.TestCase):

    def setUp(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.captured_stdout = io.StringIO()
        sys.stderr = self.captured_stderr = io.StringIO()

        self.temp_data_dir_name = "temp_integration_project_data"
        self.temp_data_dir = current_dir / self.temp_data_dir_name
        self._setup_project_data_dir() # Creates an empty dir

        # Store original API key state
        self.original_env_api_key = os.environ.get("OPENAI_API_KEY")
        self.original_app_module_api_key_value = current_app_module.OPENAI_API_KEY_VALUE
        
        # .env file path for tests (project root)
        self.dotenv_path = project_root / ".env"
        self.original_dotenv_exists = self.dotenv_path.exists()
        self.original_dotenv_content = ""
        if self.original_dotenv_exists:
            with open(self.dotenv_path, "r") as f:
                self.original_dotenv_content = f.read()
        
        # Ensure a clean state for os.environ regarding the key for each test, if needed by test.
        # Tests themselves will now be responsible for setting os.environ["OPENAI_API_KEY"]
        # or current_app_module.OPENAI_API_KEY_VALUE as per their specific scenario.
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        current_app_module.OPENAI_API_KEY_VALUE = None


    def _create_env_file(self, content: str):
        # Clean up if a .env file from a previous test (or user) exists
        if self.dotenv_path.exists():
            self.dotenv_path.unlink()
        with open(self.dotenv_path, "w") as f:
            f.write(content)

    def _remove_env_file(self):
        if self.dotenv_path.exists():
            self.dotenv_path.unlink()
        # Restore original .env if it existed
        if self.original_dotenv_exists:
            with open(self.dotenv_path, "w") as f:
                f.write(self.original_dotenv_content)

    def _setup_project_data_dir(self, docs_content: dict = None):
        if self.temp_data_dir.exists():
            shutil.rmtree(self.temp_data_dir)
        os.makedirs(self.temp_data_dir, exist_ok=True)
        
        # In app.py, PROJECT_DATA_DIR is hardcoded relative to app.py if app.py is in root
        # For testing, we need main() to look into our self.temp_data_dir
        # We achieve this by patching app.PROJECT_DATA_DIR
        self.project_data_dir_patcher = patch('app.PROJECT_DATA_DIR', str(self.temp_data_dir))
        self.mock_project_data_dir = self.project_data_dir_patcher.start()

        if docs_content:
            for filename, content in docs_content.items():
                with open(self.temp_data_dir / filename, "w") as f:
                    f.write(content)

    def tearDown(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        if self.temp_data_dir.exists():
            shutil.rmtree(self.temp_data_dir)
        
        self.project_data_dir_patcher.stop()

        # Restore original API key
        if self.original_env_api_key is None:
            if "OPENAI_API_KEY" in os.environ: # if test set it
                del os.environ["OPENAI_API_KEY"]
        else:
            os.environ["OPENAI_API_KEY"] = self.original_env_api_key
        
        current_app_module.OPENAI_API_KEY_VALUE = self.original_app_module_api_key_value
        self._remove_env_file() # Clean up any test .env and restore original if existed
        
        # Remove project root from sys.path if it was added
        if str(project_root) in sys.path:
            sys.path.remove(str(project_root))


    @patch('app.call_openai_api')
    def test_successful_flow_with_rag(self, mock_call_openai_api: MagicMock):
        self._setup_project_data_dir({"cv_jane_doe.txt": "Jane Doe is a Python expert with 7 years in cloud computing."})
        
        mock_call_openai_api.side_effect = [
            "Draft: Job review for Python expert Jane Doe, cloud.", # GPT-3.5
            "Refined: Awesome Python expert Jane Doe, 7 years cloud experience, needed now!" # GPT-4o
        ]

        main()

        output = self.captured_stdout.getvalue()
        
        self.assertEqual(mock_call_openai_api.call_count, 2)
        
        # Check first call (drafting)
        args_draft, _ = mock_call_openai_api.call_args_list[0]
        self.assertEqual(args_draft[0], "gpt-3.5-turbo")
        self.assertIn("Jane Doe", args_draft[3]) # Check RAG context in call
        self.assertIn("Python expert", args_draft[3])

        # Check second call (refinement)
        args_refine, _ = mock_call_openai_api.call_args_list[1]
        self.assertEqual(args_refine[0], "gpt-4o")
        self.assertIn("Draft: Job review for Python expert Jane Doe, cloud.", args_refine[2]) # Check draft in user_prompt

        self.assertIn("Refined: Awesome Python expert Jane Doe, 7 years cloud experience, needed now!", output)
        self.assertIn("Retrieved RAG context", output)

    @patch('app.sys.exit') # Mock sys.exit to prevent test termination
    @patch('app.retrieve_relevant_chunk', MagicMock(return_value="dummy rag context")) # Mock RAG
    @patch('app.call_openai_api')
    def test_api_key_loaded_from_dotenv_file(self, mock_call_openai_api: MagicMock, mock_sys_exit: MagicMock):
        # Ensure os.environ is clean for this key
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        current_app_module.OPENAI_API_KEY_VALUE = None # Reflect that it's not in env at app import

        self._create_env_file('OPENAI_API_KEY="key_from_test_dotenv"')
        
        mock_call_openai_api.side_effect = ["Draft from .env", "Refined from .env"]
        main()
        
        self.assertEqual(current_app_module.OPENAI_API_KEY_VALUE, "key_from_test_dotenv")
        mock_sys_exit.assert_not_called() # Should not exit if key is found
        self.assertEqual(mock_call_openai_api.call_count, 2)
        output = self.captured_stdout.getvalue()
        self.assertIn("Refined from .env", output)

    @patch('app.sys.exit')
    @patch('app.retrieve_relevant_chunk', MagicMock(return_value="dummy rag context"))
    @patch('app.call_openai_api')
    def test_dotenv_does_not_override_existing_env_variable(self, mock_call_openai_api: MagicMock, mock_sys_exit: MagicMock):
        # Set key in os.environ BEFORE main() and .env creation
        os.environ["OPENAI_API_KEY"] = "key_from_os_environ"
        # This simulates app.py's global OPENAI_API_KEY_VALUE being set at import time
        current_app_module.OPENAI_API_KEY_VALUE = "key_from_os_environ" 

        self._create_env_file('OPENAI_API_KEY="key_from_test_dotenv_should_be_ignored"')
        
        mock_call_openai_api.side_effect = ["Draft from os_environ", "Refined from os_environ"]
        main()

        # python-dotenv by default does NOT override existing environment variables.
        # The app.py logic also updates global OPENAI_API_KEY_VALUE based on os.getenv after load_dotenv.
        self.assertEqual(current_app_module.OPENAI_API_KEY_VALUE, "key_from_os_environ")
        mock_sys_exit.assert_not_called()
        self.assertEqual(mock_call_openai_api.call_count, 2)
        output = self.captured_stdout.getvalue()
        self.assertIn("Refined from os_environ", output)

    @patch('app.sys.exit')
    @patch('app.retrieve_relevant_chunk', MagicMock(return_value="dummy rag context"))
    @patch('app.call_openai_api')
    def test_api_key_loaded_from_dotenv_if_not_in_os_environ(self, mock_call_openai_api: MagicMock, mock_sys_exit: MagicMock):
        # Explicitly ensure OPENAI_API_KEY is not in os.environ
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        # Simulate that the global key was None at import time
        current_app_module.OPENAI_API_KEY_VALUE = None

        self._create_env_file('OPENAI_API_KEY="key_from_dotenv_when_os_empty"')
        
        mock_call_openai_api.side_effect = ["Draft from .env (os empty)", "Refined from .env (os empty)"]
        main()

        self.assertEqual(current_app_module.OPENAI_API_KEY_VALUE, "key_from_dotenv_when_os_empty")
        mock_sys_exit.assert_not_called()
        self.assertEqual(mock_call_openai_api.call_count, 2)
        output = self.captured_stdout.getvalue()
        self.assertIn("Refined from .env (os empty)", output)


    @patch('app.call_openai_api')
    def test_rag_finds_no_relevant_documents(self, mock_call_openai_api: MagicMock):
        # This test might need OPENAI_API_KEY to be set to avoid sys.exit
        # For simplicity, let's ensure it's set (e.g. by .env or os.environ)
        # The focus here is RAG, not API key loading.
        # We can use the .env creation for a default valid key.
        self._create_env_file('OPENAI_API_KEY="dummy_key_for_rag_test"')
        current_app_module.OPENAI_API_KEY_VALUE = None # Ensure it's loaded by main

        self._setup_project_data_dir({"internal_memo.txt": "Company holiday party next week."})
        
        mock_call_openai_api.side_effect = [
            "Draft: Standard software engineer role.",
            "Refined: Standard software engineer role, good benefits."
        ]

        main()
        output = self.captured_stdout.getvalue()

        self.assertEqual(mock_call_openai_api.call_count, 2)
        # Check that RAG context passed to LLM was empty or indicated no useful info
        args_draft, _ = mock_call_openai_api.call_args_list[0]
        # The context passed to call_openai_api includes "Context from retrieved documents:\n{rag_context}"
        # If rag_context itself is "No relevant document found...", it will be part of the full context.
        self.assertIn("No relevant document found for the query.", args_draft[3])
        
        self.assertIn("No relevant context found for the query", output) # Check console message
        self.assertIn("Refined: Standard software engineer role, good benefits.", output)

    @patch('app.call_openai_api')
    def test_no_documents_in_project_data(self, mock_call_openai_api: MagicMock):
        self._create_env_file('OPENAI_API_KEY="dummy_key_for_nodoc_test"')
        current_app_module.OPENAI_API_KEY_VALUE = None

        self._setup_project_data_dir() # Empty
        
        mock_call_openai_api.side_effect = [
            "Draft: Generic developer position.",
            "Refined: Generic developer position, apply today."
        ]

        main()
        output = self.captured_stdout.getvalue()

        self.assertEqual(mock_call_openai_api.call_count, 2)
        self.assertIn(f"No documents found in {str(self.temp_data_dir)}. RAG will be skipped.", output)
        self.assertIn("Refined: Generic developer position, apply today.", output)
        
        args_draft, _ = mock_call_openai_api.call_args_list[0]
        self.assertIn("Context from retrieved documents:\n\nUser Query/Instruction:", args_draft[3]) # Empty RAG context

    @patch('app.sys.exit') # To check if sys.exit is called
    def test_missing_openai_api_key(self, mock_sys_exit: MagicMock):
        # Temporarily unset the API key for this test
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        original_app_key_val = current_app_module.OPENAI_API_KEY_VALUE
        current_app_module.OPENAI_API_KEY_VALUE = None
        
        # Patch the global OPENAI_API_KEY_VALUE directly in the app module
        with patch('app.OPENAI_API_KEY_VALUE', None):
            main()
        
        mock_sys_exit.assert_called_once_with(1)
        output = self.captured_stdout.getvalue() # main() prints to stdout
        self.assertIn("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.", output)

        # Restore
        if original_key is not None:
            os.environ["OPENAI_API_KEY"] = original_key
        current_app_module.OPENAI_API_KEY_VALUE = original_app_key_val


    @patch('app.call_openai_api')
    def test_llm_draft_fails(self, mock_call_openai_api: MagicMock):
        self._create_env_file('OPENAI_API_KEY="dummy_key_for_draftfail_test"')
        current_app_module.OPENAI_API_KEY_VALUE = None

        self._setup_project_data_dir() # Empty
        
        mock_call_openai_api.return_value = "" # Simulate first call (draft) failing

        main()
        output = self.captured_stdout.getvalue()

        mock_call_openai_api.assert_called_once() # Only draft call should happen
        self.assertEqual(mock_call_openai_api.call_args_list[0][0][0], "gpt-3.5-turbo")
        self.assertIn("Error: Failed to generate the initial draft for the job review.", output)
        self.assertNotIn("Refined:", output) # No refined output

    @patch('app.call_openai_api')
    def test_llm_refinement_fails(self, mock_call_openai_api: MagicMock):
        self._create_env_file('OPENAI_API_KEY="dummy_key_for_refinefail_test"')
        current_app_module.OPENAI_API_KEY_VALUE = None

        self._setup_project_data_dir() # Empty
        
        mock_call_openai_api.side_effect = [
            "Draft: A basic software job.", # GPT-3.5 succeeds
            ""                             # GPT-4o fails
        ]

        main()
        output = self.captured_stdout.getvalue()

        self.assertEqual(mock_call_openai_api.call_count, 2)
        self.assertIn("GPT-3.5-turbo Draft (Refinement Failed):\nDraft: A basic software job.", output)

if __name__ == '__main__':
    unittest.main()
