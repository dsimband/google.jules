import unittest
import os
import shutil
import sys
from pathlib import Path

# Temporarily add the parent directory (project root) to sys.path
# to allow importing 'app' from the 'tests' directory.
# This is a common way to handle imports for testing when the test runner
# is invoked from the project root.
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from app import load_documents, initialize_vectorizer, retrieve_relevant_chunk
from sklearn.feature_extraction.text import TfidfVectorizer

# scipy.sparse.csr_matrix is used for type checking, ensure scipy is installed
# or this import will fail. We noted earlier it's not installed, will add to requirements.
try:
    from scipy.sparse import csr_matrix
except ImportError:
    # This allows the file to be parsed, but tests requiring csr_matrix will fail
    # if scipy is not actually installed during test execution.
    csr_matrix = None


class TestRAGComponents(unittest.TestCase):

    def setUp(self):
        """Set up temporary directory and files for testing."""
        self.temp_dir = Path("tests") / "temp_project_data"
        self.empty_dir = Path("tests") / "temp_empty_dir"
        
        # Ensure directories are clean before each test
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        if self.empty_dir.exists():
            shutil.rmtree(self.empty_dir)
            
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.empty_dir, exist_ok=True)

        self.doc1_content = "This is a test document about Python."
        self.doc2_content = "Markdown file discussing software engineering."
        self.empty_content = ""

        with open(self.temp_dir / "doc1.txt", "w") as f:
            f.write(self.doc1_content)
        with open(self.temp_dir / "doc2.md", "w") as f:
            f.write(self.doc2_content)
        with open(self.temp_dir / "invalid.pdf", "w") as f:
            f.write("This should be ignored by load_documents.")
        with open(self.temp_dir / "empty.txt", "w") as f:
            f.write(self.empty_content)
            
        # For retrieve_relevant_chunk
        self.sample_docs_for_retrieval = [
            "The quick brown fox", 
            "jumps over the lazy dog", 
            "Python programming is great"
        ]
        self.vectorizer_retrieval, self.tfidf_matrix_retrieval = initialize_vectorizer(self.sample_docs_for_retrieval)


    def tearDown(self):
        """Clean up temporary directory and files after tests."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        if self.empty_dir.exists():
            shutil.rmtree(self.empty_dir)
        # Remove the project root from sys.path if it was added
        if str(project_root) in sys.path:
            sys.path.remove(str(project_root))

    def test_load_documents(self):
        """Test loading documents from various scenarios."""
        # Test with the populated temporary directory
        loaded_docs = load_documents(str(self.temp_dir))
        self.assertEqual(len(loaded_docs), 3, "Should load .txt and .md files, including empty ones.")
        
        # Check for content, order might vary depending on glob results
        self.assertTrue(self.doc1_content in loaded_docs)
        self.assertTrue(self.doc2_content in loaded_docs)
        self.assertTrue(self.empty_content in loaded_docs)

        # Test with a non-existent directory
        non_existent_dir = self.temp_dir.parent / "non_existent_dir_for_test"
        self.assertEqual(load_documents(str(non_existent_dir)), [], "Should return empty list for non-existent directory.")

        # Test with an empty directory
        self.assertEqual(load_documents(str(self.empty_dir)), [], "Should return empty list for an empty directory.")

    def test_initialize_vectorizer(self):
        """Test TfidfVectorizer initialization."""
        if csr_matrix is None:
            self.skipTest("scipy.sparse.csr_matrix not available, skipping this test.")

        sample_docs = ["Python is fun", "Java is versatile", "Python and Java"]
        vectorizer, tfidf_matrix = initialize_vectorizer(sample_docs)

        self.assertIsNotNone(vectorizer, "Vectorizer should not be None.")
        self.assertIsNotNone(tfidf_matrix, "TF-IDF matrix should not be None.")
        self.assertIsInstance(vectorizer, TfidfVectorizer, "Should return a TfidfVectorizer instance.")
        self.assertIsInstance(tfidf_matrix, csr_matrix, "Should return a scipy sparse csr_matrix.")
        
        # Vocabulary: {'python', 'is', 'fun', 'java', 'versatile', 'and'} -> 6 unique terms
        expected_shape = (len(sample_docs), 6) 
        self.assertEqual(tfidf_matrix.shape, expected_shape, f"TF-IDF matrix shape should be {expected_shape}.")

        # Test with empty list
        vectorizer_empty, tfidf_matrix_empty = initialize_vectorizer([])
        self.assertIsNone(vectorizer_empty, "Vectorizer should be None for empty input.")
        self.assertIsNone(tfidf_matrix_empty, "TF-IDF matrix should be None for empty input.")

    def test_retrieve_relevant_chunk(self):
        """Test retrieving relevant document chunks."""
        if self.vectorizer_retrieval is None or self.tfidf_matrix_retrieval is None:
            self.skipTest("Vectorizer or TF-IDF matrix for retrieval tests is None, possibly due to empty sample_docs_for_retrieval.")

        # Test query matching one document
        query_python = "Python"
        expected_python_doc = "Python programming is great"
        retrieved_python = retrieve_relevant_chunk(query_python, self.vectorizer_retrieval, self.tfidf_matrix_retrieval, self.sample_docs_for_retrieval)
        self.assertEqual(retrieved_python, expected_python_doc)

        # Test query matching another document
        query_fox = "fox"
        expected_fox_doc = "The quick brown fox"
        retrieved_fox = retrieve_relevant_chunk(query_fox, self.vectorizer_retrieval, self.tfidf_matrix_retrieval, self.sample_docs_for_retrieval)
        self.assertEqual(retrieved_fox, expected_fox_doc)

        # Test with a query that doesn't match any document well
        query_nomatch = "lorem ipsum dolor sit amet"
        # Based on app.py, it returns "No relevant document found for the query."
        expected_nomatch_result = "No relevant document found for the query."
        retrieved_nomatch = retrieve_relevant_chunk(query_nomatch, self.vectorizer_retrieval, self.tfidf_matrix_retrieval, self.sample_docs_for_retrieval)
        self.assertEqual(retrieved_nomatch, expected_nomatch_result)

        # Test with top_n=2
        query_quick_dog = "quick dog" # Should match "The quick brown fox" and "jumps over the lazy dog"
        # Order of results from cosine similarity with "quick dog" might depend on TF-IDF specifics.
        # "The quick brown fox" (has "quick")
        # "jumps over the lazy dog" (has "dog")
        # We expect both, joined by "\n---\n"
        retrieved_top2 = retrieve_relevant_chunk(query_quick_dog, self.vectorizer_retrieval, self.tfidf_matrix_retrieval, self.sample_docs_for_retrieval, top_n=2)
        self.assertIn("The quick brown fox", retrieved_top2)
        self.assertIn("jumps over the lazy dog", retrieved_top2)
        self.assertIn("\n---\n", retrieved_top2) # Check for separator

        # Test with empty documents list (but valid vectorizer/matrix from other docs)
        # This scenario might be a bit artificial, as vectorizer is usually fit on the docs you search.
        # However, the function should handle it.
        retrieved_empty_docs = retrieve_relevant_chunk("query", self.vectorizer_retrieval, self.tfidf_matrix_retrieval, [])
        self.assertEqual(retrieved_empty_docs, "", "Should return empty string or error if documents list is empty.")
        
        # Test with None vectorizer/matrix
        retrieved_none_vectorizer = retrieve_relevant_chunk("query", None, None, self.sample_docs_for_retrieval)
        self.assertEqual(retrieved_none_vectorizer, "", "Should return empty string if vectorizer/matrix is None.")

if __name__ == '__main__':
    unittest.main()
