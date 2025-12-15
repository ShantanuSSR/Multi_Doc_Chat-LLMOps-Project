"""
Shared fixtures and utilities for all tests.
"""
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock, patch
import pytest

# Export LCELCompatibleMock for use in test files
__all__ = ['LCELCompatibleMock']

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain.schema import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable


class LCELCompatibleMock:
    """Mock that supports LangChain LCEL pipe operator."""
    def __init__(self, return_value=None):
        self.return_value = return_value
        self._invoke_called = False
    
    def __or__(self, other):
        """Support pipe operator for LCEL chaining."""
        return LCELCompatibleMock(return_value=self.return_value)
    
    def __ror__(self, other):
        """Support reverse pipe operator for LCEL chaining."""
        return LCELCompatibleMock(return_value=self.return_value)
    
    def invoke(self, input_data, config=None):
        """Invoke method for Runnable interface."""
        self._invoke_called = True
        if callable(self.return_value):
            return self.return_value(input_data)
        return self.return_value
    
    def __call__(self, *args, **kwargs):
        """Make it callable."""
        if callable(self.return_value):
            return self.return_value(*args, **kwargs)
        return self.return_value


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for API keys."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test_google_key_12345")
    monkeypatch.setenv("GROQ_API_KEY", "test_groq_key_12345")
    monkeypatch.setenv("ENV", "local")
    monkeypatch.setenv("LLM_PROVIDER", "google")
    yield
    # Cleanup handled by monkeypatch


@pytest.fixture
def mock_config():
    """Mock configuration dictionary."""
    return {
        "embedding_model": {
            "model_name": "models/text-embedding-004"
        },
        "llm": {
            "google": {
                "provider": "google",
                "model_name": "gemini-2.0-flash",
                "temperature": 0,
                "max_output_tokens": 2048
            },
            "groq": {
                "provider": "groq",
                "model_name": "openai/gpt-oss-20b",
                "temperature": 0,
                "max_output_tokens": 2048
            }
        }
    }


@pytest.fixture
def mock_llm():
    """Mock LLM object."""
    llm = Mock()
    llm.invoke = Mock(return_value=Mock(content="Test response from LLM"))
    llm.generate = Mock(return_value=Mock(generations=[[Mock(text="Test response")]]))
    return llm


@pytest.fixture
def mock_embeddings():
    """Mock embeddings model."""
    embeddings = Mock()
    embeddings.embed_query = Mock(return_value=[0.1] * 768)
    embeddings.embed_documents = Mock(return_value=[[0.1] * 768] * 3)
    return embeddings


@pytest.fixture
def mock_retriever():
    """Mock retriever object."""
    retriever = Mock()
    mock_docs = [
        Document(page_content="Document 1 content", metadata={"source": "doc1.pdf"}),
        Document(page_content="Document 2 content", metadata={"source": "doc2.pdf"}),
        Document(page_content="Document 3 content", metadata={"source": "doc3.pdf"}),
    ]
    retriever.invoke = Mock(return_value=mock_docs)
    retriever.get_relevant_documents = Mock(return_value=mock_docs)
    return retriever


@pytest.fixture
def sample_documents():
    """Sample Document objects for testing."""
    return [
        Document(
            page_content="This is the first document about machine learning.",
            metadata={"source": "doc1.pdf", "page": 1}
        ),
        Document(
            page_content="This is the second document about natural language processing.",
            metadata={"source": "doc2.pdf", "page": 1}
        ),
        Document(
            page_content="This is the third document about deep learning.",
            metadata={"source": "doc3.pdf", "page": 1}
        ),
    ]


@pytest.fixture
def sample_chat_history():
    """Sample chat history for testing."""
    return [
        HumanMessage(content="What is machine learning?"),
        AIMessage(content="Machine learning is a subset of AI."),
    ]


@pytest.fixture
def mock_file_upload():
    """Mock file upload object."""
    class MockUploadFile:
        def __init__(self, filename: str, content: bytes):
            self.filename = filename
            self._content = content
            self.file = Mock()
            self.file.read = Mock(return_value=content)
            self.file.seek = Mock()
        
        def read(self):
            return self._content
        
        def getbuffer(self):
            return self._content
    
    return MockUploadFile("test_document.pdf", b"Mock PDF content")


@pytest.fixture
def mock_faiss_index(temp_dir):
    """Create a mock FAISS index directory structure."""
    index_dir = temp_dir / "faiss_index" / "test_session"
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock index files
    (index_dir / "index.faiss").touch()
    (index_dir / "index.pkl").touch()
    
    # Create metadata file
    meta = {"rows": {}}
    (index_dir / "ingested_meta.json").write_text(json.dumps(meta))
    
    return index_dir


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file for testing."""
    text_file = temp_dir / "sample.txt"
    text_file.write_text("This is a sample text file for testing document ingestion.")
    return text_file


@pytest.fixture
def mock_lcel_prompt():
    """Create an LCEL-compatible mock prompt."""
    return LCELCompatibleMock()


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """Reset environment variables before each test."""
    # Clear any existing env vars that might interfere
    for key in ["apikeyliveclass", "CONFIG_PATH"]:
        monkeypatch.delenv(key, raising=False)
    yield
