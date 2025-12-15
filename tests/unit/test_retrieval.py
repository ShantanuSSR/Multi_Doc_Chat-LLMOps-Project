"""
Unit tests for ConversationalRAG class.
"""
import sys
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from google.api_core import exceptions as google_exceptions

# Import LCELCompatibleMock from conftest
# Add parent directory to path to import conftest
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from conftest import LCELCompatibleMock

from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from multi_doc_chat.exception.custom_exception import DocumentPortalException


class TestConversationalRAG:
    """Test ConversationalRAG class."""

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_initialization_success(self, mock_prompt_registry, mock_model_loader_class, mock_llm, mock_env_vars):
        """Test successful ConversationalRAG initialization."""
        mock_model_loader = Mock()
        mock_model_loader.load_llm.return_value = mock_llm
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        rag = ConversationalRAG(session_id="test_session")
        
        assert rag.session_id == "test_session"
        assert rag.llm == mock_llm
        assert rag.retriever is None
        assert rag.chain is None

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_initialization_with_retriever(self, mock_prompt_registry, mock_model_loader_class, mock_llm, mock_retriever, mock_env_vars):
        """Test initialization with retriever builds chain immediately."""
        mock_model_loader = Mock()
        mock_model_loader.load_llm.return_value = mock_llm
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        rag = ConversationalRAG(session_id="test_session", retriever=mock_retriever)
        
        assert rag.retriever == mock_retriever
        assert rag.chain is not None

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.FAISS")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_load_retriever_from_faiss_success(
        self, mock_prompt_registry, mock_faiss_class, mock_model_loader_class, 
        mock_llm, mock_embeddings, mock_env_vars, temp_dir
    ):
        """Test loading retriever from FAISS index."""
        # Setup mocks
        mock_model_loader = Mock()
        mock_model_loader.load_llm.return_value = mock_llm
        mock_model_loader.load_embeddings.return_value = mock_embeddings
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        # Create mock FAISS index directory
        index_dir = temp_dir / "faiss_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss_class.load_local.return_value = mock_vectorstore
        
        rag = ConversationalRAG(session_id="test_session")
        retriever = rag.load_retriever_from_faiss(str(index_dir), k=5)
        
        assert retriever == mock_retriever
        assert rag.retriever == mock_retriever
        assert rag.chain is not None
        mock_faiss_class.load_local.assert_called_once()

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_load_retriever_from_faiss_invalid_path(
        self, mock_prompt_registry, mock_model_loader_class, mock_llm, mock_env_vars
    ):
        """Test loading retriever with invalid path raises error."""
        mock_model_loader = Mock()
        mock_model_loader.load_llm.return_value = mock_llm
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        rag = ConversationalRAG(session_id="test_session")
        
        with pytest.raises(DocumentPortalException) as exc_info:
            rag.load_retriever_from_faiss("/nonexistent/path")
        
        assert "Loading error" in str(exc_info.value)

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_invoke_success(self, mock_prompt_registry, mock_model_loader_class, mock_llm, mock_retriever, mock_env_vars):
        """Test successful chain invocation."""
        mock_model_loader = Mock()
        mock_model_loader.load_llm.return_value = mock_llm
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        # Mock chain
        mock_chain = Mock()
        mock_chain.invoke.return_value = "This is a test answer."
        
        rag = ConversationalRAG(session_id="test_session", retriever=mock_retriever)
        rag.chain = mock_chain
        
        result = rag.invoke("What is machine learning?", chat_history=[])
        
        assert result == "This is a test answer."
        mock_chain.invoke.assert_called_once()

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_invoke_with_fallback_on_quota_exhaustion(
        self, mock_prompt_registry, mock_model_loader_class, mock_llm, mock_retriever, mock_env_vars
    ):
        """Test that invoke falls back to Groq when Google quota is exhausted."""
        mock_groq_llm = Mock()
        mock_model_loader = Mock()
        # First call returns initial LLM, second call (during fallback) returns Groq LLM
        mock_model_loader.load_llm.side_effect = [mock_llm, mock_groq_llm]
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        # Mock chain that raises ResourceExhausted first, then succeeds after fallback
        original_build_chain = None
        call_count = [0]
        
        def invoke_side_effect(payload):
            call_count[0] += 1
            if call_count[0] == 1:
                raise google_exceptions.ResourceExhausted("Quota exceeded")
            return "Fallback answer from Groq"
        
        mock_chain = Mock()
        mock_chain.invoke.side_effect = invoke_side_effect
        
        rag = ConversationalRAG(session_id="test_session", retriever=mock_retriever)
        original_build_chain = rag._build_lcel_chain
        
        def mock_build_chain():
            """Mock _build_lcel_chain to set a mock chain after rebuild."""
            original_build_chain()
            # After chain is rebuilt, replace it with a mock that returns expected value
            mock_chain_after_fallback = Mock()
            mock_chain_after_fallback.invoke.return_value = "Fallback answer from Groq"
            rag.chain = mock_chain_after_fallback
        
        rag.chain = mock_chain
        rag._build_lcel_chain = mock_build_chain
        
        result = rag.invoke("What is machine learning?")
        
        assert result == "Fallback answer from Groq"
        # Should have called load_llm at least twice: once during init, once during fallback
        # Verify fallback happened by checking call count
        assert mock_model_loader.load_llm.call_count >= 2, \
            f"Expected load_llm to be called at least twice (init + fallback), got {mock_model_loader.load_llm.call_count}"
        # Check if any call had provider_override="groq" 
        # call_args_list items are call objects with .args and .kwargs, or can be indexed as (args, kwargs)
        has_groq_call = any(
            (hasattr(call, 'kwargs') and call.kwargs.get("provider_override") == "groq") or
            (isinstance(call, tuple) and len(call) >= 2 and call[1].get("provider_override") == "groq") or
            "groq" in str(call).lower()
            for call in mock_model_loader.load_llm.call_args_list
        )
        assert has_groq_call, \
            f"Expected load_llm to be called with provider_override='groq'. Calls: {mock_model_loader.load_llm.call_args_list}"

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_invoke_without_chain_raises_error(
        self, mock_prompt_registry, mock_model_loader_class, mock_llm, mock_env_vars
    ):
        """Test that invoking without chain raises error."""
        mock_model_loader = Mock()
        mock_model_loader.load_llm.return_value = mock_llm
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        rag = ConversationalRAG(session_id="test_session")
        
        with pytest.raises(DocumentPortalException) as exc_info:
            rag.invoke("Test question")
        
        assert "not initialized" in str(exc_info.value)

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_format_docs(self, mock_prompt_registry, mock_model_loader_class, mock_llm, sample_documents, mock_env_vars):
        """Test document formatting."""
        mock_model_loader = Mock()
        mock_model_loader.load_llm.return_value = mock_llm
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        formatted = ConversationalRAG._format_docs(sample_documents)
        
        assert "machine learning" in formatted
        assert "natural language processing" in formatted
        assert "deep learning" in formatted

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_build_lcel_chain_without_retriever_raises_error(
        self, mock_prompt_registry, mock_model_loader_class, mock_llm, mock_env_vars
    ):
        """Test that building chain without retriever raises error."""
        mock_model_loader = Mock()
        mock_model_loader.load_llm.return_value = mock_llm
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        rag = ConversationalRAG(session_id="test_session")
        
        with pytest.raises(DocumentPortalException) as exc_info:
            rag._build_lcel_chain()
        
        assert "No retriever set" in str(exc_info.value)
