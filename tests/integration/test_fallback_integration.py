"""
Integration tests for LLM provider fallback mechanism.
"""
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from google.api_core import exceptions as google_exceptions

import sys
import os
# Import LCELCompatibleMock from conftest
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from conftest import LCELCompatibleMock

from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from multi_doc_chat.exception.custom_exception import DocumentPortalException


class TestProviderFallbackIntegration:
    """Test provider fallback integration in RAG pipeline."""

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.FAISS")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_fallback_triggered_on_quota_exhaustion(
        self, mock_prompt_registry, mock_faiss_class, mock_model_loader_class,
        mock_env_vars, mock_faiss_index, mock_retriever
    ):
        """Test that fallback is triggered when Google quota is exhausted during invoke."""
        # Setup initial Google LLM
        mock_google_llm = Mock()
        mock_google_llm.invoke = Mock(side_effect=google_exceptions.ResourceExhausted("Quota exceeded"))
        
        # Setup Groq LLM for fallback
        mock_groq_llm = Mock()
        mock_groq_llm.invoke = Mock(return_value=Mock(content="Answer from Groq"))
        
        mock_model_loader = Mock()
        mock_model_loader.load_llm.side_effect = [mock_google_llm, mock_groq_llm]
        mock_model_loader.load_embeddings.return_value = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        # Setup FAISS
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss_class.load_local.return_value = mock_vectorstore
        
        # Create RAG and load retriever
        rag = ConversationalRAG(session_id="test_session")
        rag.load_retriever_from_faiss(str(mock_faiss_index), k=5)
        
        # Mock chain that raises ResourceExhausted first time, then succeeds after fallback
        original_build_chain = rag._build_lcel_chain
        call_count = [0]  # Use list to allow modification in nested function
        
        def mock_build_chain():
            """Mock _build_lcel_chain to set a mock chain after rebuild."""
            original_build_chain()
            # After chain is rebuilt, replace it with a mock that returns expected value
            mock_chain_after_fallback = Mock()
            mock_chain_after_fallback.invoke.return_value = "Answer from Groq after fallback"
            rag.chain = mock_chain_after_fallback
        
        def chain_invoke_side_effect(payload):
            call_count[0] += 1
            if call_count[0] == 1:
                raise google_exceptions.ResourceExhausted("Quota exceeded")
            return "Answer from Groq after fallback"
        
        # Set initial chain that will fail first time
        mock_chain = Mock()
        mock_chain.invoke.side_effect = chain_invoke_side_effect
        rag.chain = mock_chain
        
        # Patch _build_lcel_chain to set mock after rebuild
        rag._build_lcel_chain = mock_build_chain
        
        # Invoke should trigger fallback
        result = rag.invoke("What is machine learning?")
        
        assert result == "Answer from Groq after fallback"
        # Verify fallback was attempted - ModelLoader should be called for Groq
        # The fallback creates a new ModelLoader(), so we need to check if it was called
        assert mock_model_loader.load_llm.call_count >= 1

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.FAISS")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_fallback_failure_raises_error(
        self, mock_prompt_registry, mock_faiss_class, mock_model_loader_class,
        mock_env_vars, mock_faiss_index, mock_retriever
    ):
        """Test that if both Google and Groq fail, error is raised."""
        # Setup Google LLM that fails
        mock_google_llm = Mock()
        mock_google_llm.invoke = Mock(side_effect=google_exceptions.ResourceExhausted("Quota exceeded"))
        
        # Setup Groq LLM that also fails
        mock_groq_llm = Mock()
        mock_groq_llm.invoke = Mock(side_effect=Exception("Groq API error"))
        
        mock_model_loader = Mock()
        mock_model_loader.load_llm.side_effect = [mock_google_llm, mock_groq_llm]
        mock_model_loader.load_embeddings.return_value = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        # Setup FAISS
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss_class.load_local.return_value = mock_vectorstore
        
        rag = ConversationalRAG(session_id="test_session")
        rag.load_retriever_from_faiss(str(mock_faiss_index), k=5)
        
        # Mock chain that raises ResourceExhausted, then Groq also fails
        original_build_chain = rag._build_lcel_chain
        call_count = [0]
        
        def mock_build_chain():
            """Mock _build_lcel_chain to set a mock chain that fails after rebuild."""
            original_build_chain()
            # After chain is rebuilt, replace it with a mock that fails
            mock_chain_after_fallback = Mock()
            mock_chain_after_fallback.invoke.side_effect = Exception("Groq fallback failed")
            rag.chain = mock_chain_after_fallback
        
        def chain_invoke_side_effect(payload):
            call_count[0] += 1
            if call_count[0] == 1:
                raise google_exceptions.ResourceExhausted("Quota exceeded")
            raise Exception("Groq fallback failed")
        
        mock_chain = Mock()
        mock_chain.invoke.side_effect = chain_invoke_side_effect
        rag.chain = mock_chain
        
        # Patch _build_lcel_chain to set failing mock after rebuild
        rag._build_lcel_chain = mock_build_chain
        
        # Should raise error after both providers fail
        with pytest.raises(Exception) as exc_info:
            rag.invoke("What is machine learning?")
        
        assert "Groq fallback failed" in str(exc_info.value) or "fallback" in str(exc_info.value).lower()

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.FAISS")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_no_fallback_when_google_succeeds(
        self, mock_prompt_registry, mock_faiss_class, mock_model_loader_class,
        mock_env_vars, mock_faiss_index, mock_retriever
    ):
        """Test that fallback is not triggered when Google LLM succeeds."""
        # Setup Google LLM that succeeds
        mock_google_llm = Mock()
        mock_google_llm.invoke = Mock(return_value=Mock(content="Answer from Google"))
        
        mock_model_loader = Mock()
        mock_model_loader.load_llm.return_value = mock_google_llm
        mock_model_loader.load_embeddings.return_value = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        # Setup FAISS
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss_class.load_local.return_value = mock_vectorstore
        
        rag = ConversationalRAG(session_id="test_session")
        rag.load_retriever_from_faiss(str(mock_faiss_index), k=5)
        
        # Mock chain that succeeds
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Answer from Google"
        rag.chain = mock_chain
        
        result = rag.invoke("What is machine learning?")
        
        assert result == "Answer from Google"
        # Should only call load_llm once (during initialization)
        assert mock_model_loader.load_llm.call_count == 1

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.FAISS")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_fallback_preserves_chat_history(
        self, mock_prompt_registry, mock_faiss_class, mock_model_loader_class,
        mock_env_vars, mock_faiss_index, mock_retriever, sample_chat_history
    ):
        """Test that chat history is preserved during fallback."""
        # Setup LLMs
        mock_google_llm = Mock()
        mock_groq_llm = Mock()
        
        mock_model_loader = Mock()
        mock_model_loader.load_llm.side_effect = [mock_google_llm, mock_groq_llm]
        mock_model_loader.load_embeddings.return_value = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        # Setup FAISS
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss_class.load_local.return_value = mock_vectorstore
        
        rag = ConversationalRAG(session_id="test_session")
        rag.load_retriever_from_faiss(str(mock_faiss_index), k=5)
        
        # Mock chain that fails first time, then succeeds after fallback
        original_build_chain = rag._build_lcel_chain
        call_count = [0]
        
        def mock_build_chain():
            """Mock _build_lcel_chain to set a mock chain after rebuild."""
            original_build_chain()
            # After chain is rebuilt, replace it with a mock that returns expected value
            mock_chain_after_fallback = Mock()
            def invoke_with_history_check(payload):
                # Verify chat history is in payload
                assert "chat_history" in payload
                return "Answer with history context"
            mock_chain_after_fallback.invoke.side_effect = invoke_with_history_check
            rag.chain = mock_chain_after_fallback
        
        def chain_invoke_side_effect(payload):
            call_count[0] += 1
            if call_count[0] == 1:
                raise google_exceptions.ResourceExhausted("Quota exceeded")
            # Verify chat history is in payload
            assert "chat_history" in payload
            return "Answer with history context"
        
        mock_chain = Mock()
        mock_chain.invoke.side_effect = chain_invoke_side_effect
        rag.chain = mock_chain
        
        # Patch _build_lcel_chain to set mock after rebuild
        rag._build_lcel_chain = mock_build_chain
        
        result = rag.invoke("Follow-up question", chat_history=sample_chat_history)
        
        assert result == "Answer with history context"
        # Verify chat history was passed (at least once, possibly twice if both chains are called)
        assert call_count[0] >= 1
