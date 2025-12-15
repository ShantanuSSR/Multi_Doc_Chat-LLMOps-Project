"""
Integration tests for error handling and recovery scenarios.
"""
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
import os
# Import LCELCompatibleMock from conftest
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from conftest import LCELCompatibleMock

from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from multi_doc_chat.exception.custom_exception import DocumentPortalException


class TestErrorHandling:
    """Test error handling and recovery scenarios."""

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.save_uploaded_files")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.load_documents")
    def test_ingestion_with_invalid_files_handles_gracefully(
        self, mock_load_docs, mock_save_files, mock_model_loader_class, mock_env_vars, temp_dir
    ):
        """Test that ingestion handles invalid files gracefully."""
        mock_model_loader = Mock()
        mock_model_loader.load_embeddings.return_value = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        # Simulate no documents loaded (invalid files)
        mock_save_files.return_value = []
        mock_load_docs.return_value = []
        
        ingestor = ChatIngestor(
            temp_base=str(temp_dir / "data"),
            faiss_base=str(temp_dir / "faiss_index"),
            session_id="test_session"
        )
        
        mock_uploaded_files = [Mock(filename="invalid.exe")]
        
        with pytest.raises(DocumentPortalException) as exc_info:
            ingestor.built_retriver(mock_uploaded_files)
        
        assert "Failed to build retriever" in str(exc_info.value)

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.FAISS")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_rag_with_missing_index_raises_error(
        self, mock_prompt_registry, mock_faiss_class, mock_model_loader_class,
        mock_env_vars, temp_dir
    ):
        """Test that RAG raises error when index path doesn't exist."""
        mock_model_loader = Mock()
        mock_model_loader.load_llm.return_value = Mock()
        mock_model_loader.load_embeddings.return_value = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        rag = ConversationalRAG(session_id="test_session")
        
        nonexistent_path = temp_dir / "nonexistent_index"
        
        with pytest.raises(DocumentPortalException) as exc_info:
            rag.load_retriever_from_faiss(str(nonexistent_path))
        
        assert "Loading error" in str(exc_info.value)

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_rag_invoke_without_chain_raises_clear_error(
        self, mock_prompt_registry, mock_model_loader_class, mock_env_vars
    ):
        """Test that invoking RAG without chain raises clear error message."""
        mock_model_loader = Mock()
        mock_model_loader.load_llm.return_value = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        rag = ConversationalRAG(session_id="test_session")
        
        with pytest.raises(DocumentPortalException) as exc_info:
            rag.invoke("Test question")
        
        assert "not initialized" in str(exc_info.value)
        assert "load_retriever_from_faiss" in str(exc_info.value)

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.FAISS")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.save_uploaded_files")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.load_documents")
    def test_ingestion_retry_on_faiss_error(
        self, mock_load_docs, mock_save_files, mock_faiss_class,
        mock_model_loader_class, mock_env_vars, temp_dir, sample_documents
    ):
        """Test that ingestion retries on FAISS errors."""
        mock_model_loader = Mock()
        mock_model_loader.load_embeddings.return_value = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_save_files.return_value = [temp_dir / "doc1.pdf"]
        mock_load_docs.return_value = sample_documents
        
        # First call fails, second succeeds
        mock_vectorstore = Mock()
        mock_faiss_manager = Mock()
        mock_faiss_manager.load_or_create.side_effect = [
            Exception("First attempt failed"),
            mock_vectorstore
        ]
        
        # We need to patch FaissManager
        with patch("multi_doc_chat.src.document_ingestion.data_ingestion.FaissManager") as mock_faiss_manager_class:
            mock_faiss_manager_class.return_value = mock_faiss_manager
            
            ingestor = ChatIngestor(
                temp_base=str(temp_dir / "data"),
                faiss_base=str(temp_dir / "faiss_index"),
                session_id="test_session"
            )
            
            mock_uploaded_files = [Mock(filename="doc1.pdf")]
            
            # Should succeed on retry
            retriever = ingestor.built_retriver(mock_uploaded_files)
            
            # Verify retry was attempted
            assert mock_faiss_manager.load_or_create.call_count == 2

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.FAISS")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_rag_handles_empty_answer_gracefully(
        self, mock_prompt_registry, mock_faiss_class, mock_model_loader_class,
        mock_env_vars, mock_faiss_index, mock_retriever
    ):
        """Test that RAG handles empty answers gracefully."""
        mock_model_loader = Mock()
        mock_model_loader.load_llm.return_value = Mock()
        mock_model_loader.load_embeddings.return_value = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss_class.load_local.return_value = mock_vectorstore
        
        rag = ConversationalRAG(session_id="test_session")
        rag.load_retriever_from_faiss(str(mock_faiss_index), k=5)
        
        # Mock chain that returns empty answer
        mock_chain = Mock()
        mock_chain.invoke.return_value = ""
        rag.chain = mock_chain
        
        result = rag.invoke("Test question")
        
        assert result == "no answer generated."

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.FAISS")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_rag_handles_invalid_answer_validation(
        self, mock_prompt_registry, mock_faiss_class, mock_model_loader_class,
        mock_env_vars, mock_faiss_index, mock_retriever
    ):
        """Test that RAG validates answer format and handles invalid answers."""
        mock_model_loader = Mock()
        mock_model_loader.load_llm.return_value = Mock()
        mock_model_loader.load_embeddings.return_value = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss_class.load_local.return_value = mock_vectorstore
        
        rag = ConversationalRAG(session_id="test_session")
        rag.load_retriever_from_faiss(str(mock_faiss_index), k=5)
        
        # Mock chain that returns answer exceeding max length
        mock_chain = Mock()
        # Create answer that exceeds ChatAnswer max_length (4096)
        invalid_answer = "a" * 5000
        mock_chain.invoke.return_value = invalid_answer
        rag.chain = mock_chain
        
        with pytest.raises(DocumentPortalException) as exc_info:
            rag.invoke("Test question")
        
        assert "Invalid chat answer" in str(exc_info.value)
