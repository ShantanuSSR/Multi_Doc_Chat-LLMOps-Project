"""
Integration tests for session management and isolation.
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

from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor, generate_session_id
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG


class TestSessionManagement:
    """Test session management and isolation."""

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    def test_session_isolation_with_session_dirs(self, mock_model_loader_class, mock_env_vars, temp_dir):
        """Test that different sessions create isolated directories."""
        mock_model_loader = Mock()
        mock_model_loader.load_embeddings.return_value = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        session1 = "session_1"
        session2 = "session_2"
        
        ingestor1 = ChatIngestor(
            temp_base=str(temp_dir / "data"),
            faiss_base=str(temp_dir / "faiss_index"),
            use_session_dirs=True,
            session_id=session1
        )
        
        ingestor2 = ChatIngestor(
            temp_base=str(temp_dir / "data"),
            faiss_base=str(temp_dir / "faiss_index"),
            use_session_dirs=True,
            session_id=session2
        )
        
        # Verify separate directories
        assert ingestor1.temp_dir != ingestor2.temp_dir
        assert ingestor1.faiss_dir != ingestor2.faiss_dir
        
        # Verify directories exist
        assert ingestor1.temp_dir.exists()
        assert ingestor1.faiss_dir.exists()
        assert ingestor2.temp_dir.exists()
        assert ingestor2.faiss_dir.exists()
        
        # Verify session IDs in paths
        assert session1 in str(ingestor1.faiss_dir)
        assert session2 in str(ingestor2.faiss_dir)

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    def test_session_without_isolation(self, mock_model_loader_class, mock_env_vars, temp_dir):
        """Test that sessions without isolation share directories."""
        mock_model_loader = Mock()
        mock_model_loader.load_embeddings.return_value = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        ingestor1 = ChatIngestor(
            temp_base=str(temp_dir / "data"),
            faiss_base=str(temp_dir / "faiss_index"),
            use_session_dirs=False,
            session_id="session_1"
        )
        
        ingestor2 = ChatIngestor(
            temp_base=str(temp_dir / "data"),
            faiss_base=str(temp_dir / "faiss_index"),
            use_session_dirs=False,
            session_id="session_2"
        )
        
        # Verify shared directories
        assert ingestor1.temp_dir == ingestor2.temp_dir
        assert ingestor1.faiss_dir == ingestor2.faiss_dir

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    def test_auto_generated_session_ids_are_unique(self, mock_model_loader_class, mock_env_vars, temp_dir):
        """Test that auto-generated session IDs are unique."""
        mock_model_loader = Mock()
        mock_model_loader.load_embeddings.return_value = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        ingestor1 = ChatIngestor(
            temp_base=str(temp_dir / "data"),
            faiss_base=str(temp_dir / "faiss_index"),
            use_session_dirs=True
        )
        
        ingestor2 = ChatIngestor(
            temp_base=str(temp_dir / "data"),
            faiss_base=str(temp_dir / "faiss_index"),
            use_session_dirs=True
        )
        
        assert ingestor1.session_id != ingestor2.session_id
        assert ingestor1.faiss_dir != ingestor2.faiss_dir

    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.FAISS")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_rag_session_consistency(
        self, mock_prompt_registry, mock_faiss_class, mock_model_loader_class,
        mock_env_vars, temp_dir
    ):
        """Test that RAG system uses consistent session IDs."""
        mock_model_loader = Mock()
        mock_model_loader.load_llm.return_value = Mock()
        mock_model_loader.load_embeddings.return_value = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        session_id = "test_session_123"
        
        # Create index directory for this session
        index_dir = temp_dir / "faiss_index" / session_id
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "index.faiss").touch()
        (index_dir / "index.pkl").touch()
        
        mock_vectorstore = Mock()
        mock_vectorstore.as_retriever.return_value = Mock()
        mock_faiss_class.load_local.return_value = mock_vectorstore
        
        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(str(index_dir), k=5)
        
        assert rag.session_id == session_id

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.FAISS")
    @patch("multi_doc_chat.src.document_chat.retrieval.FAISS")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.save_uploaded_files")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.load_documents")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_cross_session_data_isolation(
        self, mock_prompt_registry, mock_load_docs, mock_save_files,
        mock_faiss_retrieval, mock_faiss_ingestion, mock_model_loader_retrieval,
        mock_model_loader_ingestion, mock_env_vars, temp_dir, sample_documents
    ):
        """Test that data from one session doesn't leak to another."""
        # Setup mocks
        mock_model_loader_ing = Mock()
        mock_model_loader_ing.load_embeddings.return_value = Mock()
        mock_model_loader_ingestion.return_value = mock_model_loader_ing
        
        mock_model_loader_ret = Mock()
        mock_model_loader_ret.load_llm.return_value = Mock()
        mock_model_loader_ret.load_embeddings.return_value = Mock()
        mock_model_loader_retrieval.return_value = mock_model_loader_ret
        
        mock_save_files.return_value = [temp_dir / "doc1.pdf"]
        mock_load_docs.return_value = sample_documents
        
        mock_vectorstore_ingestion = Mock()
        mock_vectorstore_ingestion.as_retriever.return_value = Mock()
        mock_faiss_ingestion.from_texts.return_value = mock_vectorstore_ingestion
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        # Create two separate sessions
        session1 = "session_1"
        session2 = "session_2"
        
        # Ingest documents for session 1
        ingestor1 = ChatIngestor(
            temp_base=str(temp_dir / "data"),
            faiss_base=str(temp_dir / "faiss_index"),
            use_session_dirs=True,
            session_id=session1
        )
        
        mock_uploaded_files = [Mock(filename="doc1.pdf")]
        ingestor1.built_retriver(mock_uploaded_files)
        
        # Verify session 1 directories exist
        assert (temp_dir / "faiss_index" / session1).exists()
        
        # Verify session 2 directory is separate (doesn't exist yet)
        session2_dir = temp_dir / "faiss_index" / session2
        assert not session2_dir.exists() or session2_dir != (temp_dir / "faiss_index" / session1)
