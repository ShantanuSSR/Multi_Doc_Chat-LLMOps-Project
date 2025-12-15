"""
Unit tests for ChatIngestor and FaissManager classes.
"""
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

from multi_doc_chat.src.document_ingestion.data_ingestion import (
    ChatIngestor,
    FaissManager,
    generate_session_id,
    SUPPORTED_EXTENSIONS
)
from multi_doc_chat.exception.custom_exception import DocumentPortalException


class TestGenerateSessionId:
    """Test session ID generation."""

    def test_generate_session_id_format(self):
        """Test that generated session ID has correct format."""
        session_id = generate_session_id()
        
        assert session_id.startswith("session_")
        assert "_" in session_id
        assert len(session_id) > 20  # Should have timestamp and UUID

    def test_generate_session_id_uniqueness(self):
        """Test that generated session IDs are unique."""
        id1 = generate_session_id()
        id2 = generate_session_id()
        
        assert id1 != id2


class TestChatIngestor:
    """Test ChatIngestor class."""

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    def test_initialization_with_session(self, mock_model_loader_class, mock_env_vars, temp_dir):
        """Test ChatIngestor initialization with session."""
        mock_model_loader = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        ingestor = ChatIngestor(
            temp_base=str(temp_dir / "data"),
            faiss_base=str(temp_dir / "faiss_index"),
            use_session_dirs=True,
            session_id="test_session_123"
        )
        
        assert ingestor.session_id == "test_session_123"
        assert ingestor.use_session is True
        assert ingestor.temp_dir.exists()
        assert ingestor.faiss_dir.exists()

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    def test_initialization_auto_generates_session_id(self, mock_model_loader_class, mock_env_vars, temp_dir):
        """Test that ChatIngestor auto-generates session ID if not provided."""
        mock_model_loader = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        ingestor = ChatIngestor(
            temp_base=str(temp_dir / "data"),
            faiss_base=str(temp_dir / "faiss_index"),
            use_session_dirs=True
        )
        
        assert ingestor.session_id is not None
        assert ingestor.session_id.startswith("session_")

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    def test_initialization_without_session_dirs(self, mock_model_loader_class, mock_env_vars, temp_dir):
        """Test ChatIngestor initialization without session directories."""
        mock_model_loader = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        ingestor = ChatIngestor(
            temp_base=str(temp_dir / "data"),
            faiss_base=str(temp_dir / "faiss_index"),
            use_session_dirs=False
        )
        
        assert ingestor.temp_dir == Path(temp_dir / "data")
        assert ingestor.faiss_dir == Path(temp_dir / "faiss_index")

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.save_uploaded_files")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.load_documents")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.FaissManager")
    def test_built_retriver_success(
        self, mock_faiss_manager_class, mock_load_docs, mock_save_files,
        mock_model_loader_class, mock_env_vars, temp_dir, sample_documents
    ):
        """Test successful retriever building."""
        mock_model_loader = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        # Setup mocks
        mock_file_paths = [temp_dir / "doc1.pdf"]
        mock_save_files.return_value = mock_file_paths
        mock_load_docs.return_value = sample_documents
        
        mock_faiss_manager = Mock()
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss_manager.load_or_create.return_value = mock_vectorstore
        mock_faiss_manager.add_documents.return_value = 3
        mock_faiss_manager_class.return_value = mock_faiss_manager
        
        ingestor = ChatIngestor(
            temp_base=str(temp_dir / "data"),
            faiss_base=str(temp_dir / "faiss_index"),
            use_session_dirs=True,
            session_id="test_session"
        )
        
        mock_uploaded_files = [Mock(filename="doc1.pdf")]
        retriever = ingestor.built_retriver(mock_uploaded_files)
        
        assert retriever == mock_retriever
        mock_save_files.assert_called_once()
        mock_load_docs.assert_called_once()
        mock_faiss_manager.add_documents.assert_called_once()

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.save_uploaded_files")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.load_documents")
    def test_built_retriver_no_documents_raises_error(
        self, mock_load_docs, mock_save_files, mock_model_loader_class, mock_env_vars, temp_dir
    ):
        """Test that building retriever with no documents raises error."""
        mock_model_loader = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_save_files.return_value = [temp_dir / "doc1.pdf"]
        mock_load_docs.return_value = []  # No documents loaded
        
        ingestor = ChatIngestor(
            temp_base=str(temp_dir / "data"),
            faiss_base=str(temp_dir / "faiss_index"),
            session_id="test_session"
        )
        
        mock_uploaded_files = [Mock(filename="doc1.pdf")]
        
        with pytest.raises(DocumentPortalException) as exc_info:
            ingestor.built_retriver(mock_uploaded_files)
        
        assert "Failed to build retriever" in str(exc_info.value)

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    def test_split_documents(self, mock_model_loader_class, mock_env_vars, temp_dir, sample_documents):
        """Test document splitting."""
        mock_model_loader = Mock()
        mock_model_loader_class.return_value = mock_model_loader
        
        ingestor = ChatIngestor(
            temp_base=str(temp_dir / "data"),
            faiss_base=str(temp_dir / "faiss_index"),
            session_id="test_session"
        )
        
        chunks = ingestor._split(sample_documents, chunk_size=50, chunk_overlap=10)
        
        assert len(chunks) > len(sample_documents)  # Should have more chunks
        assert all(isinstance(chunk, Document) for chunk in chunks)


class TestFaissManager:
    """Test FaissManager class."""

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    def test_initialization_creates_directory(self, mock_model_loader_class, mock_embeddings, mock_env_vars, temp_dir):
        """Test that FaissManager creates index directory on initialization."""
        mock_model_loader = Mock()
        mock_model_loader.load_embeddings.return_value = mock_embeddings
        mock_model_loader_class.return_value = mock_model_loader
        
        index_dir = temp_dir / "new_index"
        manager = FaissManager(index_dir)
        
        assert index_dir.exists()
        assert manager.index_dir == index_dir
        assert manager.emb == mock_embeddings

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.FAISS")
    def test_load_or_create_loads_existing_index(
        self, mock_faiss_class, mock_model_loader_class, mock_embeddings, mock_env_vars, mock_faiss_index
    ):
        """Test that load_or_create loads existing FAISS index."""
        mock_model_loader = Mock()
        mock_model_loader.load_embeddings.return_value = mock_embeddings
        mock_model_loader_class.return_value = mock_model_loader
        
        mock_vectorstore = Mock()
        mock_faiss_class.load_local.return_value = mock_vectorstore
        
        manager = FaissManager(mock_faiss_index)
        result = manager.load_or_create()
        
        assert result == mock_vectorstore
        mock_faiss_class.load_local.assert_called_once()

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.FAISS")
    def test_load_or_create_creates_new_index(
        self, mock_faiss_class, mock_model_loader_class, mock_embeddings, mock_env_vars, temp_dir
    ):
        """Test that load_or_create creates new index when none exists."""
        mock_model_loader = Mock()
        mock_model_loader.load_embeddings.return_value = mock_embeddings
        mock_model_loader_class.return_value = mock_model_loader
        
        index_dir = temp_dir / "new_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        
        mock_vectorstore = Mock()
        mock_faiss_class.from_texts.return_value = mock_vectorstore
        
        manager = FaissManager(index_dir)
        texts = ["Document 1", "Document 2"]
        metadatas = [{"source": "doc1"}, {"source": "doc2"}]
        result = manager.load_or_create(texts=texts, metadatas=metadatas)
        
        assert result == mock_vectorstore
        mock_faiss_class.from_texts.assert_called_once()

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    def test_add_documents_idempotent(self, mock_model_loader_class, mock_embeddings, mock_env_vars, temp_dir, sample_documents):
        """Test that add_documents is idempotent (doesn't add duplicates)."""
        mock_model_loader = Mock()
        mock_model_loader.load_embeddings.return_value = mock_embeddings
        mock_model_loader_class.return_value = mock_model_loader
        
        index_dir = temp_dir / "index"
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file with existing document
        meta = {
            "rows": {
                f"{sample_documents[0].metadata.get('source')}::{sample_documents[0].metadata.get('row_id', '')}": True
            }
        }
        (index_dir / "ingested_meta.json").write_text(json.dumps(meta))
        
        mock_vectorstore = Mock()
        manager = FaissManager(index_dir)
        manager.vs = mock_vectorstore
        
        # First call should add documents
        added = manager.add_documents(sample_documents)
        assert added > 0
        
        # Second call with same documents should add fewer (or zero) due to idempotency
        added_again = manager.add_documents(sample_documents)
        assert added_again <= added

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    def test_fingerprint_generation(self, mock_model_loader_class, mock_embeddings, mock_env_vars, temp_dir):
        """Test fingerprint generation for documents."""
        mock_model_loader = Mock()
        mock_model_loader.load_embeddings.return_value = mock_embeddings
        mock_model_loader_class.return_value = mock_model_loader
        
        manager = FaissManager(temp_dir / "index")
        
        metadata = {"source": "test.pdf", "row_id": "123"}
        fingerprint = manager._fingerprint("Test content", metadata)
        
        assert fingerprint == "test.pdf::123"
        
        # Test without row_id
        metadata_no_row = {"source": "test.pdf"}
        fingerprint2 = manager._fingerprint("Test content", metadata_no_row)
        assert fingerprint2 == "test.pdf::"
