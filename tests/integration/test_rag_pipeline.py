"""
Integration tests for end-to-end RAG pipeline.
"""
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document
from langchain_core.messages import HumanMessage, AIMessage

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


class TestEndToEndRAGPipeline:
    """Test complete RAG pipeline from ingestion to retrieval."""

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.FAISS")
    @patch("multi_doc_chat.src.document_chat.retrieval.FAISS")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.save_uploaded_files")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.load_documents")
    def test_complete_pipeline_ingestion_to_retrieval(
        self, mock_load_docs, mock_save_files, mock_faiss_retrieval, mock_faiss_ingestion,
        mock_model_loader_retrieval, mock_model_loader_ingestion, mock_env_vars, temp_dir, sample_documents
    ):
        """Test complete workflow: ingest documents, build index, retrieve and answer."""
        # Setup mocks for ingestion
        mock_model_loader_ing = Mock()
        mock_model_loader_ing.load_embeddings.return_value = mock_embeddings()
        mock_model_loader_ingestion.return_value = mock_model_loader_ing
        
        # Setup mocks for retrieval
        mock_model_loader_ret = Mock()
        mock_model_loader_ret.load_llm.return_value = mock_llm()
        mock_model_loader_ret.load_embeddings.return_value = mock_embeddings()
        mock_model_loader_retrieval.return_value = mock_model_loader_ret
        
        # Setup file operations
        mock_file_paths = [temp_dir / "doc1.pdf"]
        mock_save_files.return_value = mock_file_paths
        mock_load_docs.return_value = sample_documents
        
        # Setup FAISS for ingestion
        mock_vectorstore_ingestion = Mock()
        mock_vectorstore_ingestion.as_retriever.return_value = Mock()
        mock_faiss_ingestion.from_texts.return_value = mock_vectorstore_ingestion
        mock_faiss_ingestion.load_local.return_value = None  # No existing index
        
        # Setup FAISS for retrieval
        mock_vectorstore_retrieval = Mock()
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = sample_documents[:2]
        mock_vectorstore_retrieval.as_retriever.return_value = mock_retriever
        mock_faiss_retrieval.load_local.return_value = mock_vectorstore_retrieval
        
        # Step 1: Ingest documents
        ingestor = ChatIngestor(
            temp_base=str(temp_dir / "data"),
            faiss_base=str(temp_dir / "faiss_index"),
            session_id="test_session"
        )
        
        mock_uploaded_files = [Mock(filename="doc1.pdf")]
        retriever = ingestor.built_retriver(mock_uploaded_files)
        
        assert retriever is not None
        
        # Step 2: Load RAG system and retrieve
        index_path = temp_dir / "faiss_index" / "test_session"
        index_path.mkdir(parents=True, exist_ok=True)
        
        rag = ConversationalRAG(session_id="test_session")
        rag.load_retriever_from_faiss(str(index_path), k=2)
        
        assert rag.chain is not None

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.FAISS")
    @patch("multi_doc_chat.src.document_chat.retrieval.FAISS")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.save_uploaded_files")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.load_documents")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_pipeline_with_chat_history(
        self, mock_prompt_registry, mock_load_docs, mock_save_files, mock_faiss_retrieval,
        mock_faiss_ingestion, mock_model_loader_retrieval, mock_model_loader_ingestion,
        mock_env_vars, temp_dir, sample_documents, sample_chat_history
    ):
        """Test RAG pipeline with chat history context."""
        # Setup mocks
        mock_model_loader_ing = Mock()
        mock_model_loader_ing.load_embeddings.return_value = mock_embeddings()
        mock_model_loader_ingestion.return_value = mock_model_loader_ing
        
        mock_model_loader_ret = Mock()
        mock_model_loader_ret.load_llm.return_value = mock_llm()
        mock_model_loader_ret.load_embeddings.return_value = mock_embeddings()
        mock_model_loader_retrieval.return_value = mock_model_loader_ret
        
        mock_save_files.return_value = [temp_dir / "doc1.pdf"]
        mock_load_docs.return_value = sample_documents
        
        mock_vectorstore_ingestion = Mock()
        mock_vectorstore_ingestion.as_retriever.return_value = Mock()
        mock_faiss_ingestion.from_texts.return_value = mock_vectorstore_ingestion
        
        mock_vectorstore_retrieval = Mock()
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = sample_documents[:2]
        mock_vectorstore_retrieval.as_retriever.return_value = mock_retriever
        mock_faiss_retrieval.load_local.return_value = mock_vectorstore_retrieval
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        # Ingest
        ingestor = ChatIngestor(
            temp_base=str(temp_dir / "data"),
            faiss_base=str(temp_dir / "faiss_index"),
            session_id="test_session"
        )
        
        mock_uploaded_files = [Mock(filename="doc1.pdf")]
        ingestor.built_retriver(mock_uploaded_files)
        
        # Retrieve with chat history
        index_path = temp_dir / "faiss_index" / "test_session"
        index_path.mkdir(parents=True, exist_ok=True)
        
        rag = ConversationalRAG(session_id="test_session")
        rag.load_retriever_from_faiss(str(index_path), k=2)
        
        # Mock chain to return answer
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Based on the context, machine learning is..."
        rag.chain = mock_chain
        
        result = rag.invoke("Tell me more about it.", chat_history=sample_chat_history)
        
        assert result is not None
        # Verify chat history was passed to chain
        call_args = mock_chain.invoke.call_args[0][0]
        assert "chat_history" in call_args

    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.ModelLoader")
    @patch("multi_doc_chat.src.document_chat.retrieval.ModelLoader")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.FAISS")
    @patch("multi_doc_chat.src.document_chat.retrieval.FAISS")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.save_uploaded_files")
    @patch("multi_doc_chat.src.document_ingestion.data_ingestion.load_documents")
    @patch("multi_doc_chat.src.document_chat.retrieval.PROMPT_REGISTRY")
    def test_pipeline_multiple_documents(
        self, mock_prompt_registry, mock_load_docs, mock_save_files, mock_faiss_retrieval,
        mock_faiss_ingestion, mock_model_loader_retrieval, mock_model_loader_ingestion,
        mock_env_vars, temp_dir, sample_documents
    ):
        """Test RAG pipeline with multiple documents."""
        # Setup mocks
        mock_model_loader_ing = Mock()
        mock_model_loader_ing.load_embeddings.return_value = mock_embeddings()
        mock_model_loader_ingestion.return_value = mock_model_loader_ing
        
        mock_model_loader_ret = Mock()
        mock_model_loader_ret.load_llm.return_value = mock_llm()
        mock_model_loader_ret.load_embeddings.return_value = mock_embeddings()
        mock_model_loader_retrieval.return_value = mock_model_loader_ret
        
        # Multiple documents
        extended_docs = sample_documents * 2
        mock_save_files.return_value = [temp_dir / f"doc{i}.pdf" for i in range(1, 4)]
        mock_load_docs.return_value = extended_docs
        
        mock_vectorstore_ingestion = Mock()
        mock_vectorstore_ingestion.as_retriever.return_value = Mock()
        mock_faiss_ingestion.from_texts.return_value = mock_vectorstore_ingestion
        
        mock_vectorstore_retrieval = Mock()
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = extended_docs[:3]
        mock_vectorstore_retrieval.as_retriever.return_value = mock_retriever
        mock_faiss_retrieval.load_local.return_value = mock_vectorstore_retrieval
        
        mock_prompt = LCELCompatibleMock()
        mock_prompt_registry.__getitem__ = Mock(return_value=mock_prompt)
        
        # Ingest multiple documents
        ingestor = ChatIngestor(
            temp_base=str(temp_dir / "data"),
            faiss_base=str(temp_dir / "faiss_index"),
            session_id="test_session"
        )
        
        mock_uploaded_files = [Mock(filename=f"doc{i}.pdf") for i in range(1, 4)]
        retriever = ingestor.built_retriver(mock_uploaded_files)
        
        assert retriever is not None
        
        # Retrieve
        index_path = temp_dir / "faiss_index" / "test_session"
        index_path.mkdir(parents=True, exist_ok=True)
        
        rag = ConversationalRAG(session_id="test_session")
        rag.load_retriever_from_faiss(str(index_path), k=3)
        
        assert rag.chain is not None


# Helper functions for mocks
def mock_llm():
    """Create a mock LLM."""
    llm = Mock()
    llm.invoke = Mock(return_value=Mock(content="Test response"))
    return llm


def mock_embeddings():
    """Create a mock embeddings model."""
    emb = Mock()
    emb.embed_query = Mock(return_value=[0.1] * 768)
    emb.embed_documents = Mock(return_value=[[0.1] * 768] * 3)
    return emb
