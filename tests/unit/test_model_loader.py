"""
Unit tests for ModelLoader and ApiKeyManager.
"""
import os
import sys
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from multi_doc_chat.utils.model_loader import ModelLoader, ApiKeyManager
from multi_doc_chat.exception.custom_exception import DocumentPortalException


class TestApiKeyManager:
    """Test ApiKeyManager class."""

    def test_api_key_manager_loads_from_env(self, mock_env_vars):
        """Test that ApiKeyManager loads API keys from environment variables."""
        manager = ApiKeyManager()
        
        assert manager.get("GOOGLE_API_KEY") == "test_google_key_12345"
        assert manager.get("GROQ_API_KEY") == "test_groq_key_12345"

    def test_api_key_manager_loads_from_json_env(self, monkeypatch):
        """Test that ApiKeyManager loads API keys from JSON environment variable."""
        api_keys_json = json.dumps({
            "GOOGLE_API_KEY": "json_google_key",
            "GROQ_API_KEY": "json_groq_key"
        })
        monkeypatch.setenv("apikeyliveclass", api_keys_json)
        
        manager = ApiKeyManager()
        
        assert manager.get("GOOGLE_API_KEY") == "json_google_key"
        assert manager.get("GROQ_API_KEY") == "json_groq_key"

    def test_api_key_manager_missing_keys_raises_exception(self, monkeypatch):
        """Test that missing API keys raise DocumentPortalException."""
        # Remove all API keys
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("apikeyliveclass", raising=False)
        
        with pytest.raises(DocumentPortalException) as exc_info:
            ApiKeyManager()
        
        assert "Missing API keys" in str(exc_info.value)

    def test_api_key_manager_get_missing_key_raises_keyerror(self, mock_env_vars):
        """Test that getting a missing key raises KeyError."""
        manager = ApiKeyManager()
        
        with pytest.raises(KeyError) as exc_info:
            manager.get("NONEXISTENT_KEY")
        
        assert "NONEXISTENT_KEY" in str(exc_info.value)


class TestModelLoader:
    """Test ModelLoader class."""

    @patch("multi_doc_chat.utils.model_loader.load_config")
    @patch("multi_doc_chat.utils.model_loader.GoogleGenerativeAIEmbeddings")
    def test_load_embeddings_success(self, mock_embeddings_class, mock_load_config, mock_env_vars, mock_config):
        """Test successful embedding model loading."""
        mock_load_config.return_value = mock_config
        mock_emb_instance = Mock()
        mock_embeddings_class.return_value = mock_emb_instance
        
        loader = ModelLoader()
        embeddings = loader.load_embeddings()
        
        assert embeddings == mock_emb_instance
        mock_embeddings_class.assert_called_once()
        assert "text-embedding-004" in str(mock_embeddings_class.call_args)

    @patch("multi_doc_chat.utils.model_loader.load_config")
    @patch("multi_doc_chat.utils.model_loader.ChatGoogleGenerativeAI")
    def test_load_llm_google_provider(self, mock_llm_class, mock_load_config, mock_env_vars, mock_config):
        """Test loading Google LLM provider."""
        mock_load_config.return_value = mock_config
        mock_llm_instance = Mock()
        mock_llm_class.return_value = mock_llm_instance
        
        loader = ModelLoader()
        llm = loader.load_llm()
        
        assert llm == mock_llm_instance
        mock_llm_class.assert_called_once()
        call_kwargs = mock_llm_class.call_args[1]
        assert call_kwargs["model"] == "gemini-2.0-flash"
        assert call_kwargs["temperature"] == 0

    @patch("multi_doc_chat.utils.model_loader.load_config")
    @patch("multi_doc_chat.utils.model_loader.ChatGroq")
    def test_load_llm_groq_provider(self, mock_llm_class, mock_load_config, mock_env_vars, mock_config):
        """Test loading Groq LLM provider."""
        mock_load_config.return_value = mock_config
        mock_llm_instance = Mock()
        mock_llm_class.return_value = mock_llm_instance
        
        loader = ModelLoader()
        llm = loader.load_llm(provider_override="groq")
        
        assert llm == mock_llm_instance
        mock_llm_class.assert_called_once()
        call_kwargs = mock_llm_class.call_args[1]
        assert call_kwargs["model"] == "openai/gpt-oss-20b"

    @patch("multi_doc_chat.utils.model_loader.load_config")
    @patch("multi_doc_chat.utils.model_loader.ChatGoogleGenerativeAI")
    @patch("multi_doc_chat.utils.model_loader.ChatGroq")
    def test_load_llm_fallback_to_groq_on_error(
        self, mock_groq_class, mock_google_class, mock_load_config, mock_env_vars, mock_config
    ):
        """Test that LLM loader falls back to Groq when Google fails."""
        mock_load_config.return_value = mock_config
        
        # Make Google LLM raise an exception
        mock_google_class.side_effect = Exception("Google API error")
        mock_groq_instance = Mock()
        mock_groq_class.return_value = mock_groq_instance
        
        loader = ModelLoader()
        llm = loader.load_llm()
        
        # Should fallback to Groq
        assert llm == mock_groq_instance
        mock_google_class.assert_called_once()
        mock_groq_class.assert_called_once()

    @patch("multi_doc_chat.utils.model_loader.load_config")
    def test_load_llm_invalid_provider_raises_error(self, mock_load_config, mock_env_vars, mock_config):
        """Test that invalid provider raises ValueError."""
        mock_load_config.return_value = mock_config
        
        loader = ModelLoader()
        
        with pytest.raises(ValueError) as exc_info:
            loader.load_llm(provider_override="invalid_provider")
        
        assert "not found in config" in str(exc_info.value)

    @patch("multi_doc_chat.utils.model_loader.load_config")
    @patch("multi_doc_chat.utils.model_loader.GoogleGenerativeAIEmbeddings")
    def test_load_embeddings_handles_exception(self, mock_embeddings_class, mock_load_config, mock_env_vars, mock_config):
        """Test that embedding loading handles exceptions properly."""
        mock_load_config.return_value = mock_config
        mock_embeddings_class.side_effect = Exception("Embedding API error")
        
        loader = ModelLoader()
        
        with pytest.raises(DocumentPortalException) as exc_info:
            loader.load_embeddings()
        
        assert "Failed to load embedding model" in str(exc_info.value)
