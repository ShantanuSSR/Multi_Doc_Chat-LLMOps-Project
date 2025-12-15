"""
Unit tests for Pydantic models.
"""
import pytest
from pydantic import ValidationError

from multi_doc_chat.model.models import (
    ChatAnswer,
    PromptType,
    UploadResponse,
    ChatRequest,
    ChatResponse
)


class TestChatAnswer:
    """Test ChatAnswer model."""

    def test_chat_answer_valid(self):
        """Test valid chat answer creation."""
        answer = ChatAnswer(answer="This is a valid answer.")
        
        assert answer.answer == "This is a valid answer."

    def test_chat_answer_min_length(self):
        """Test that chat answer enforces minimum length."""
        with pytest.raises(ValidationError) as exc_info:
            ChatAnswer(answer="")
        
        errors = exc_info.value.errors()
        assert any("min_length" in str(error) for error in errors)

    def test_chat_answer_max_length(self):
        """Test that chat answer enforces maximum length."""
        long_answer = "a" * 4097  # Exceeds max_length of 4096
        
        with pytest.raises(ValidationError) as exc_info:
            ChatAnswer(answer=long_answer)
        
        errors = exc_info.value.errors()
        assert any("max_length" in str(error) for error in errors)

    def test_chat_answer_boundary_values(self):
        """Test chat answer with boundary length values."""
        # Minimum length (1 character)
        answer_min = ChatAnswer(answer="a")
        assert len(answer_min.answer) == 1
        
        # Maximum length (4096 characters)
        answer_max = ChatAnswer(answer="a" * 4096)
        assert len(answer_max.answer) == 4096


class TestPromptType:
    """Test PromptType enum."""

    def test_prompt_type_values(self):
        """Test that PromptType has correct values."""
        assert PromptType.CONTEXTUALIZE_QUESTION.value == "contextualize_question"
        assert PromptType.CONTEXT_QA.value == "context_qa"

    def test_prompt_type_enum_comparison(self):
        """Test PromptType enum comparison."""
        assert PromptType.CONTEXTUALIZE_QUESTION == PromptType.CONTEXTUALIZE_QUESTION
        assert PromptType.CONTEXTUALIZE_QUESTION != PromptType.CONTEXT_QA


class TestUploadResponse:
    """Test UploadResponse model."""

    def test_upload_response_valid(self):
        """Test valid upload response creation."""
        response = UploadResponse(
            session_id="test_session_123",
            indexed=True,
            message="Files uploaded successfully"
        )
        
        assert response.session_id == "test_session_123"
        assert response.indexed is True
        assert response.message == "Files uploaded successfully"

    def test_upload_response_optional_message(self):
        """Test upload response with optional message."""
        response = UploadResponse(
            session_id="test_session_123",
            indexed=False
        )
        
        assert response.message is None

    def test_upload_response_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            UploadResponse(indexed=True)  # Missing session_id


class TestChatRequest:
    """Test ChatRequest model."""

    def test_chat_request_valid(self):
        """Test valid chat request creation."""
        request = ChatRequest(
            session_id="test_session_123",
            message="What is machine learning?"
        )
        
        assert request.session_id == "test_session_123"
        assert request.message == "What is machine learning?"

    def test_chat_request_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            ChatRequest(session_id="test")  # Missing message
        
        with pytest.raises(ValidationError):
            ChatRequest(message="Test message")  # Missing session_id


class TestChatResponse:
    """Test ChatResponse model."""

    def test_chat_response_valid(self):
        """Test valid chat response creation."""
        response = ChatResponse(
            answer="Machine learning is a subset of artificial intelligence."
        )
        
        assert "Machine learning" in response.answer

    def test_chat_response_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            ChatResponse()  # Missing answer
