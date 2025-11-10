# Multi-Document Chat - LLMOps Project

A production-ready RAG (Retrieval-Augmented Generation) system for multi-document question-answering using LangChain, FAISS, and multiple LLM providers.

## Features

- **Multi-Document Ingestion**: Support for PDF, DOCX, and TXT files
- **Vector Search**: FAISS-based semantic search with MMR (Maximal Marginal Relevance)
- **Multiple LLM Providers**: Support for Google Gemini, Groq, and OpenAI
- **Conversational RAG**: Context-aware multi-turn conversations
- **Session Management**: Isolated document sessions for different users/conversations
- **Production-Ready**: Structured logging, error handling, and configuration management

## Tech Stack

- **Python 3.13+**
- **LangChain**: LLM orchestration and document processing
- **FAISS**: Vector similarity search
- **FastAPI**: Web framework (for API endpoints)
- **Pydantic**: Data validation
- **Structlog**: Structured logging

## Setup

1. Clone the repository
2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```
3. Create a `.env` file with required API keys:
   ```bash
   GOOGLE_API_KEY=your_google_api_key
   GROQ_API_KEY=your_groq_api_key
   ```
4. Run tests:
   ```bash
   uv run python tests.py
   ```

## Project Structure

```
multi_doc_chat/
├── config/          # Configuration files
├── exception/       # Custom exception handling
├── logger/          # Structured logging
├── model/           # Pydantic models
├── prompts/         # Prompt templates
├── src/
│   ├── document_ingestion/  # Document processing
│   └── document_chat/       # RAG implementation
└── utils/           # Utility functions
```

## License

MIT

