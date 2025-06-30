"""
Optimized setup.py for FastAPI + LangChain Chatbot Project
Includes improved error handling, logging, and modern Python practices.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

from setuptools import Command, find_packages, setup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FolderBuilder(Command):
    """Create the FastAPI + LangChain chatbot project structure with boilerplate code."""
    
    description = "Generate project tree and prefill key files with working code."
    user_options = [
        ('force', 'f', 'Force overwrite existing files'),
        ('verbose', 'v', 'Verbose output'),
    ]
    
    def initialize_options(self):
        self.force = False
        self.verbose = False
    
    def finalize_options(self):
        pass
    
    def run(self):
        """Execute the skeleton building process."""
        try:
            if self.verbose:
                logger.setLevel(logging.DEBUG)
            
            logger.info("Starting project skeleton generation...")
            
            # Generate project structure
            self._create_project_structure()
            
            logger.info("✔ Project skeleton with boilerplate code generated successfully!")
            logger.info("Next steps:")
            logger.info("1. Update .env with your actual API keys and database URL")
            logger.info("2. Install dependencies: pip install -r requirements.txt")
            logger.info("3. Set up your PostgreSQL database")
            logger.info("4. Run the application: uvicorn chatbot.main:app --reload")
            
        except Exception as e:
            logger.error(f"Failed to generate project skeleton: {e}")
            sys.exit(1)
    
    def _create_project_structure(self):
        """Create the complete project structure with all necessary files."""
        skeleton = self._get_project_skeleton()
        
        created_files = 0
        skipped_files = 0
        
        for rel_path, content in skeleton.items():
            if self._create_file(rel_path, content):
                created_files += 1
            else:
                skipped_files += 1
        
        logger.info(f"Created {created_files} files, skipped {skipped_files} existing files")
    
    def _create_file(self, rel_path: str, content: str) -> bool:
        """Create a single file with the given content."""
        path = Path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.exists() and not self.force:
            logger.warning(f"File already exists (use --force to overwrite): {rel_path}")
            return False
        
        try:
            path.write_text(content.strip() + "\n", encoding="utf-8")
            logger.debug(f"Created: {rel_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create {rel_path}: {e}")
            return False
    
    def _get_project_skeleton(self) -> Dict[str, str]:
        """Return the complete project skeleton structure."""
        return {
            # Main application files
            "chatbot/__init__.py": self._get_init_content(),
            "chatbot/main.py": self._get_main_content(),
            "chatbot/conversation.py": self._get_conversation_content(),
            "chatbot/database.py": self._get_database_content(),
            "chatbot/models.py": self._get_models_content(),
            "chatbot/config.py": self._get_config_content(),
            "chatbot/exceptions.py": self._get_exceptions_content(),
            
            # Utility files
            "chatbot/utils/__init__.py": "",
            "chatbot/utils/logger.py": self._get_logger_content(),
            
            # API routes
            "chatbot/api/__init__.py": "",
            "chatbot/api/chat.py": self._get_chat_api_content(),
            
            # Configuration files
            ".env": self._get_env_content(),
            ".env.example": self._get_env_example_content(),
            "requirements.txt": self._get_requirements_content(),
            "requirements-dev.txt": self._get_dev_requirements_content(),
            
            # Prompt templates
            "prompts/assistant_prompt.txt": self._get_prompt_content(),
            "prompts/system_prompts.py": self._get_system_prompts_content(),
            
            # Documentation
            "README.md": self._get_readme_content(),
            "docs/setup.md": self._get_setup_docs_content(),
            
            # Docker support
            "Dockerfile": self._get_dockerfile_content(),
            "docker-compose.yml": self._get_docker_compose_content(),
            
            # Testing
            "tests/__init__.py": "",
            "tests/test_conversation.py": self._get_test_content(),
            
            # Git and other configs
            ".gitignore": self._get_gitignore_content(),
            "pyproject.toml": self._get_pyproject_content(),
        }
    
    def _get_init_content(self) -> str:
        return '''"""
FastAPI + LangChain Chatbot Package
A conversational AI application with FastAPI backend and LangChain integration.
"""

__version__ = "0.1.0"
__author__ = "Ismail Abbas"
'''
    
    def _get_main_content(self) -> str:
        return '''"""
Main FastAPI application entry point.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from chatbot.api.chat import chat_router
from chatbot.config import settings
from chatbot.exceptions import ChatbotException
from chatbot.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Conversational Chatbot API",
    description="AI-powered chatbot using LangChain and OpenAI",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api/v1")

@app.exception_handler(ChatbotException)
async def chatbot_exception_handler(request, exc: ChatbotException):
    logger.error(f"Chatbot error: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "type": exc.error_type}
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "chatbot.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
'''
    
    def _get_conversation_content(self) -> str:
        return '''"""
Conversation handling with LangChain integration.
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from chatbot.config import settings
from chatbot.database import get_db
from chatbot.exceptions import ConversationError
from chatbot.models import ConversationHistory
from chatbot.utils.logger import get_logger

try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
except ImportError:
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import BaseMessage, HumanMessage, AIMessage

logger = get_logger(__name__)

class ConversationManager:
    """Manages conversation history and LLM interactions."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=settings.LLM_TEMPERATURE,
            model_name=settings.LLM_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=settings.MAX_TOKENS
        )
    
    def get_bot_response(self, session_id: str, user_input: str) -> str:
        """Generate bot response for user input."""
        try:
            # Fetch conversation history
            history = self._fetch_history(session_id)
            
            # Add user message to history
            history.append(HumanMessage(content=user_input))
            
            # Generate response
            response = self.llm(history)
            
            # Save messages to database
            self._save_message(session_id, "user", user_input)
            self._save_message(session_id, "assistant", response.content)
            
            logger.info(f"Generated response for session {session_id}")
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise ConversationError(f"Failed to generate response: {str(e)}")
    
    def _fetch_history(self, session_id: str) -> List[BaseMessage]:
        """Fetch conversation history from database."""
        try:
            with get_db() as db:
                history_records = (
                    db.query(ConversationHistory)
                    .filter_by(session_id=session_id)
                    .order_by(ConversationHistory.timestamp)
                    .limit(settings.MAX_HISTORY_LENGTH)
                    .all()
                )
            
            messages = []
            for record in history_records:
                if record.role == "user":
                    messages.append(HumanMessage(content=record.message))
                elif record.role == "assistant":
                    messages.append(AIMessage(content=record.message))
            
            return messages
            
        except Exception as e:
            logger.error(f"Error fetching history: {e}")
            return []
    
    def _save_message(self, session_id: str, role: str, message: str) -> None:
        """Save message to database."""
        try:
            with get_db() as db:
                record = ConversationHistory(
                    session_id=session_id,
                    role=role,
                    message=message
                )
                db.add(record)
                db.commit()
                
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            raise ConversationError(f"Failed to save message: {str(e)}")

# Global conversation manager instance
conversation_manager = ConversationManager()

def get_bot_response(session_id: str, user_input: str) -> str:
    """Public interface for getting bot responses."""
    return conversation_manager.get_bot_response(session_id, user_input)
'''
    
    def _get_database_content(self) -> str:
        return '''"""
Database configuration and session management.
"""

from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from chatbot.config import settings
from chatbot.utils.logger import get_logger

logger = get_logger(__name__)

# Database engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.DEBUG
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for models
Base = declarative_base()

@contextmanager
def get_db() -> Session:
    """Database session context manager."""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def create_tables():
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise
'''
    
    def _get_models_content(self) -> str:
        return '''"""
Database models for the chatbot application.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, func, Index
from chatbot.database import Base

class ConversationHistory(Base):
    """Model for storing conversation history."""
    
    __tablename__ = "conversation_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True, nullable=False)
    role = Column(String(50), nullable=False)  # 'user' or 'assistant'
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes for better query performance
    __table_args__ = (
        Index('idx_session_timestamp', 'session_id', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<ConversationHistory(session_id={self.session_id}, role={self.role})>"
'''
    
    def _get_config_content(self) -> str:
        return '''"""
Application configuration using Pydantic settings.
"""

import os
from typing import List
from pydantic import BaseSettings, validator

class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Database Configuration
    DATABASE_URL: str
    
    # OpenAI Configuration
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-4"
    LLM_TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1000
    MAX_HISTORY_LENGTH: int = 50
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    @validator('DATABASE_URL')
    def validate_database_url(cls, v):
        if not v:
            raise ValueError('DATABASE_URL is required')
        return v
    
    @validator('OPENAI_API_KEY')
    def validate_openai_key(cls, v):
        if not v:
            raise ValueError('OPENAI_API_KEY is required')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
'''
    
    def _get_exceptions_content(self) -> str:
        return '''"""
Custom exceptions for the chatbot application.
"""

class ChatbotException(Exception):
    """Base exception for chatbot errors."""
    
    def __init__(self, message: str, status_code: int = 500, error_type: str = "CHATBOT_ERROR"):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(self.message)

class ConversationError(ChatbotException):
    """Exception for conversation-related errors."""
    
    def __init__(self, message: str):
        super().__init__(message, status_code=400, error_type="CONVERSATION_ERROR")

class DatabaseError(ChatbotException):
    """Exception for database-related errors."""
    
    def __init__(self, message: str):
        super().__init__(message, status_code=500, error_type="DATABASE_ERROR")

class LLMError(ChatbotException):
    """Exception for LLM-related errors."""
    
    def __init__(self, message: str):
        super().__init__(message, status_code=502, error_type="LLM_ERROR")
'''
    
    def _get_logger_content(self) -> str:
        return '''"""
Logging configuration for the application.
"""

import logging
from chatbot.config import settings

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    return logger
'''
    
    def _get_chat_api_content(self) -> str:
        return '''"""
Chat API endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chatbot.conversation import get_bot_response
from chatbot.exceptions import ConversationError
from chatbot.utils.logger import get_logger

logger = get_logger(__name__)

chat_router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., min_length=1, max_length=1000, description="User message")

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    reply: str = Field(..., description="Bot response")
    session_id: str = Field(..., description="Session identifier")

@chat_router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the chatbot and get a response.
    
    - **session_id**: Unique identifier for the conversation session
    - **message**: The user's message to the chatbot
    """
    try:
        reply = get_bot_response(request.session_id, request.message)
        return ChatResponse(reply=reply, session_id=request.session_id)
    
    except ConversationError as e:
        logger.error(f"Conversation error: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
'''
    
    def _get_env_content(self) -> str:
        return '''# Environment Configuration
# Copy this file to .env and fill in your actual values

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/chatbot_db

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# API Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Logging
LOG_LEVEL=INFO
'''
    
    def _get_env_example_content(self) -> str:
        return '''# Example environment configuration
# Copy this to .env and update with your values

DATABASE_URL=postgresql://user:password@localhost:5432/chatbot_db
OPENAI_API_KEY=sk-your-openai-api-key-here
HOST=0.0.0.0
PORT=8000
DEBUG=false
LOG_LEVEL=INFO
'''
    
    def _get_requirements_content(self) -> str:
        return '''# Core dependencies
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
python-dotenv>=1.0.0

# LangChain and AI
langchain>=0.0.350
langchain-openai>=0.0.5
openai>=1.0.0

# Utilities
python-multipart>=0.0.6
'''
    
    def _get_dev_requirements_content(self) -> str:
        return '''# Development dependencies
-r requirements.txt

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
httpx>=0.25.0

# Code quality
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.6.0

# Development tools
pre-commit>=3.5.0
'''
    
    def _get_prompt_content(self) -> str:
        return '''You are Ava, a friendly and helpful AI assistant. Your job is to help users with various tasks including:

- Answering questions and providing information
- Helping with problem-solving
- Offering suggestions and recommendations
- Engaging in natural conversation

Guidelines:
- Be helpful, accurate, and concise
- Maintain a warm and professional tone
- Ask clarifying questions when needed
- Admit when you don't know something
- Respect user privacy and boundaries
'''
    
    def _get_system_prompts_content(self) -> str:
        return '''"""
System prompt templates for different conversation contexts.
"""

DEFAULT_PROMPT = """
You are Ava, a friendly and helpful AI assistant. Respond naturally and helpfully to user queries.
"""

PROFESSIONAL_PROMPT = """
You are a professional AI assistant. Provide accurate, concise, and business-appropriate responses.
"""

CASUAL_PROMPT = """
You are a casual, friendly AI companion. Keep the conversation light and engaging.
"""

def get_prompt(context: str = "default") -> str:
    """Get system prompt based on context."""
    prompts = {
        "default": DEFAULT_PROMPT,
        "professional": PROFESSIONAL_PROMPT,
        "casual": CASUAL_PROMPT,
    }
    return prompts.get(context, DEFAULT_PROMPT)
'''
    
    def _get_readme_content(self) -> str:
        return '''# Conversational Chatbot

A FastAPI-based conversational AI chatbot using LangChain and OpenAI GPT models.

## Features

- FastAPI REST API
- LangChain integration
- PostgreSQL database for conversation history
- Session-based conversations
- Docker support
- Comprehensive error handling

## Quick Start

1. **Clone and setup**
   ```bash
   git clone <your-repo>
   cd conversational-chatbot
   python setup.py build_skeleton
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Setup database**
   ```bash
   # Create PostgreSQL database
   createdb chatbot_db
   ```

5. **Run the application**
   ```bash
   uvicorn chatbot.main:app --reload
   ```

6. **Test the API**
   - Visit http://localhost:8000/docs for API documentation
   - Send POST request to http://localhost:8000/api/v1/chat/

## API Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/chat/",
    json={
        "session_id": "user123",
        "message": "Hello, how are you?"
    }
)
print(response.json())
```

## Development

See `docs/setup.md` for detailed development setup instructions.

## License

MIT License
'''
    
    def _get_setup_docs_content(self) -> str:
        return '''# Development Setup Guide

## Prerequisites

- Python 3.11+
- PostgreSQL 12+
- OpenAI API key

## Installation

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\\Scripts\\activate  # Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Database setup**
   ```bash
   createdb chatbot_db
   ```

4. **Environment configuration**
   ```bash
   cp .env.example .env
   # Update .env with your settings
   ```

## Running the Application

### Development mode
```bash
uvicorn chatbot.main:app --reload --port 8000
```

### Production mode
```bash
uvicorn chatbot.main:app --host 0.0.0.0 --port 8000
```

### Using Docker
```bash
docker-compose up --build
```

## Testing

```bash
pytest tests/
```

## Code Quality

```bash
# Format code
black chatbot/
isort chatbot/

# Lint code
flake8 chatbot/
mypy chatbot/
```
'''
    
    def _get_dockerfile_content(self) -> str:
        return '''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    def _get_docker_compose_content(self) -> str:
        return '''version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/chatbot_db
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
    volumes:
      - .:/app
    command: uvicorn chatbot.main:app --host 0.0.0.0 --port 8000 --reload

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=chatbot_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
'''
    
    def _get_test_content(self) -> str:
        return '''"""
Tests for conversation functionality.
"""

import pytest
from unittest.mock import Mock, patch
from chatbot.conversation import ConversationManager

class TestConversationManager:
    """Test conversation manager functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.conversation_manager = ConversationManager()
    
    @patch('chatbot.conversation.get_db')
    @patch('chatbot.conversation.ChatOpenAI')
    def test_get_bot_response(self, mock_chat_openai, mock_get_db):
        """Test bot response generation."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Hello! How can I help you?"
        mock_chat_openai.return_value.return_value = mock_response
        
        # Mock database
        mock_db = Mock()
        mock_get_db.return_value.__enter__.return_value = mock_db
        mock_db.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = []
        
        # Test response generation
        response = self.conversation_manager.get_bot_response("test_session", "Hello")
        
        assert response == "Hello! How can I help you?"
        assert mock_db.add.called
        assert mock_db.commit.called
'''
    
    def _get_gitignore_content(self) -> str:
        return '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment variables
.env
.env.local

# Database
*.db
*.sqlite3

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Docker
.dockerignore
'''
    
    def _get_pyproject_content(self) -> str:
        return '''[tool.black]
line-length = 88
target-version = ['py311']
include = '\\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
'''


# Setup configuration
setup(
    name="conversational-chatbot",
    version="0.1.0",
    description="A conversational AI chatbot using FastAPI, LangChain, and OpenAI",
    long_description=open("README.md").read() if Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="Ismail Abbas",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/conversational-chatbot",
    python_requires=">=3.11",
    packages=find_packages(exclude=["tests*", "docs*"]),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "python-dotenv>=1.0.0",
        "langchain>=0.0.350",
        "langchain-openai>=0.0.5",
        "openai>=1.0.0",
        "python-multipart>=0.0.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.25.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.6.0",
            "pre-commit>=3.5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "chatbot-server=chatbot.main:main",
        ]
    },
    cmdclass={
        "build_skeleton": FolderBuilder,
    },
     classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Framework :: FastAPI",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="chatbot, fastapi, langchain, openai, conversational-ai",
    project_urls={
        "Documentation": "https://github.com/yourusername/conversational-chatbot/wiki",
        "Source": "https://github.com/yourusername/conversational-chatbot",
        "Tracker": "https://github.com/yourusername/conversational-chatbot/issues",
    },
)

# Usage instructions
if __name__ == "__main__":
    print("""
    🤖 Conversational Chatbot Setup
    
    To generate the project skeleton, run:
        python setup.py build_skeleton
    
    For verbose output:
        python setup.py build_skeleton --verbose
    
    To force overwrite existing files:
        python setup.py build_skeleton --force
    
    After generation:
    1. Update .env with your API keys and database URL
    2. Install dependencies: pip install -r requirements.txt
    3. Set up PostgreSQL database
    4. Run: uvicorn chatbot.main:app --reload
    """)
    
    # Show help for build_skeleton command
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build_skeleton":
        if "--help" in sys.argv or "-h" in sys.argv:
            print("""
            build_skeleton command options:
            
            --force, -f     Force overwrite existing files
            --verbose, -v   Show detailed output during generation
            --help, -h      Show this help message
            
            Examples:
                python setup.py build_skeleton
                python setup.py build_skeleton --verbose
                python setup.py build_skeleton --force --verbose
            """)
        elif "--force" in sys.argv or "-f" in sys.argv:
            print("⚠️  Force option enabled: Overwriting existing files.")
        elif "--verbose" in sys.argv or "-v" in sys.argv:
            print("🔍 Verbose output enabled.")