"""
Configuration settings for the RAG Platform
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Text Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))        # Default: 800 (antes estaba en 2000)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))  # Default: 100 (~12% de solape)

# Document Processing
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 50))      # Maximum file size in MB
MAX_PAGES = int(os.getenv("MAX_PAGES", 1000))                 # Maximum pages per document

# LLM Configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.0))

# Retrieval Configuration
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 6))      # Number of chunks to retrieve

# API Configuration
API_TIMEOUT = int(os.getenv("API_TIMEOUT", 120))              # Timeout in seconds