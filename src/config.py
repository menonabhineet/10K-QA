import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# DeepSeek Configuration
# The OpenAI Python SDK is fully compatible with DeepSeek's API structure
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-v4-flash"

# Local Embedding Configuration
# all-MiniLM-L6-v2 is fast, runs locally, and is standard for lightweight RAG
EMBEDDING_MODEL = "all-MiniLM-L6-v2"