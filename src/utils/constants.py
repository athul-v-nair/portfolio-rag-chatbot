import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY=os.getenv('GEMINI_API_KEY')
HUGGINGFACE_API_KEY=os.getenv('HUGGINGFACE_API_KEY')

# Embedding models
GEMINI_EMBEDDING_MODEL=os.getenv('GEMINI_EMBEDDING_MODEL', "gemini-embedding-2-preview")
BACKUP_EMBEDDING_MODEL=os.getenv('BACKUP_EMBEDDING_MODEL', "all-MiniLM-L6-v2")
BACKUP_EMBEDDING_MODEL_REPO_ID=os.getenv('BACKUP_EMBEDDING_MODEL_REPO_ID', "sentence-transformers/all-MiniLM-L6-v2")

# Text generation models
GEMINI_TEXT_GENERATION_MODEL=os.getenv('GEMINI_TEXT_GENERATION_MODEL', "gemini-2.5-flash")

# Vector DB
COLLECTION_NAME=os.getenv('COLLECTION_NAME', "portfolio_rag")
DB_PATH=os.getenv('DB_PATH', "data/db")

DOCS_PATH=os.getenv('DOCS_PATH', "data/raw")

# Generation 
GENERATION_TEMPERATURE: float = float(os.getenv("GEN_TEMPERATURE", "0.2"))
GENERATION_MAX_TOKENS: int = int(os.getenv("GEN_MAX_TOKENS", "512"))
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))
SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.45"))