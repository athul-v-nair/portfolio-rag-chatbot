import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY=os.getenv('GEMINI_API_KEY')
HUGGINGFACE_API_KEY=os.getenv('HUGGINGFACE_API_KEY')

# Embedding models
GEMINI_EMBEDDING_MODEL="gemini-embedding-2-preview"
BACKUP_EMBEDDING_MODEL="all-MiniLM-L6-v2"
BACKUP_EMBEDDING_MODEL_REPO_ID="sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME=os.getenv('COLLECTION_NAME')
DB_PATH=os.getenv('DB_PATH')

DOCS_PATH=os.getenv('DOCS_PATH')