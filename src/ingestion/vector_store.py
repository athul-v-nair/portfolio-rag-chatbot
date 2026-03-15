from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from typing import List

from src.utils.logger import logger
from src.ingestion.embedding import EmbeddingModel
from src.utils.constants import DB_PATH, COLLECTION_NAME

class VectorStore():
    def __init__(self, persist_directory: str = "data/db", collection_name: str = "portfolio_rag"):
        self.persist_directory = persist_directory
        self.collection_name   = collection_name
 
        logger.info(f"Loading embedding model")
        self.embedding_model = EmbeddingModel().model


    def create_vector_db(self, chunks: List[Document], rebuild: bool = False) -> Chroma:
        """
        Create or update the vector database.

        Steps:
            1. Load embedding model
            2. Store document embeddings in Chroma DB
            3. Persist the database locally

        Args:
            chunks (List[Document]):
                Chunked documents ready for embedding.

        Returns:
            Chroma:
                Vector store instance.
        """
        db_exists = (
            os.path.exists(self.persist_directory)
            and bool(os.listdir(self.persist_directory))
            and os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3"))
        )
        
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_model,
            persist_directory=DB_PATH
        )

        if rebuild and db_exists:
            # If rebuild requested → delete and recreate
            logger.info("Rebuilding vector database")
            
            # Re initialize the collection
            vector_store.reset_collection()

            vector_store.add_documents(documents=chunks)

        elif not db_exists:
            # If DB doesn't exist → create it
            logger.info("Vector DB not found. Creating new database")
            logger.info(f"Example chunk: {chunks[0]}")
            vector_store.add_documents(documents=chunks)
        
        else:
            logger.info("Vector DB already exists. Loading existing database")

        return vector_store