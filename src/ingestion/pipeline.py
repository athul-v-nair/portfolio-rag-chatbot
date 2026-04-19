from langchain_chroma import Chroma
import os,sys

from src.ingestion.loaders import Loader
from src.ingestion.vector_store import VectorStore
from src.utils.logger import logger
from src.ingestion.document_parser import DocumentParser
from src.ingestion.markdown_parser import MarkdownParser
from src.ingestion.chunker import Chunker

from src.utils.constants import DB_PATH, COLLECTION_NAME

def data_ingestion_pipeline(directory: str = 'data/raw', rebuild: bool = False) -> Chroma:
    """
    Execute the end-to-end data ingestion pipeline.

    The pipeline performs the following steps:
        1. Load source documents from the filesystem.
        2. Parse structured sections from files.
        3. Chunk parsed documents into smaller segments suitable for embedding.
        4. Generate embeddings and store them in a Chroma vector database.

    Args:
        pdf_directory (str):
            Directory containing PDF files to be ingested.

        md_directory (str):
            Directory containing Markdown files (reserved for future support).

        txt_directory (str):
            Directory containing text files (reserved for future support).

    Returns:
        Chroma:
            Initialized Chroma vector store containing document embeddings.
    """
    try:
        if not os.path.exists(directory):
            logger.warning(f"Raw data directory not found. Creating: {directory}")
            os.makedirs(directory, exist_ok=True)
        
        logger.info("Starting data ingestion pipeline")
        
        # 1 Load documents
        logger.info("Loading documents")
        file_loader = Loader(directory)
        raw_documents = file_loader.files
        logger.info(f"Total documents Pages loaded: {len(raw_documents)}")

        if not raw_documents:
            logger.warning("No documents found in %s — nothing to do.", raw_documents)
            return
        
        # 2 Parse documents
        logger.info("Parsing the documents")
        
        pdf_docs = [doc for doc in raw_documents if doc.metadata.get("file_type") == "pdf"]
        md_docs = [doc for doc in raw_documents if doc.metadata.get("file_type") == "md"]
        
        parsed_documents = []
        if pdf_docs:
            logger.info("Parsing PDF documents using LLM")
            document_parser = DocumentParser(pdf_docs)
            parsed_documents.extend(document_parser.parsed_documents)
            
        if md_docs:
            logger.info("Parsing Markdown documents using Header Splitter")
            markdown_parser = MarkdownParser(md_docs)
            parsed_documents.extend(markdown_parser.parsed_documents)
            
        logger.info("Parsing Completed")
        logger.info(f"Parsed into {len(parsed_documents)} element(s).")
        logger.info(f"First 5 Documents into {parsed_documents[:5]}.")
        
        # 3 Chunking
        logger.info("Chunking documents")
        chunker = Chunker()
        chunks = chunker.chunk_documents(parsed_documents)
        logger.info("Chunking Completed")
        
        # Save metadata for versioning documents
        logger.info("Saving metadata for version control")
        try:
            from datetime import datetime
            
            metadata_list = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in parsed_documents
            ]
            version_data = {
                "ingested_at": datetime.utcnow().isoformat(),
                "documents": metadata_list
            }
            os.makedirs("data", exist_ok=True)
            with open("data/metadata.json", "w", encoding="utf-8") as f:
                import json
                json.dump(version_data, f, indent=4)
        except Exception as meta_ex:
            logger.warning(f"Failed to save metadata: {meta_ex}")

        # 4 Storing to a vector db after embedding
        logger.info("Creating vector database")
        vector_store = VectorStore(DB_PATH, COLLECTION_NAME)
        vector_store = vector_store.create_vector_db(chunks,rebuild)
        logger.info("Vector database successfully created")

        return vector_store
    
    except Exception as e:
        raise Exception(f"Error in ingestion pipeline: {e}")

if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    data_ingestion_pipeline(rebuild=rebuild)