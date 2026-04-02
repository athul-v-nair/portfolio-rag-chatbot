from langchain_chroma import Chroma

from src.utils.constants import DB_PATH, COLLECTION_NAME
from src.ingestion.embedding import EmbeddingModel
from src.utils.logger import logger

_VECTOR_DB = None

def get_vector_db() -> Chroma:
    """
    Initialize or return the singleton Chroma vector DB instance.
    """
    global _VECTOR_DB
    if _VECTOR_DB is None:
        logger.info("Initializing Vector Database Instance...")
        embedding_model = EmbeddingModel().model
        _VECTOR_DB = Chroma(
            persist_directory=DB_PATH,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model
        )
    return _VECTOR_DB

def search_vector_db(query: str, top_k: int = 5) -> list:
    """
    Search vector database for similar documents.
    Args:
        query (str):
            The input query string used to search for similar documents.

        top_k (int, optional):
            The number of top similar results to retrieve. Defaults to 5.

    Returns:
        list:
            A list of dictionaries, where each dictionary contains:
                - content (str): The text content of the matched document.
                - metadata (dict): Metadata associated with the document.
                - section (dict): Duplicate of metadata (can be customized if needed).
                - score (float): Similarity score (lower typically means more similar).

    Raises:
        Exception:
            Propagates any exceptions raised during embedding model loading,
            database initialization, or similarity search.

    """
    vector_db = get_vector_db()

    results = vector_db.similarity_search_with_score(query,top_k)
    
    documents = []
    for doc, score in results:
        documents.append(
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "section": doc.metadata,
                "score": score
            }
        )

    # Sort by the score (second element of the tuple)
    results_sorted = sorted(results, key=lambda x: x[1])
    logger.info(f"Sorted Results: \n{[{'content': doc.page_content, 'metadata': doc.metadata, 'score': score} for doc, score in results_sorted]}")
    
    return documents