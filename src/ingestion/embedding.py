from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import Union

from src.utils.logger import logger
from src.utils.constants import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL

class EmbeddingModel():
    """
    Wrapper class responsible for loading the embedding model.

    This abstraction allows the pipeline to switch embedding
    providers without changing other ingestion components.
    """
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self) -> Union[GoogleGenerativeAIEmbeddings]:
        """
        Initialize and return an embeddings model.

        Returns:
            Union[GoogleGenerativeAIEmbeddings]:
                An initialized embeddings model instance.

        Raises:
            Exception:
                Raised if both embedding providers fail to initialize.
        """
        # Try Google embedding model
        try:
            logger.info("Using Google Generative AI Embeddings")

            return GoogleGenerativeAIEmbeddings(
                api_key=GEMINI_API_KEY,
                model=GEMINI_EMBEDDING_MODEL,
                output_dimensionality=768
            )
        
        except Exception as e:
            logger.info("Google embeddings failed. Embedding service unavailable.")
            logger.info("Error in Embedding:", e)
            raise e