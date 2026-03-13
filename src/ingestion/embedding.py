from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from typing import Union

from src.utils.logger import logger
from src.utils.constants import GEMINI_API_KEY, HUGGINGFACE_API_KEY, GEMINI_EMBEDDING_MODEL, BACKUP_EMBEDDING_MODEL_REPO_ID

class EmbeddingModel():
    """
    Wrapper class responsible for loading the embedding model.

    Priority:
        1. Google Gemini Embeddings
        2. HuggingFace sentence-transformers fallback

    This abstraction allows the pipeline to switch embedding
    providers without changing other ingestion components.
    """
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self) -> Union[GoogleGenerativeAIEmbeddings,HuggingFaceEndpointEmbeddings]:
        """
        Initialize and return an embeddings model.

        Returns:
            Union[GoogleGenerativeAIEmbeddings, HuggingFaceEndpointEmbeddings]:
                An initialized embeddings model instance.

        Raises:
            Exception:
                Raised if both embedding providers fail to initialize.
        """
        # Try Google embeddings first
        try:
            logger.info("Using Google Generative AI Embeddings")

            return GoogleGenerativeAIEmbeddings(
                api_key=GEMINI_API_KEY,
                model=GEMINI_EMBEDDING_MODEL,
                output_dimensionality=768
            )
        
        except Exception as e:
            logger.info("Google embeddings failed. Falling back to HuggingFace.")
            logger.info("Error:", e)
            return HuggingFaceEndpointEmbeddings(
                repo_id=BACKUP_EMBEDDING_MODEL_REPO_ID,
                huggingfacehub_api_token=HUGGINGFACE_API_KEY
            )