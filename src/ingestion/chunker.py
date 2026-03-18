from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters  import RecursiveCharacterTextSplitter
import uuid
from src.utils.logger import logger

class Chunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size:    Target token/character ceiling per chunk.
            chunk_overlap: Overlap between consecutive chunks when a section
                           must be split further.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
 
        # Used only when a single section exceeds chunk_size
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        # Section-level chunks are already done; just add metadata
        section = doc.metadata.get("section", "unknown")
        chunked_docs = []
        for doc in documents:
            new_metadata = doc.metadata.copy()
            new_metadata["chunk_index"] = 0
            new_metadata["chunk_id"] = f"{new_metadata.get('file_name', 'unknown_file')}_section_{section}_{uuid.uuid4().hex[:8]}"
            chunked_docs.append(Document(page_content=doc.page_content, metadata=new_metadata))
        return chunked_docs