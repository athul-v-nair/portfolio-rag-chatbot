from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters  import RecursiveCharacterTextSplitter
import uuid
from src.utils.logger import logger

class Chunker:
    """
    Section-aware chunker for PDF/MD documents and paragraph-based chunker for TXT.
    
    - PDF/MD: Groups elements by section, then splits oversized sections while
              preserving the section metadata on every resulting chunk.
    - TXT:    Each paragraph from the parser is a chunk already; oversized
              paragraphs are split with a simple recursive splitter.
    """
 
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 90):
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
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Dispatch to the correct chunking strategy based on file_type metadata.
 
        Args:
            documents: Parsed Document objects produced by DocumentParser.
 
        Returns:
            List of chunked Document objects ready for embedding.
        """
        pdf_md_docs = [d for d in documents if d.metadata.get("file_type") in ("pdf", "md")]
        txt_docs    = [d for d in documents if d.metadata.get("file_type") == "txt"]

        chunks: List[Document] = []
        chunks.extend(self._chunk_structured(pdf_md_docs))
        chunks.extend(self._chunk_txt(txt_docs))
        
        # Assign ids to each chunk
        chunks_id_metadata = self._assign_chunk_ids(chunks)

        logger.info(f"Chunking complete — input docs sections: {len(documents)}, output chunks: {len(chunks_id_metadata)}")
        
        return chunks_id_metadata
    
    def _chunk_structured(self, documents: List[Document]) -> List[Document]:
        """
        Section-aware chunking for PDF and MD documents.
 
        Strategy:
          1. Group consecutive elements that share the same (file_name, section).
          2. Concatenate their text with double newlines.
          3. If the combined text fits within chunk_size, emit one chunk.
          4. Otherwise split with RecursiveCharacterTextSplitter and attach the
             section metadata to every sub-chunk.
        """
        if not documents:
            return []
 
        # Group by (file_name, section) preserving document order
        groups: Dict[tuple, List[Document]] = {}
        order: List[tuple] = []
 
        for doc in documents:
            key = (
                doc.metadata.get("file_name", "unknown"),
                doc.metadata.get("section", "general"),
            )
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(doc)

        chunks: List[Document] = []
        chunk_index = 0
 
        for key in order:
            group = groups[key]
            file_name, section = key
 
            # Build a representative metadata dict from the first element
            base_meta = {**group[0].metadata}
 
            combined_text = "\n\n".join(doc.page_content for doc in group)
 
            if len(combined_text) <= self.chunk_size:
                # Section fits in one chunk
                chunks.append(
                    Document(
                        page_content=combined_text,
                        metadata={
                            **base_meta,
                            "chunk_index": chunk_index,
                            "chunk_type": "section_unsplit",
                            "file_name": file_name,
                            "section": section
                        },
                    )
                )
                chunk_index += 1
            else:
                # Section too large — split while keeping section metadata
                sub_texts = self._splitter.split_text(combined_text)
                for sub in sub_texts:
                    chunks.append(
                        Document(
                            page_content=sub,
                            metadata={
                                **base_meta,
                                "chunk_index": chunk_index,
                                "chunk_type": "section_split",
                                "file_name": file_name,
                                "section": section
                            },
                        )
                    )
                    chunk_index += 1

        return chunks
 
    def _chunk_txt(self, documents: List[Document]) -> List[Document]:
        """
        Paragraph-based chunking for TXT documents.
 
        Each element from the parser is already a paragraph.  Paragraphs that
        exceed chunk_size are further split; all others are kept as-is.
        """
        if not documents:
            return []
 
        chunks: List[Document] = []
        chunk_index = 0
 
        for doc in documents:
            if len(doc.page_content) <= self.chunk_size:
                chunks.append(
                    Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            "chunk_index": chunk_index,
                            "chunk_type": "paragraph",
                        },
                    )
                )
                chunk_index += 1
            else:
                sub_texts = self._splitter.split_text(doc.page_content)
                for sub in sub_texts:
                    chunks.append(
                        Document(
                            page_content=sub,
                            metadata={
                                **doc.metadata,
                                "chunk_index": chunk_index,
                                "chunk_type": "paragraph_split",
                            },
                        )
                    )
                    chunk_index += 1

        return chunks
    
    def _assign_chunk_ids(self, chunks: List[Document]) -> List[Document]:
        for idx, chunk in enumerate(chunks):
            file_name = chunk.metadata.get('file_name', 'unknown_source')
            page = chunk.metadata.get('page_number', 0)

            unique_id = f"{file_name}_p{page}_chunk{idx}_{uuid.uuid4().hex[:8]}"

            chunk.metadata['chunk_id'] = unique_id
        
        return chunks