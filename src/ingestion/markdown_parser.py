import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

class MarkdownParser:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        
        # Define headers to split on
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5")
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        self.parsed_documents = self.parse_documents()

    def parse_documents(self) -> List[Document]:
        parsed_docs = []
        
        for doc in self.documents:
            file_name = doc.metadata.get("file_name", "unknown.md")
            
            # Clean HTML tags using a simple regex as requested
            clean_text = re.sub(r'<[^>]+>', '', doc.page_content)
            
            # Split the document based on headers
            splits = self.markdown_splitter.split_text(clean_text)
            
            for split in splits:
                # Merge headers into a single section metadata string
                # e.g., "Projects History - Project 1"
                headers = []
                if "Header 1" in split.metadata:
                    headers.append(split.metadata["Header 1"])
                if "Header 2" in split.metadata:
                    headers.append(split.metadata["Header 2"])
                if "Header 3" in split.metadata:
                    headers.append(split.metadata["Header 3"])
                if "Header 4" in split.metadata:
                    headers.append(split.metadata["Header 4"])
                if "Header 5" in split.metadata:
                    headers.append(split.metadata["Header 5"])
                
                section_name = " - ".join(headers) if headers else "General"
                
                # Assign aligned metadata
                new_metadata = doc.metadata.copy()
                new_metadata["section"] = section_name
                # Keep the original headers in metadata for potential advanced filtering
                new_metadata.update(split.metadata)
                
                parsed_docs.append(
                    Document(
                        page_content=split.page_content.strip(),
                        metadata=new_metadata
                    )
                )
                
        return parsed_docs
