from typing import List, Union
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.md import partition_md
from unstructured.partition.text import partition_text

class DocumentParser:
    def __init__(self, documents: List[Document]):
        self.all_documents: List[Document] = []
        self._parse_files(documents)

    def _parse_files(self, documents: List[Document]):
        for doc in documents:
            parsed_docs = []
            file_type = doc.metadata.get('file_type')
            if file_type == 'pdf':
                parsed_docs = self.parse_pdf(doc.metadata.get('source') , doc.metadata.get('file_name'), file_type)
            elif file_type == 'md':
                parsed_docs = self.parse_markdown(doc.metadata.get('source'), doc.metadata.get('file_name'), file_type)
            elif file_type == 'txt':
                parsed_docs = self.parse_text(doc.metadata.get('source') , doc.metadata.get('file_name'), file_type)

            self.all_documents.extend(parsed_docs)

    def parse_pdf(self, source_file: str, file_name: str, file_type:str ='pdf') -> List[Document]:
        """
        Parse a PDF file and return LangChain Documents.
        """
        try:
            elements = partition_pdf(filename=source_file, strategy="auto", languages=["english"])
            try:
                document_data = []
                current_section = "general"

                for element in elements:
                    category = element.category
                    text = element.text
                    if not text:
                        continue

                    if category == "Title":
                        current_section = text.lower()
                    
                    document_data.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": source_file,
                                "section": current_section,
                                "element_type": category,
                                "page_number": getattr(element.metadata, "page_number", 0),
                                "file_name": file_name,
                                "file_type": file_type
                            }
                        )
                    )
                return document_data
            
            except Exception as e:
                raise Exception(f"Error converting elements to documents: {e}")
            
        except Exception as e:
            raise Exception(f"Error parsing PDF {file_name}: {e}")
        
    def parse_markdown(self, source_file: str, file_name: str, file_type:str ='md') -> List[Document]:
        try:
            elements = partition_md(filename=source_file)
            try:
                document_data = []
                current_section = "general"

                for element in elements:
                    category = element.category
                    text = element.text
                    if not text:
                        continue

                    if category == "Title":
                        current_section = text.lower()
                    
                    document_data.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": source_file,
                                "section": current_section,
                                "element_type": category,
                                "file_name": file_name,
                                "file_type": file_type
                            }
                        )
                    )
                return document_data
            
            except Exception as e:
                raise Exception(f"Error converting elements to documents: {e}")
            
        except Exception as e:
            raise Exception(f"Error parsing Markdown {file_name}: {e}")
        
    def parse_text(self, source_file: str, file_name: str, file_type:str ='txt') -> List[Document]:
        """
        Parse a Text file and return LangChain Documents.
        """
        try:
            document_data = []
            current_section = "General Text"

            # Read file
            with open(source_file, "r", encoding="utf-8") as f:
                text = f.read()

            # Split by double newlines into paragraphs
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            for idx, para in enumerate(paragraphs):
                document_data.append(
                    Document(
                        page_content=para,
                        metadata={
                            "source": source_file,
                            "section": current_section,
                            "element_type": "Paragraph",
                            "paragraph_index": idx,
                            "file_name": file_name,
                            "file_type": file_type
                        }
                    )
                )

            return document_data
            
        except Exception as e:
            raise Exception(f"Error parsing PDF {file_name}: {e}")