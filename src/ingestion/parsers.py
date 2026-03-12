import re
from typing import List
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element

# FOR TESTING
from src.ingestion.loaders import load_pdf

def parse_pdf_sections(documents: List[Document]) -> List[Document]:
    try:
        all_documents = list()
        for doc in documents:
            source_file = doc.metadata.get('source')
            elements = partition_pdf(
                filename=source_file,
                strategy="auto", 
                languages=["english"]
            )
        
            parsed_docs = convert_partition_to_document(elements, source_file)

            all_documents.extend(parsed_docs)
        
        return all_documents
    except Exception as e:
        raise Exception(f"An error has occurred in partition: {e}")

def convert_partition_to_document(elements: list[Element], source_file: str) -> List[Document]:
    try:
        document_data = []
        current_section = "general"
        for element in elements:
            category = element.category
            text = element.text

            # skipping newlines and whitespaces
            if not text:
                continue

            # Checking for the heading blocks
            if category == "Title":
                current_section = text.lower()

            document_data.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_file,
                        "section": current_section,
                        "element_type": category,
                        "page_number": element.metadata.page_number
                    }
                )
            )
            print("Document Data at each iter", document_data)
        return document_data
    except Exception as e:
        raise Exception(f"An error has occurred in conversion to document: {e}")
        