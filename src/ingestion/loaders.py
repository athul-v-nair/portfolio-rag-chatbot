from langchain_community.document_loaders import PyPDFLoader, TextLoader 
from pathlib import Path
from langchain_core.documents import Document
from typing import List
import os

from src.utils.logger import logger

class Loader():
    def __init__(self, directory: str):
        """
        Initialize loader with a directory containing files to ingest.
        
        Args:
            directory (str): Directory path with files.
        """
        self.directory = directory
        self.files = self.load_file()

    def load_file(self):
        """
        Detect file types in the directory and load them with the appropriate loader.
        """
        documents = []

        for file_name in os.listdir(self.directory):
            file_path = os.path.join(self.directory,file_name)
            logger.info(f"file path: {file_path}")
            
            # Skip directories
            if not os.path.isfile(file_path):
                continue

            # Detect file type and call appropriate loader
            if file_name.lower().endswith(".pdf"):
                documents.extend(self.load_pdf(file_path, file_type='pdf'))

            elif file_name.lower().endswith(".md"):
                documents.extend(self.load_md(file_path, file_type='md'))

            elif file_name.lower().endswith(".txt"):
                documents.extend(self.load_txt(file_path, file_type='txt'))

        return documents

    def load_pdf(self, file_path: str, file_type: str) -> List[Document]:
        """
        Load all PDFs from a directory.

        Args:
            file_path (str): File path of pdf files containing files to be ingested.
            file_type (str): File type which is equal to pdf.

        Returns:
            List[Document]: LangChain Document objects with metadata
        """
        try:
            loader = PyPDFLoader(file_path)
            pdf_documents = loader.load()

            # Adding some more metadata
            for doc in pdf_documents:
                doc.metadata.update({"file_name": Path(doc.metadata.get("source")).name})
                doc.metadata.update({"file_type": file_type})

            return pdf_documents

        except Exception as e:
            raise Exception(f"An error has occurred in load pdf: {e}")

    def load_md(self, file_path: str, file_type: str) -> List[Document]:
        """
        Load all Markdown(md) files from a directory.

        Args:
            file_path (str): File path of md files containing files to be ingested.
            file_type (str): File type which is equal to  md.

        Returns:
            List[Document]: LangChain Document objects with metadata
        """
        try:
            loader = TextLoader(file_path)
            md_documents = loader.load()
            
            # Adding some more metadata
            for doc in md_documents:
                doc.metadata.update({"file_name": Path(doc.metadata.get("source")).name})
                doc.metadata.update({"file_type": file_type})

            return md_documents
        
        except Exception as e:
            raise Exception(f"An error has occurred: {e}")
        
    def load_txt(self, file_path: str, file_type: str):
        """
        Load all Text Document(txt) files from a directory.

        Args:
            file_path (str): File path of txt files containing files to be ingested.
            file_type (str): File type which is equal to txt.

        Returns:
            List[Document]: LangChain Document objects with metadata
        """
        try:
            loader = TextLoader(file_path)
            txt_documents = loader.load()
            
            # Adding some more meta data
            for doc in txt_documents:
                doc.metadata.update({"file_name": Path(doc.metadata.get("source")).name})
                doc.metadata.update({"file_type": file_type})

            return txt_documents
        
        except Exception as e:
            raise Exception(f"An error has occurred: {e}")