from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredMarkdownLoader 
from pathlib import Path
from langchain_core.documents import Document
from typing import List

def load_pdf(pdf_directory: str) -> List[Document]:
    """
    Load all PDFs from a directory.

    Returns:
        List[Document]: LangChain Document objects with metadata
    """
    try:
        loader = DirectoryLoader(
            pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        pdf_documents = loader.load()

        # Adding some more meta data
        for doc in pdf_documents:
            doc.metadata.update({"file_name": Path(doc.metadata.get("source")).name})
            doc.metadata.update({"file_type": "pdf"})

        print("Total PDF Document Pages loaded: ", len(pdf_documents))
        return pdf_documents

    except Exception as e:
        raise Exception(f"An error has occurred in load pdf: {e}")

def load_md(md_directory: str) -> List[Document]:
    """
    Load all Markdown(md) files from a directory.

    Returns:
        List[Document]: LangChain Document objects with metadata
    """
    try:
        loader = DirectoryLoader(
            md_directory,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader
        )
        md_documents = loader.load()
        
        # Adding some more meta data
        for doc in md_documents:
            doc.metadata.update({"file_name": Path(doc.metadata.get("source")).name})
            doc.metadata.update({"file_type": "md"})
        
        print("Total Markdown Document Pages loaded in load md: ", len(md_documents))
        return md_documents
    
    except Exception as e:
        raise Exception(f"An error has occurred: {e}")
    

# load_md('data/md')
load_pdf('data/pdf')