import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """ Handles ingestion and chunking of financial documents"""

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap
        )

    def load_pdf(self, file_path: str) --> List[Document]:

        # Make sure that the file path exists

        if not os.path.exists(file_path):
            raise FileNotFoundError("Error: The file_path does not exist")


        try:
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            print(f"Successfully loaded {len(docs)} pages")
            return docs

        except exception as e:
            print(f"Failed to load PDF. Error: {e}")


    def process_document(self, documents: List[Document]) --> List[Document]:

        # Check if documents exists before starting the pipeline

        if not documents:
            raise ValueError("No doucments provided for processing")

        print("Chunking Documents")
        chunks = self.text_splitter.split_documents(documents)

        print(f"Created {len(chunks)} overlapping chunks")

        return chunks







