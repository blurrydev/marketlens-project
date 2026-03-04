import os
import uuid
import chromadb
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.core import Document

class VectorStore:

    def __init__(self, collection_name: str = "marketlens_docs", persist_dir: str = "./marketlens_db"):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print(f"Initializing ChromaDB client at {self.persist_dir}...")
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path = self.persist_dir)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata = {"description": "Raw insertion for MarketLens"}
        )


    def build_database(self, chunks: List[Document]):

        # Check if chunks are available before starting the pipeline

        if not chunks:
            raise ValueError("Error: No chunks provided for database insertion")

        print(f"Extracting the raw text and calulating embeddings for {len(chunks)} chunks...")

        raw_texts = [chunk.page_content for chunk in chunks]
    
        # Encoding embeddings
        embeddings = self.model.encode(raw_texts, show_progress_bar=True).tolist()

        # Preparing metadata for the insertion in collection
        ids = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            
            page_num = chunk.metadata.get("page", 0)
            metadatas.append({"page_number": page_num})

        print("Executing raw database insertion")

        self.collection.add(
            documents = raw_texts,
            embeddings = embeddings,
            ids = ids,
            metadatas = metadatas
        )

        print(f"Success! Collection now contains {self.collection.count()} items.")


    def search(self, query: str, n_results: int = 3) --> List[str]:

        print(f"Received query: {query}.")
        
        query_embeddings = self.model.encode([query]).tolist()

        print("Searching database for nearest neighbour")
        results = self.collection.query(
            query_embeddings = query_embeddings,
            n_results = n_results
        )

        if not results['documents'] or not results['documents'][0]:
            print("INFO: No matching documents found")
            return []

