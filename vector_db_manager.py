import os
import json
from typing import List, Dict, Any, Optional, Union
import numpy as np

# Import embedding models and vector store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

class VectorDatabaseManager:
    """Manager for vector database operations."""
    
    def __init__(self, 
                 persist_directory: str = "vectorstore",
                 embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initialize the vector database manager.
        
        Args:
            persist_directory: Directory to store the vector database
            embedding_model_name: Name of the HuggingFace embedding model to use
        """
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name
        )
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize or load vector store
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the vector store or load if it exists."""
        if os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3")):
            # Load existing vector store
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model
            )
            print(f"Loaded existing vector store from {self.persist_directory}")
        else:
            # Initialize empty vector store
            self.vectorstore = None
            print("No existing vector store found. Will create new one when adding documents.")
    
    def _convert_to_documents(self, data: List[Dict[str, Any]]) -> List[Document]:
        """Convert processed data format to Langchain Document objects.
        
        Args:
            data: List of processed documents
            
        Returns:
            List of Langchain Document objects
        """
        documents = []
        
        for item in data:
            doc = Document(
                page_content=item["content"],
                metadata=item["metadata"]
            )
            documents.append(doc)
        
        return documents
    
    def create_from_processed_data(self, processed_data_path: str):
        """Create vector store from processed data file.
        
        Args:
            processed_data_path: Path to processed data JSON file
        """
        # Load the processed data
        with open(processed_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to documents
        documents = self._convert_to_documents(data)
        
        # Create and persist vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        
        # Persist to disk
        self.vectorstore.persist()
        print(f"Created vector store with {len(documents)} documents")
    
    def add_documents(self, processed_data_path: str):
        """Add documents to existing vector store.
        
        Args:
            processed_data_path: Path to processed data JSON file
        """
        # Load the processed data
        with open(processed_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to documents
        documents = self._convert_to_documents(data)
        
        # If vector store doesn't exist, create it
        if self.vectorstore is None:
            self.create_from_processed_data(processed_data_path)
            return
        
        # Add documents to existing vector store
        self.vectorstore.add_documents(documents)
        
        # Persist to disk
        self.vectorstore.persist()
        print(f"Added {len(documents)} documents to vector store")
    
    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            k: Number of relevant chunks to retrieve
            
        Returns:
            List of relevant document chunks
        """
        if self.vectorstore is None:
            print("Vector store not initialized. Please add documents first.")
            return []
        
        # Retrieve documents
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs
    
    def retrieve_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """Retrieve relevant chunks with similarity scores.
        
        Args:
            query: User query
            k: Number of relevant chunks to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        if self.vectorstore is None:
            print("Vector store not initialized. Please add documents first.")
            return []
        
        # Retrieve documents with scores
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        return docs_and_scores
    
    def delete_collection(self):
        """Delete the entire vector store collection."""
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            self.vectorstore = None
            print("Vector store collection deleted")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection.
        
        Returns:
            Dictionary of collection statistics
        """
        if self.vectorstore is None:
            return {"status": "not_initialized"}
        
        try:
            count = self.vectorstore._collection.count()
            return {
                "status": "active",
                "document_count": count,
                "embedding_model": self.embedding_model_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Example usage
if __name__ == "__main__":
    vector_manager = VectorDatabaseManager()
    
    # Create from processed data
    vector_manager.create_from_processed_data("processed_data/processed_documents.json")
    
    # Test retrieval
    docs = vector_manager.retrieve_relevant_chunks("what is the account type of little champs account is it saving or current ?")
    for doc in docs:
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)
    
    # Get collection stats
    stats = vector_manager.get_collection_stats()
    print(f"Collection stats: {stats}")