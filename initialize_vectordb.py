
import os
from data_processor import DataProcessor
from vector_db_manager import VectorDatabaseManager

def initialize_database():
    # Paths
    data_dir = "data"
    processed_dir = "processed_data"
    vector_dir = "vectorstore"
    
    # Create directories if they don't exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(vector_dir, exist_ok=True)
    
    # Initialize components
    data_processor = DataProcessor(output_dir=processed_dir)
    vector_manager = VectorDatabaseManager(persist_directory=vector_dir)
    
    # Process all files in data directory
    print(f"Processing files from {data_dir}...")
    all_documents = data_processor.process_directory(data_dir)
    
    # Save processed documents
    processed_path = os.path.join(processed_dir, "initial_processed_documents.json")
    data_processor.save_processed_data(all_documents, processed_path)
    print(f"Saved {len(all_documents)} processed document chunks to {processed_path}")
    
    # Create vector database
    print("Creating vector database...")
    vector_manager.create_from_processed_data(processed_path)
    
    # Get stats
    stats = vector_manager.get_collection_stats()
    print("\nVector Database Creation Complete!")
    print(f"Status: {stats.get('status', 'unknown')}")
    print(f"Document count: {stats.get('document_count', 0)}")
    print(f"Embedding model: {stats.get('embedding_model', 'unknown')}")

if __name__ == "__main__":
    initialize_database()