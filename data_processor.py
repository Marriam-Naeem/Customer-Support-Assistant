import os
import time
import re
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple

# Import necessary libraries for document processing
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DataProcessor:
    """Enhanced data processor for the Bank Customer Service chatbot."""
    
    def __init__(self, 
                 output_dir: str = "processed_data",
                 chunk_size: int = 300,
                 chunk_overlap: int = 50):
        """Initialize the data processor.
        
        Args:
            output_dir: Directory to store processed data
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """Apply text cleaning and anonymization.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and anonymized text
        """
        # 1. Anonymize email addresses
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '<EMAIL>', text)

        # 2. Anonymize phone numbers (various formats)
        text = re.sub(r'(\+\d{1,3}[\s\-]?)?\(?\d{3,5}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}', '<PHONE>', text)

        # 3. Anonymize IBAN and account numbers
        text = re.sub(r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b', '<IBAN>', text)  # IBAN
        text = re.sub(r'\b\d{12,20}\b', '<ACCOUNT_NO>', text)  # Numeric account numbers

        # 4. Anonymize currency values and amounts
        text = re.sub(r'\b(PKR|Rs\.?)\s?\d{1,3}(,\d{3})*(\.\d+)?\b', '<AMOUNT>', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(USD|EUR|GBP)\s?\d{1,3}(,\d{3})*(\.\d+)?\b', '<AMOUNT>', text, flags=re.IGNORECASE)

        # 5. Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '<URL>', text)

        # 6. Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single file based on its type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of document chunks with metadata
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        
        try:
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif file_ext == '.csv':
                loader = CSVLoader(file_path)
                docs = loader.load()
            elif file_ext == '.txt':
                loader = TextLoader(file_path)
                docs = loader.load()
            elif file_ext == '.json':
                # Handle JSON files - could be FAQ format or other
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                docs = []
                # Check if it's our FAQ format
                if isinstance(data, dict) and "categories" in data:
                    for category in data["categories"]:
                        category_name = category["category"]
                        for qa in category["questions"]:
                            content = f"Category: {category_name}\nQ: {qa['question']}\nA: {qa['answer']}"
                            docs.append({
                                "page_content": content,
                                "metadata": {"source": filename, "category": category_name}
                            })
                else:
                    # Generic JSON handling
                    content = json.dumps(data, indent=2)
                    docs = [{"page_content": content, "metadata": {"source": filename}}]
            else:
                # For unknown file types, try using a generic loader
                loader = UnstructuredFileLoader(file_path)
                docs = loader.load()
                
            # Clean and split the documents
            cleaned_docs = []
            for doc in docs:
                if isinstance(doc, dict):
                    content = doc["page_content"]
                    metadata = doc["metadata"]
                else:
                    content = doc.page_content
                    metadata = doc.metadata
                    
                cleaned_content = self.clean_text(content)
                splits = self.text_splitter.split_text(cleaned_content)
                
                for i, split in enumerate(splits):
                    split_metadata = metadata.copy()
                    split_metadata["chunk_id"] = i
                    cleaned_docs.append({
                        "content": split,
                        "metadata": split_metadata
                    })
                    
            return cleaned_docs
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Process all files in a directory.
        
        Args:
            directory_path: Path to directory containing files
            
        Returns:
            List of all document chunks
        """
        all_docs = []
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                # Skip hidden files
                if file.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, file)
                processed_docs = self.process_file(file_path)
                all_docs.extend(processed_docs)
                
        return all_docs
    
    def save_processed_data(self, documents: List[Dict[str, Any]], filename: str = "processed_documents.json"):
        """Save processed documents to file.
        
        Args:
            documents: List of processed documents
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(documents)} processed documents to {output_path}")
    
    def add_new_document(self, file_path: str, output_filename: str = "updated_documents.json"):
        """Process a new document and add it to existing processed data.
        
        Args:
            file_path: Path to new document
            output_filename: Name of updated output file
        """
        # Load existing documents if available
        existing_docs = []
        existing_file = os.path.join(self.output_dir, "processed_documents.json")
        
        if os.path.exists(existing_file):
            with open(existing_file, 'r', encoding='utf-8') as f:
                existing_docs = json.load(f)
        
        # Process the new document
        new_docs = self.process_file(file_path)
        
        # Combine and save
        updated_docs = existing_docs + new_docs
        self.save_processed_data(updated_docs, output_filename)
        
        return len(new_docs)

# Example usage
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Process all files in a directory
    docs = processor.process_directory("data")
    processor.save_processed_data(docs)
    
    # Add a new document
    num_added = processor.add_new_document("data/new_document.pdf")
    print(f"Added {num_added} chunks from new document")