import json
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import os

# Load both JSON files
with open("Loaders_result\output_pymupdf_resursive_splitter_final.json", "r", encoding="utf-8") as f1:
    first_json = json.load(f1)

with open("data\\funds_transfer_app_features_faq.json", "r", encoding="utf-8") as f2:
    second_json = json.load(f2)

# Convert first JSON to Document format
first_docs = [
    Document(page_content=item["content"], metadata=item.get("metadata", {}))
    for item in first_json
]

# Convert second JSON to Document format
second_docs = []
for category_block in second_json["categories"]:
    category = category_block["category"]
    for q in category_block["questions"]:
        q_text = f"Q: {q['question']}\nA: {q['answer']}"
        second_docs.append(Document(page_content=q_text, metadata={"category": category}))

# Combine all documents
all_docs = first_docs + second_docs

# Define embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Directory to persist the database
persist_dir = "vectorstore"

# Create and persist Chroma vector DB
vectordb = Chroma.from_documents(documents=all_docs,
                                 embedding=embedding_model,
                                 persist_directory=persist_dir)

# Save vectorstore to disk
vectordb.persist()

print(f"Chroma DB created and persisted at {persist_dir}")
