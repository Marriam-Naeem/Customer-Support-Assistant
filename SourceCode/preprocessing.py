import os
import time 
import re
import json
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document



os.getenv("GOOGLE_API_KEY")


"""
Approach 1: Use PDF reader and ignore the images using PDFLoader
Approach 2 : Use such loaders that offer OCR capabilities

"""

def process_pdfs_v1(pdf_docs):
    """Extracts text from PDF documents and writes to a text file with chunk of 300"""
    data = []
    
    
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text = "".join([page.extract_text() for page in pdf_reader.pages])

       
        file_name = os.path.basename(pdf)  


      
        chunk_size = 300
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            data.append({"file_name": file_name, "text": chunk})

    with open("Loaders_result/output_textonly.txt", "w", encoding='utf-8') as file:  # 'w' to overwrite the file or 'a' to append
       file.write(str(data))


def process_pdf_v2(pdf_docs):
    start_time = time.time()

    """
    Problem is that it is compuatationally expensive and slow
    for 3 slides:in 638.71 seconds.
    """
    with open("Loaders_result/output_pdfiumtry.txt", "a", encoding='utf-8') as file: 
        for pdf in pdf_docs:
            loader = PyPDFium2Loader(file_path=pdf, extract_images=True)
            docs = loader.load()
            file.write(str(docs)) 
            file.write("\n")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processed {pdf} in {elapsed_time:.2f} seconds.")
    return docs


import re

def clean_text(text):
    # 1. Anonymize email addresses
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '<EMAIL>', text)

    # 2. Anonymize phone numbers (11+ digits)
    text = re.sub(r'\b\d{11,}\b', '<PHONE>', text)

    # 3. Anonymize IBAN (common format)
    text = re.sub(r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b', '<IBAN>', text)

    # 4. Anonymize numeric account numbers
    text = re.sub(r'\b\d{12,20}\b', '<ACCOUNT_NO>', text)

    # 5. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '<URL>', text)

    # 6. Remove non-alphanumeric characters except spaces and few safe punctuations
    text = re.sub(r'[^\w\s.,:;!?]', '', text)

    # 7. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 8. Lowercase the entire text
    text = text.lower()

    return text


def process_pdf_v3(pdf_docs):
    start_time = time.time()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    all_chunks = []

    with open("Loaders_result/output_pymupdf_resursive_splitter_FINAL.json", "w", encoding='utf-8') as file:
        for pdf in pdf_docs:
            full_path = os.path.join(os.path.dirname(__file__), '..', pdf)
            full_path = os.path.abspath(full_path)
            print(f"Full path of {pdf}: {full_path}")

            loader = PyMuPDFLoader(file_path=full_path, extract_images=True)
            docs = loader.load_and_split(text_splitter=text_splitter)

            for doc in docs:
                cleaned = clean_text(doc.page_content)
                filename_only = os.path.splitext(os.path.basename(pdf))[0]  # Extract file name without .pdf
                chunk_data = {
                    "content": cleaned,
                    "metadata": {
                        "source": filename_only
                    }
                }
                all_chunks.append(chunk_data)

        json.dump(all_chunks, file, indent=4, ensure_ascii=False)

    end_time = time.time()
    print(f"Processed {len(pdf_docs)} files in {end_time - start_time:.2f} seconds.")
    return all_chunks






base_dir = "e:/Customer-Support-Assistant/data/pdfs"
pdf_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir) if file.endswith('.pdf')]
process_pdf_v3(pdf_files)

def clean_text(text):
    # 1. Anonymize emails
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '<EMAIL>', text)

    # 2. Anonymize phone numbers (11+ digits)
    text = re.sub(r'\b\d{11,}\b', '<PHONE>', text)

    # 3. Anonymize account numbers or IBANs (usually 16–24 alphanumeric characters)
    text = re.sub(r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b', '<IBAN>', text)  # IBAN
    text = re.sub(r'\b\d{12,20}\b', '<ACCOUNT_NO>', text)               # Numeric account numbers

    # # 4. Anonymize currency values (e.g., PKR 1,000,000 or Rs. 250000)
    # text = re.sub(r'\b(PKR|Rs\.?)\s?\d{1,3}(,\d{3})*(\.\d+)?\b', '<AMOUNT>', text, flags=re.IGNORECASE)
    # text = re.sub(r'\bUSD\s?\d{1,3}(,\d{3})*(\.\d+)?\b', '<AMOUNT>', text, flags=re.IGNORECASE)

    # # 5. Protect institution and financial terms
    # protected_entities = ['State Bank', 'Meezan Bank', 'Chief Financial Officer', 'Annual Report', 'Balance Sheet']
    # for phrase in protected_entities:
    #     text = text.replace(phrase, phrase.replace(" ", "_"))

    # # 6. Anonymize most names (two capitalized words, not part of protected terms)
    # text = re.sub(r'\b(?!State|Meezan|Chief|Annual|Balance)[A-Z][a-z]+\s(?!Bank|Officer|Report|Sheet)[A-Z][a-z]+\b', '<NAME>', text)

    # # 7. Restore protected entities
    # for phrase in protected_entities:
    #     text = text.replace(phrase.replace(" ", "_"), phrase)

    # 8. Normalize text
    text = re.sub(r'\s+', ' ', text).strip().lower()
    # 9. Remove excess whitespace
    text = re.sub(r'\s+', ' ', text)  

    return text


# # Load the txt file
# with open("Loaders_result/output_pymupdf_resursive_splitter_final.txt", "r", encoding="utf-8") as file:
#     raw_data = file.read()

# pattern = r"page_content='(.*?)'\)"
# matches = re.findall(pattern, raw_data, re.DOTALL)
# cleaned_chunks = [clean_text(chunk) for chunk in matches]


# # Optional: Save to JSONL for ingestion
# with open("Loaders_result/output_cleaned_final.json", "w", encoding="utf-8") as f:
#     json.dump(cleaned_chunks, f, indent=2, ensure_ascii=False)