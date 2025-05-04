import os
import time 
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


def process_pdf_v3(pdf_docs):
    start_time = time.time()
    """ Text spliter and Chunking overlap and size"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 300 ,chunk_overlap=50)
    """
    Problem is that it is computationally expensive and slow
    for 3 slides: in 564.89 seconds.
    for 3 slides with chunking: 528.98 seconds
    """
    with open("Loaders_result/output_pymupdf_resursive_splitter.txt", "w", encoding='utf-8') as file:  
        for pdf in pdf_docs:
            full_path = os.path.join(os.path.dirname(__file__), '..', pdf)
            full_path = os.path.abspath(full_path)
            print(f"Full path of {pdf}: {full_path}")

   

            loader = PyMuPDFLoader(file_path=full_path, extract_images=True)
            docs = loader.load_and_split(text_splitter=text_splitter)
            for doc in docs:
                file.write(str(docs))  
                file.write("\n")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processed {pdf} in {elapsed_time:.2f} seconds.")
    return docs




# pdf_folder = os.path.join("data", "pdfs")
# pdf_files = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.endswith('.pdf')]
# process_pdf_v3(pdf_files)
import os

base_dir = "e:/Customer-Support-Assistant/data/pdfs"
pdf_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir) if file.endswith('.pdf')]
process_pdf_v3(pdf_files)
