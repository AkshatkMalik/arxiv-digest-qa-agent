"""
utils.py - Utility Layer for arXiv Agent
----------------------------------------
Handles safe PDF parsing via PyMuPDF, semantic text chunking, 
and local ChromaDB vector database persistence.
"""

import os
import fitz  # PyMuPDF
import chromadb


def parse_pdf_text(pdf_path: str) -> str:
    """
    Safely opens a local PDF file and extracts all text content page by page.
    Uses a context manager to prevent 'document closed' errors.
    """
    print(f"[INFO] Extracting text from PDF: {pdf_path}")
    extracted_text = ""
    
    try:
        # Using a context manager ensures the file handle stays active during iteration
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text:
                    extracted_text += f"\n--- Page {page_num + 1} ---\n" + text
                    
        if not extracted_text.strip():
            raise ValueError("The extracted text from the PDF is empty.")
            
        return extracted_text
    except Exception as e:
        raise RuntimeError(f"PDF Parsing Failed: {e}")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Splits large raw text into overlapping semantic chunks for vector indexing.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # Move forward with overlap window
        
    return chunks


def init_vector_db(chunks: list[str], collection_name: str = "arxiv_agent_store"):
    """
    Initializes a persistent local ChromaDB client, clears old collections if any,
    and indexes the new paper chunks for similarity search.
    """
    # Initialize local persistent client stored in './chroma_db' folder
    db_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Re-create collection to ensure clean state per paper run
    try:
        db_client.delete_collection(name=collection_name)
    except Exception:
        pass
        
    collection = db_client.get_or_create_collection(name=collection_name)
    
    # Add chunks into ChromaDB with unique IDs
    ids = [str(i) for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        ids=ids
    )
    
    print(f"[+] Successfully indexed {len(chunks)} chunks into local ChromaDB vector store.")
    return collection