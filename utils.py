"""
utils.py - Core Utilities for arXiv Agent
-----------------------------------------
Handles paper fetching, local PDF caching, PyMuPDF text parsing,
semantic text chunking, and ChromaDB vector persistence.
"""

import os
import logging
import fitz  # PyMuPDF
import arxiv
import chromadb
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Set up professional logging (Replaces messy print statements)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Initialize Google GenAI client securely
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def fetch_arxiv_paper(query: str, max_results: int = 1) -> dict:
    """
    Searches arXiv by ID (e.g., '1706.03762') or topic string, 
    caches the PDF locally, and returns paper metadata.
    """
    logger.info(f"Searching arXiv for query: '{query}'")
    
    try:
        # Determine if query is an arXiv ID or a general topic search
        if "." in query and query.replace(".", "").isdigit():
            search = arxiv.Search(id_list=[query])
        else:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

        results = list(arxiv.Client().results(search))
        if not results:
            raise ValueError(f"No papers found matching query: '{query}'")

        paper = results[0]
        safe_id = paper.get_short_id().replace("/", "_")
        pdf_filename = f"temp_{safe_id}.pdf"
        
        # Smart Caching: Skip download if the PDF already exists locally
        if os.path.exists(pdf_filename):
            logger.info(f"Found cached PDF locally: {pdf_filename}. Skipping download.")
        else:
            logger.info(f"Downloading PDF: '{paper.title}'...")
            paper.download_pdf(filename=pdf_filename)

        metadata = {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "published": str(paper.published.date()),
            "summary": paper.summary,
            "pdf_url": paper.pdf_url,
            "pdf_path": pdf_filename,
            "entry_id": paper.entry_id
        }
        return metadata

    except Exception as e:
        logger.error(f"Failed to fetch paper from arXiv: {e}")
        raise


def parse_pdf_text(pdf_path: str) -> str:
    """Extracts clean plain text from a PDF file using PyMuPDF (fitz)."""
    logger.info(f"Extracting text from PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
        
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        full_text += page.get_text()
    doc.close()
    
    logger.info(f"Successfully extracted {len(full_text)} characters from {len(doc)} pages.")
    return full_text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Splits long text into smart, overlapping chunks. 
    Attempts to break at sentence or paragraph boundaries when possible.
    """
    logger.info("Chunking text for vector embeddings...")
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # If we're not at the end of the text, try to find a natural boundary (period or newline)
        if end < text_length:
            next_boundary = max(text.rfind('. ', start, end), text.rfind('\n', start, end))
            if next_boundary > start + (chunk_size // 2):
                end = next_boundary + 1

        chunks.append(text[start:end].strip())
        start += (chunk_size - overlap)

    logger.info(f"Generated {len(chunks)} text chunks.")
    return chunks


def init_vector_db(chunks: list[str], collection_name: str = "arxiv_papers"):
    """Initializes a local persistent ChromaDB instance and indexes paper chunks."""
    logger.info("Initializing local ChromaDB vector store...")
    
    # Persistent client saves database locally in './chroma_db'
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Recreate collection to ensure clean state per run
    try:
        chroma_client.delete_collection(name=collection_name)
        logger.info(f"Cleared existing collection: {collection_name}")
    except Exception:
        pass

    collection = chroma_client.create_collection(name=collection_name)

    # Assign sequential string IDs for each text chunk
    ids = [str(i) for i in range(len(chunks))]
    
    # Add documents to ChromaDB
    collection.add(
        documents=chunks,
        ids=ids
    )
    
    logger.info(f"Successfully indexed {len(chunks)} chunks into ChromaDB collection '{collection_name}'.")
    return collection