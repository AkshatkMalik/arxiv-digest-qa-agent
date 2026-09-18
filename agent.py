"""
agent.py - Master Orchestrator for Autonomous arXiv Paper Digest & QA Agent
-------------------------------------------------------------------------
Description: 
    This script coordinates the end-to-end RAG (Retrieval-Augmented Generation) pipeline:
    1. Validates and sanitizes user input.
    2. Fetches paper metadata and caches PDFs locally.
    3. Parses text with PyMuPDF and chunks it semantically.
    4. Indexes chunks into local ChromaDB for vector similarity search.
    5. Generates a structured Executive Briefing using Google Gemini with token diagnostics.
    6. Runs an interactive Q&A loop with strict anti-hallucination grounding and source citations.
"""

import os
from dotenv import load_dotenv
from google import genai
from utils import fetch_arxiv_paper, parse_pdf_text, chunk_text, init_vector_db

# Load secret environment variables (API keys) securely from .env
load_dotenv()

# Initialize the official Google GenAI client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("[-] Error: GEMINI_API_KEY is missing from your .env file!")

client = genai.Client(api_key=GEMINI_API_KEY)


def run_agent():
    print("=" * 65)
    print(" 🤖 Autonomous arXiv Paper Digest & QA Agent (Production-Grade)")
    print("=" * 65)
    
    # ==========================================
    # STEP 1: Robust Input Filtering & Validation
    # ==========================================
    query = input("\nEnter an arXiv Paper ID (e.g., 1706.03762) or research topic: ").strip()
    
    # Defensive check: Prevent empty inputs, malicious strings, or meaningless symbols
    if not query or len(query) < 2 or not any(c.isalnum() for c in query):
        print("[-] Error: Invalid query format. Please restart and provide a valid arXiv ID or topic.")
        return

    try:
        # ==========================================
        # STEP 2: Paper Acquisition & Local Caching
        # ==========================================
        metadata = fetch_arxiv_paper(query)
        print(f"\n[+] Success! Paper Acquired: {metadata['title']}")
        print(f"[+] Authors: {', '.join(metadata['authors'])}")
        print(f"[+] Published: {metadata['published']}")

        # ==========================================
        # STEP 3: Text Extraction & Vector Indexing
        # ==========================================
        # Extract plain text cleanly using PyMuPDF
        raw_text = parse_pdf_text(metadata["pdf_path"])

        # Split text into manageable chunks and store locally in ChromaDB
        chunks = chunk_text(raw_text)
        vector_collection = init_vector_db(chunks)

        # ==========================================
        # STEP 4: Executive Briefing Generation
        # ==========================================
        print("\n[*] Synthesizing Executive Briefing using Google Gemini...")
        briefing_prompt = f"""
        You are an expert AI research assistant. Analyze the academic paper text below and generate 
        a clean, structured Executive Briefing in Markdown format containing:
        1. Title & Metadata
        2. Plain-English Summary
        3. Core Problem Addressed
        4. Methodology
        5. Key Results
        6. Explicit Limitations
        7. 3 Suggested Follow-Up Questions for Deep-Dive Q&A.

        Paper Title: {metadata['title']}
        Abstract: {metadata['summary']}
        
        Full Text Excerpt:
        {raw_text[:15000]}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=briefing_prompt
        )
        
        print("\n" + "=" * 65)
        print("📋 EXECUTIVE BRIEFING")
        print("=" * 65)
        print(response.text)
        print("=" * 65)

        # Cost & performance awareness: Display token diagnostics
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            print(f"\n📊 [Token Diagnostics - Briefing]")
            print(f"   • Prompt Tokens: {response.usage_metadata.prompt_token_count}")
            print(f"   • Response Tokens: {response.usage_metadata.candidates_token_count}")
            print(f"   • Total Tokens Used: {response.usage_metadata.total_token_count}")

        # ==========================================
        # STEP 5: Interactive Q&A Loop (RAG)
        # ==========================================
        print("\n[*] Entering Interactive Q&A Mode. Type 'exit' or 'quit' to close.")
        
        while True:
            user_question = input("\n💬 Ask a question about the paper: ").strip()
            
            if user_question.lower() in ["exit", "quit"]:
                print("[*] Exiting agent. Have a wonderful day!")
                break
            
            # Input sanitization for chat questions
            if not user_question or len(user_question) < 3 or not any(c.isalnum() for c in user_question):
                print("[-] Warning: Invalid or meaningless input detected. Please ask a valid question.")
                continue

            # Retrieve top 3 relevant chunks from ChromaDB based on semantic similarity
            search_results = vector_collection.query(
                query_texts=[user_question],
                n_results=3
            )
            
            # Safety check if vector store returns empty results
            if not search_results or 'documents' not in search_results or not search_results['documents'][0]:
                print("\n🤖 Answer: I could not find any relevant sections in the paper to answer your question.")
                continue

            retrieved_chunks = search_results['documents'][0]
            chunk_ids = search_results['ids'][0] if 'ids' in search_results else [str(i) for i in range(len(retrieved_chunks))]
            combined_context = "\n\n".join(retrieved_chunks)

            # Grounded prompting to completely eliminate hallucinations
            qa_prompt = f"""
            You are a grounded QA assistant. Answer the user's question accurately using ONLY 
            the provided context from the research paper. If the answer cannot be found in the 
            context, state clearly: "I cannot find the answer in the provided paper text."

            Context from Paper:
            {combined_context}

            User Question: {user_question}
            Answer:
            """

            qa_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=qa_prompt
            )

            # Output response
            print(f"\n🤖 Answer:\n{qa_response.text}")

            # Explicit Source Citations (Traceability / Audit Trail)
            print(f"\n📌 [Source Citations / Audit Trail]")
            for idx, chunk_id in enumerate(chunk_ids):
                preview_snippet = retrieved_chunks[idx][:120].replace("\n", " ")
                print(f"   • Chunk ID [{chunk_id}]: \"{preview_snippet}...\"")

            # Token tracking for Q&A turn
            if hasattr(qa_response, 'usage_metadata') and qa_response.usage_metadata:
                print(f"📊 [Token Diagnostics - Q&A Turn] Total Tokens: {qa_response.usage_metadata.total_token_count}")

    except Exception as error:
        # Defensive error handling: Catches runtime exceptions gracefully without breaking the terminal
        print(f"\n[-] An error occurred during execution: {error}")
        print("[-] Please verify your internet connection, API key, or arXiv query format.")

if __name__ == "__main__":
    run_agent()