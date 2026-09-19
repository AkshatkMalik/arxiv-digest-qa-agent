# 🤖 Autonomous arXiv Paper Digest & QA Agent (LangGraph Edition)

An intelligent, stateful research assistant built with **LangGraph**, **Google Gemini 3.6 Flash**, **ChromaDB**, and **PyMuPDF**. This agent autonomously fetches academic papers from arXiv (via specific ID or natural-language topic search), parses and chunks PDFs, synthesizes rigorous Executive Briefings, and powers a grounded, citation-backed Interactive Q&A loop.

---

## 🏗️ Architecture & State Graph

The agent is engineered as an explicit, stateful graph using **LangGraph** rather than a single linear script or monolithic prompt chain. State persists cleanly across all execution nodes.

### State Graph Flow
```text
[User Input] 
     ↓
(1) Query Understanding Node  ---> (Parses Intent: ID Lookup vs. Topic Search)
     ↓
(2) arXiv Retrieval Node      ---> (Queries official arXiv Atom API)
     ↓
(3) Selection / Ranking Node  ---> (Ranks & selects top paper, downloads PDF)
     ↓
(4) Fetch & Parse Node        ---> (Extracts clean text using PyMuPDF)
     ↓
(5) Chunk & Embed Node        ---> (Semantic chunking & local ChromaDB indexing)
     ↓
(6) Summarize Node            ---> (Generates structured Executive Briefing via Gemini 3.6 Flash)
     ↓
[Interactive RAG QA Loop]     ---> (Grounded follow-up Q&A with source citations & refusals)

from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """Defines the state schema passed across all nodes in the LangGraph pipeline."""
    query: str                       # Raw user input (arXiv ID or research topic)
    intent: str                      # Classified intent: "id_lookup" or "topic_search"
    papers: List[Dict[str, Any]]     # Candidate papers metadata retrieved from arXiv API
    selected_paper: Dict[str, Any]   # Final chosen paper metadata & local PDF file path
    raw_text: str                    # Full extracted text from the PDF via PyMuPDF
    chunks: List[str]                # Semantically segmented text chunks
    vector_collection: Optional[Any] # Active ChromaDB collection reference for RAG QA
    briefing: str                    # Final generated Markdown Executive Briefing
    error: Optional[str]             # Error message string for graceful fallback handling
    execution_metadata: Dict[str, Any] # Optional: tracking runtime metrics, tokens, or timestamps

🛠️ Tech Stack & Dependencies

Orchestration: LangGraph

LLM & Summarization: Google GenAI SDK (google-genai) powered by gemini-3.6-flash

Vector Database: ChromaDB (fully local embedded vector store)

PDF Parsing: PyMuPDF (fitz)

arXiv Client: Official arxiv Python package (Atom feed API)

Environment Management: python-dotenv

🚀 Setup & Run Instructions
1. Clone the Repository & Navigate to DirectoryDOSgit clone [https://github.com/YOUR_USERNAME/arxiv-digest-qa-agent.git](https://github.com/YOUR_USERNAME/arxiv-digest-qa-agent.git)
cd arxiv-digest-qa-agent
2. Create and Activate a Virtual EnvironmentDOSpython -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
3. Install DependenciesDOSpip install -r requirements.txt
4. Configure Environment SecretsCreate a .env file in the root directory and add your free Google GenAI API key:Code snippetGEMINI_API_KEY=your_actual_api_key_here
5. Run the AgentDOSpython agent.py

💡 Example 
RunsExample A: Specific Paper ID LookupInput: 1706.03762 (Attention Is All You Need)
Output Summary: Generates a complete Markdown Executive Briefing covering Metadata, Plain-English Summary, Problem Statement, Method/Approach, Key Results, and Explicit Limitations.Interactive 
Q&A Session:User: "Why is the scaling factor divided by the square root of $d_k$?"
Agent: Provides grounded explanation referencing Chunk [14] with source citations and token diagnostics.

Example B: Natural Language Topic 
SearchInput: "recent work on KV-cache compression for LLMs"Output 
Summary: Automatically queries arXiv, ranks candidates, selects the top paper, extracts 24k+ characters, indexes 31 chunks, and synthesizes a comprehensive briefing.

🛠️ Design Decisions & Tradeoffs
1. LangGraph vs. Linear Chains:
   Decision: Built as a state machine with LangGraph.
   Why: Provides explicit state control and graceful error containment if a node fails.
   Tradeoff: Slightly more verbose code, but offers vastly superior debugging and reliability.
2. Local-First Storage (ChromaDB & PyMuPDF)
   Decision: Uses PyMuPDF for parsing and ChromaDB for local vector embeddings.
   Why: Avoids external cloud DB costs and lets the agent run instantly out-of-the-box.
   Tradeoff: Excellent for local execution, though scaling to multi-user cloud production would require a distributed database.
3. Error Resilience & Guardrails
   Decision: Added exponential backoff retries and strict anti-hallucination prompts.
   Why: Handles temporary arXiv API outages (503 errors) smoothly and forces the agent to say "I don't know" rather than guessing.
   Tradeoff: Strict refusal rules mean it won't speculate on out-of-scope queries, ensuring total factual accuracy.
4. Known Limitations
   Top-$k$ Scope: Retrieves the top $3$ chunks for speed and token efficiency.
   Complex questions spanning distant pages can occasionally miss context, which could be solved with hybrid search in future versions.
