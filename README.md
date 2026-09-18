# 🤖 Autonomous arXiv Paper Digest & QA Agent (LangGraph Edition)

An intelligent, stateful research assistant built with **LangGraph**, **Google Gemini 3.6 Flash**, **ChromaDB**, and **PyMuPDF**. This agent autonomously fetches academic papers from arXiv (via specific ID or natural-language topic search), parses and chunks PDFs, synthesizes rigorous Executive Briefings, and powers a grounded, citation-backed Interactive Q&A loop.

---

## 🏗️ Architecture & State Graph

The agent is engineered as an explicit, stateful graph using **LangGraph** rather than a single linear script or monolithic prompt chain. State persists cleanly across all execution nodes.

### State Graph Flow

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


### Shared State Definition (`AgentState`)
```python
class AgentState(TypedDict):
    query: str                       # Raw user input (ID or topic)
    intent: str                      # "id_lookup" or "topic_search"
    papers: List[Dict[str, Any]]     # Candidate papers retrieved from arXiv
    selected_paper: Dict[str, Any]   # Chosen paper metadata & local PDF path
    raw_text: str                    # Full extracted text from PDF
    chunks: List[str]                # Semantically chunked text segments
    vector_collection: Any           # ChromaDB collection reference
    briefing: str                    # Generated Markdown Executive Briefing
    error: str                       # Error message for graceful failure handling


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

📝 Design Decisions & Tradeoffs
1. Why LangGraph over Linear Chains?Decision: We selected LangGraph to enforce explicit control flow and modular separation of concerns.Tradeoff: While slightly more verbose than standard chaining libraries, it provides robust state persistence, transparent debugging at every pipeline node, and graceful error containment if a single step fails.
2. Local-First Storage (ChromaDB & PyMuPDF)Decision: Used ChromaDB for embedded vector storage and PyMuPDF for text extraction.
Tradeoff: Avoids external cloud DB dependencies and hosting costs. Everything runs entirely offline on local hardware, making setup frictionless for evaluation.
3. Error Resilience & Anti-Hallucination GuardsDecision: Implemented automatic exponential backoff retries (handling temporary 503 Service Unavailable cloud spikes) and strict system-prompt constraints during the QA loop.
Tradeoff: If a query falls outside the vector store context, the agent explicitly refuses to answer rather than hallucinating facts.
4. Known LimitationsTop-$k$ Retrieval Scope: The vector search retrieves the top $n=3$ chunks. While optimized for speed and token economy, complex multi-hop queries requiring synthesis across distant pages can occasionally miss context. Future iterations could integrate parent-document retrieval or hybrid search.