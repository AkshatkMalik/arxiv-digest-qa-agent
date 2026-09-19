Markdown
> 🎥 **Watch the 4-Minute Demo Walkthrough:** [Click here to view the Loom video demo](https://www.loom.com/share/b8441f01458a46a3966c016e106e478f)
# 🤖 Autonomous arXiv Paper Digest & QA Agent (LangGraph Edition)

An intelligent, stateful research assistant built with LangGraph, Google Gemini Flash, ChromaDB, and PyMuPDF. This agent autonomously fetches academic papers from arXiv (via specific ID or natural-language topic search), parses and chunks PDFs, synthesizes rigorous Executive Briefings, and powers a grounded, citation-backed Interactive Q&A loop.

---

## 🏗️ Architecture & State Graph

The agent is engineered as an explicit, stateful graph using LangGraph rather than a single linear script or monolithic prompt chain. 

### State Graph Flow & Execution Nodes
1. **Query Understanding Node:** Parses user intent, distinguishing between a direct arXiv ID lookup vs. a natural-language topic search.
2. **arXiv Retrieval Node:** Queries the official arXiv Atom API with built-in exponential backoff retries.
3. **Selection / Ranking Node:** Ranks and selects the target paper, checking local storage cache before triggering downloads.
4. **Fetch & Parse Node:** Extracts clean, structured text using PyMuPDF.
5. **Chunk & Embed Node:** Applies semantic text chunking and embeds the vectors into a local ChromaDB instance.
6. **Summarize Node:** Generates a structured Markdown Executive Briefing via Google Gemini.
7. **Interactive RAG QA Loop:** Enters a persistent terminal loop to handle grounded follow-up questions with source citations and strict anti-hallucination guardrails.

### State Schema Shape (`AgentState`)
The pipeline passes a strongly typed dictionary (`TypedDict`) across all nodes:
* `query` (str): Raw user input (arXiv ID or research topic).
* `intent` (str): Classified intent (`id_lookup` or `topic_search`).
* `papers` (List[Dict]): Candidate paper metadata retrieved from arXiv.
* `selected_paper` (Dict): Final chosen paper metadata & local PDF path.
* `raw_text` (str): Full extracted text from the PDF via PyMuPDF.
* `chunks` (List[str]): Semantically segmented text chunks.
* `vector_collection` (Optional[Any]): Active ChromaDB collection reference for RAG QA.
* `briefing` (str): Final generated Markdown Executive Briefing.
* `error` (Optional[str]): Error message string for graceful fallback handling.

---

## 🛠️ Tech Stack & Dependencies

* **Orchestration:** LangGraph
* **LLM & Summarization:** Google GenAI SDK (`google-genai`) powered by Gemini
* **Vector Database:** ChromaDB (fully local embedded vector store)
* **PDF Parsing:** PyMuPDF (`fitz`)
* **arXiv Client:** Official `arxiv` Python package (Atom feed API)
* **Environment Management:** `python-dotenv`



## 🚀 Setup & Run Instructions

1. Clone the Repository & Navigate to Directory

git clone https://github.com/AkshatkMalik/arxiv-digest-qa-agent.git
cd arxiv-digest-qa-agent



## 💡 Example Run & Output

### Input A: Specific Paper ID Lookup
* **Input:** `1706.03762` *(Attention Is All You Need)*
* **Briefing Output Schema:**
  * **Title & Authors:** Attention Is All You Need (Vaswani et al.)
  * **arXiv ID & Link:** `1706.03762` — [View on arXiv](https://arxiv.org/abs/1706.03762)
  * **Publish Date:** June 12, 2017
  * **1-Paragraph Summary:** Proposes the Transformer architecture, replacing traditional recurrence and convolutions entirely with attention mechanisms, setting a new standard for machine translation efficiency and quality.
  * **Problem Statement:** Recurrent models struggle with sequential computation constraints, preventing parallelization across training examples over long contexts.
  * **Method / Approach:** 
    * Multi-head self-attention mechanisms replacing recurrence.
    * Positional encodings to retain token order awareness.
    * Encoder-decoder stack architecture.
  * **Key Results / Claims:** Achieved state-of-the-art BLEU scores on WMT 2014 English-to-German and English-to-French translation tasks with significantly reduced training time.
  * **Explicit Limitations:** Computational complexity per layer scales quadratically with sequence length, making ultra-long context processing expensive without sparse attention approximations.
  * **Suggested Follow-up Questions:** *"How does multi-head attention compare to single-head attention?"*, *"What is the purpose of scaling dot-product attention by the square root of $d_k$?"*

### Interactive Q&A Sample Exchanges
* **User Question 1:** *"Why is the scaling factor divided by the square root of $d_k$?"*
  * **Agent Answer:** For large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients. Dividing by $\sqrt{d_k}$ prevents this gradient vanishing effect. *(Source Citation: Chunk [14])*
* **User Question 2:** *"What optimizer and learning rate schedule did they use for the base model?"*
  * **Agent Answer:** They used the Adam optimizer with $\beta_1 = 0.9$, $\beta_2 = 0.98$, and $\epsilon = 10^{-9}$, along with a custom warmup learning rate schedule described in Section 5.3. *(Source Citation: Chunk [22])*



## 🛠️ Design Decisions, Tradeoffs & Known Limitations

### 1. LangGraph vs. Linear Chains
* **Decision:** Built as an explicit state machine using LangGraph.
* **Why:** Linear chains fail unpredictably during production errors. LangGraph provides state control, making debugging transparent and enabling graceful error containment if a node fails.
* **Tradeoff:** Slightly more verbose code boilerplate, but offers vastly superior debugging and pipeline reliability.

### 2. Local-First Storage (ChromaDB & PyMuPDF)
* **Decision:** Uses PyMuPDF for parsing and ChromaDB for local embedded vector storage.
* **Why:** Avoids external cloud database hosting costs and API overhead, allowing the entire pipeline to run locally out-of-the-box.
* **Tradeoff:** Ideal for single-user execution and local testing; scaling to multi-tenant cloud usage would eventually require a distributed vector database like Pinecone or Milvus.

### 3. Error Resilience & Anti-Hallucination Guardrails
* **Decision:** Integrated automatic exponential backoff retries and strict system-prompt grounding constraints.
* **Why:** Handles temporary arXiv API outages (`503 Service Unavailable` spikes) smoothly and forces the agent to refuse out-of-scope queries rather than hallucinating facts.
* **Tradeoff:** Strict refusal rules mean the agent will decline to speculate on answers not present in the text, prioritizing factual integrity over guessing.

### 4. Known Limitations & Future Roadmap
* **Retrieval Scope (Top-$k$ Chunks):** The vector search retrieves the top $n=3$ chunks for speed and token economy. Complex questions spanning distant, disconnected pages of a long paper can occasionally miss context.
