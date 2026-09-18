# Autonomous arXiv Paper Digest & QA Agent

An autonomous, state-driven multi-agent system built with **LangGraph**, **Google Gemini**, and **ChromaDB** that ingests academic papers from arXiv, generates structured executive briefings, and provides hallucination-free, grounded Q&A through local vector retrieval.

## 🏗️ Architecture & State Graph

The agent uses **LangGraph** to manage explicit state transitions between pipeline nodes.

## 🚀 Setup & Run Instructions

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure `.env` with your Google AI Studio API key
4. Run: `python agent.py`
