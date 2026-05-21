# RAG From Scratch

A progressive implementation of Retrieval-Augmented Generation (RAG) pipelines, built with LangChain and Claude. Each module introduces a more sophisticated retrieval strategy, from basic semantic search to advanced query transformation techniques.

## What is RAG?

Large language models have a knowledge cutoff and no access to your private data. RAG solves this by retrieving relevant context from a document store at query time and injecting it into the prompt, grounding the model's response in real, up-to-date sources rather than relying on training data alone.

## Project Structure

```
RAG-From-Scratch/
├── basic_rag.py              # URL + PDF input, interactive Q&A loop
├── multi_query.py            # 5-perspective query expansion
├── rag_fusion.py             # Coming soon
├── decomposition.py          # Coming soon
├── step_back.py              # Coming soon
├── hyde.py                   # Coming soon
├── .env.example
└── requirements.txt
```

## Implementations

### 01 — Basic RAG `basic_rag.py`
The foundation. Load any webpage or PDF, chunk it, embed it with a local HuggingFace model, and query it interactively with Claude.

**Supports:**
- Any public URL
- Local PDF files
- Multi-turn Q&A session without reloading the document

### 02 — Multi-Query 'multi_query_rag.py'
Rewrites the user's question into 5 different perspectives and retrieves documents for each, then takes the unique union. Solves the problem of a single poorly-worded query missing relevant chunks.

### 03 — RAG Fusion *(coming soon)*
Extends Multi-Query by adding Reciprocal Rank Fusion (RRF) scoring documents that appear consistently across multiple query results are ranked higher, improving retrieval precision.

### 04 — Decomposition *(coming soon)*
Breaks complex multi-part questions into isolated sub-questions, answers each with its own RAG chain, then synthesizes a final answer. Effective for questions that span multiple topics.

### 05 — Step Back *(coming soon)*
Rewrites the user's specific question into a broader, more general version before retrieval — capturing higher-level context that a narrow query might miss.

### 06 — HyDE *(coming soon)*
Hypothetical Document Embeddings. Instead of searching with the question directly, generates a hypothetical answer first and uses that as the search query. Particularly effective for technical and domain-specific content.

## Stack

| Component | Tool |
|---|---|
| LLM | Claude (claude-sonnet-4-6) via Anthropic API |
| Orchestration | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, no API key needed) |
| Vector Store | ChromaDB |
| Document Loading | LangChain WebBaseLoader + PyPDFLoader |

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/crishikawa/RAG-From-Scratch.git
cd RAG-From-Scratch
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
```
Add your Anthropic API key to `.env`:
```
ANTHROPIC_API_KEY=your-key-here
```
Get a key at [console.anthropic.com](https://console.anthropic.com)

### 4. Run
```bash
python basic_rag.py
```
Then enter any URL or local PDF path when prompted:
```
Enter a URL or path to a PDF: https://en.wikipedia.org/wiki/Retrieval-augmented_generation
Ask a question (or 'quit' to exit): What are the main components of RAG?
```

## Requirements

```
langchain
langchain-anthropic
langchain-community
langchain-text-splitters
langchain-huggingface
chromadb
sentence-transformers
pypdf
python-dotenv
beautifulsoup4
```

## Acknowledgements
Built by following the [RAG From Scratch](https://github.com/langchain-ai/rag-from-scratch) series by LangChain, 
adapted to use Anthropic's Claude instead of OpenAI and extended with additional features.