# RAG From Scratch

A progressive implementation of Retrieval-Augmented Generation (RAG) pipelines built with LangChain and Claude. Each file introduces a more sophisticated retrieval strategy, organized from foundational to advanced.

## What is RAG?

Large language models have a knowledge cutoff and no access to your private data. RAG solves this by retrieving relevant context from a document store at query time and injecting it into the prompt, grounding the model's response in real, up-to-date sources rather than relying on training data alone.

---

## Techniques

### Foundation

**`basic_rag.py`** — The core pipeline. Load any webpage or PDF, chunk it, embed it with a local HuggingFace model, and query it interactively with Claude. Supports multi-turn Q&A without reloading the document.

---

### Query Translation
*Techniques that rewrite or transform the user's question before retrieval to improve the quality of retrieved documents.*

**`multi_query_rag.py`** — Rewrites the question into 5 different perspectives and retrieves documents for each, then takes the unique union. Solves the problem of a single poorly-worded query missing relevant chunks.

**`rag_fusion.py`** — Extends Multi-Query by adding Reciprocal Rank Fusion (RRF) scoring. Documents that appear consistently near the top across multiple query versions are ranked highest, improving retrieval precision over simple deduplication.

**`decomposition_rag.py`** — Breaks a complex question into 3 isolated sub-questions, answers each independently with its own RAG chain, then synthesizes a final answer. Effective for multi-part questions that span multiple topics.

**`step_back_rag.py`** — Rewrites the user's specific question into a broader, more general version before retrieval using few-shot prompting. Captures higher-level context that a narrow query might miss.

**`hyde_rag.py`** — Hypothetical Document Embeddings. Generates a hypothetical answer to the question first, then uses that as the search query. A question embedding looks very different from a document embedding. A fake answer looks much more like a real document, improving retrieval accuracy.

---

### Routing
*Techniques that direct questions to the right handler before retrieval.*

**`logical_routing.py`** — Uses Claude's structured output to classify a question and route it to the most relevant of two loaded sources. The router prompt dynamically includes the actual URLs so Claude knows what each source contains.

**`semantic_routing.py`** — Embeds the user's question and compares it against pre-written prompt templates using cosine similarity. Routes to a beginner-friendly or advanced explanation style based on how the question is phrased, then retrieves from the loaded document accordingly.

---

### Indexing
*Techniques that change how documents are stored and indexed to improve retrieval quality.*

**`multi_representation_indexing_rag.py`** — Separates what is searched from what is returned. Indexes short Claude-generated summaries in the vectorstore for precise retrieval, but returns the full original chunks to Claude for generation. Best of both worlds. Accurate search with complete context.

---

## Stack

| Component | Tool |
|---|---|
| LLM | Claude (claude-sonnet-4-6) via Anthropic API |
| Orchestration | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, no API key needed) |
| Vector Store | ChromaDB |
| Document Loading | LangChain WebBaseLoader + PyPDFLoader |

---

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

### 4. Run any technique
```bash
python basic_rag.py
python multi_query_rag.py
python rag_fusion.py
python decomposition_rag.py
python step_back_rag.py
python hyde_rag.py
python logical_routing.py
python semantic_routing.py
python multi_representation_indexing_rag.py
```
Each script prompts you for a URL or PDF path, then opens an interactive Q&A loop.

---

## Acknowledgements
Built by following the [RAG From Scratch](https://github.com/langchain-ai/rag-from-scratch) series by LangChain, adapted to use Anthropic's Claude instead of OpenAI and extended with URL/PDF input, interactive Q&A, and additional retrieval features.
