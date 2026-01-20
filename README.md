# 🎯 Grounded RAG Assistant with Agents & Retrieval Debugger

A **production-grade Retrieval-Augmented Generation (RAG) system** that answers questions **strictly from internal documents**. When information isn't found in the knowledge base, the system **explicitly refuses to answer** rather than hallucinating responses.

Built for **correctness, traceability, and debuggability** — designed to mirror real-world enterprise and regulated-domain RAG implementations.

---

## 🌟 Why This Project Exists

Most RAG demos prioritize "always answering" even when source documents don't support the response. This project takes a principled approach:

✅ **Fully grounded responses** - Every answer is backed by retrieved documents  
✅ **Explicit coverage gaps** - Missing information is surfaced, not fabricated  
✅ **Observable retrieval** - Inspect what the system found and why  
✅ **Auditable behavior** - Predictable, traceable decision-making  

Perfect for **AI Engineering roles**, compliance-heavy domains, and production RAG systems.

---

## ✨ Key Features

- 🔒 **Strict document grounding** - No hallucinations, no general knowledge fallback
- 🤖 **Agent-based routing** - Intelligent query classification (policy, medical, device, membership)
- 🔍 **Hybrid retrieval** - BM25 (lexical) + vector search (semantic)
- 🐛 **Retrieval debugger UI** - Inspect chunks, metadata, and retrieval decisions
- ✅ **Integration tests** - Real embeddings, real vector store, real LLM calls
- 📊 **LangSmith tracing** - Optional observability for production monitoring
- 💬 **Streamlit interface** - Chat assistant + debug mode in one app

---

## 🏗️ Architecture Overview

```
User Query
    ↓
LLM Router (classifies intent)
    ↓
Domain Agent (applies category constraints)
    ↓
RAG Pipeline (orchestrates retrieval + generation)
    ↓
Hybrid Retriever (BM25 + Vector Search)
    ↓
Chroma Vector Store
    ↓
Grounded Prompt Construction
    ↓
Answer OR Explicit Refusal
```

### Design Principles

- **Agents don't retrieve** - They only enforce domain-specific constraints
- **Decoupled architecture** - Retrieval and generation are separate concerns
- **No knowledge fallback** - Intentional design to prevent hallucinations
- **Metadata-rich chunks** - Every chunk is traceable to source document and category

---

## 📂 Project Structure

```
src/rag/
├── agents.py                 # Router + domain-specific agents
├── rag_pipeline.py          # Core grounded RAG logic
├── tracing.py               # LangSmith integration (optional)
├── settings.py              # Centralized configuration
├── retrievers/
│   ├── hybrid.py           # BM25 retriever implementation
│   ├── production.py       # Hybrid retrieval orchestration
│   └── vectorstore.py      # Chroma vector store loader
├── scripts/
│   └── ingestion.py        # Document ingestion & chunking
├── tests/
│   ├── test_rag.py        # RAG pipeline integration tests
│   └── test_agents.py     # Agent routing tests
└── ui/
    └── app.py             # Streamlit UI (Chat + Debugger)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key (or compatible LLM API)
- Git

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd grounded-rag-assistant
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=your_key_here
# LANGCHAIN_API_KEY=your_key_here (optional, for tracing)
```

### 3. Ingest Documents

Place your documents in category-specific folders (e.g., `data/policies/`, `data/medical/`), then run:

```bash
python src/rag/scripts/ingestion.py
```

**Note:** Documents are stored locally and not committed to Git. The folder name becomes the document category.

### 4. Run Tests

```bash
# Test RAG pipeline
python -m rag.tests.test_rag

# Test agent routing
python -m rag.tests.test_agents
```

### 5. Launch Application

```bash
streamlit run src/rag/ui/app.py
```

Navigate to `http://localhost:8501` to access the UI.

---

## 📥 Document Ingestion

Documents are processed with rich metadata for traceability:

- **Category**: Derived from folder name (`policies`, `medical`, `devices`, `membership`)
- **Document name**: Original filename
- **Chunk ID**: Unique identifier for each text chunk

### Example Metadata Structure

```json
{
  "category": "policies",
  "document_name": "Privacy_Policy.pdf",
  "chunk_id": "policies__Privacy_Policy.pdf__chunk_12",
  "page": 5
}
```

This enables:
- Category-constrained retrieval
- Chunk-level source attribution
- Precise debugging and auditing

---

## 🔍 Retrieval Strategy

The system uses **hybrid retrieval** for optimal coverage:

| Method | Purpose | Strengths |
|--------|---------|-----------|
| **BM25** | Lexical matching | Exact terms, acronyms, IDs |
| **Vector Search** | Semantic similarity | Conceptual matches, paraphrasing |

**Category filtering** is enforced at retrieval time to ensure domain-specific results.

**No reranking or summarization** - Kept intentionally transparent and debuggable.

---

## 🤖 Agent System

The system uses lightweight, domain-specific agents:

- **PolicyAgent** - Handles compliance, terms, privacy documents
- **MedicalAgent** - Medical records, clinical information
- **DeviceAgent** - Device specifications, manuals
- **MembershipAgent** - Account, benefits, enrollment queries

An **LLM-based router** classifies each query and directs it to the appropriate agent. All agents share the same RAG pipeline and only differ in category constraints.

---

## 🧪 Testing Philosophy

This project uses **integration-style tests** with real components:

✅ Real embeddings generation  
✅ Real vector store queries  
✅ Real LLM API calls  
✅ No mocks (validates end-to-end behavior)

### Run Tests

```bash
# All tests
python -m pytest src/rag/tests/

# Specific test file
python -m rag.tests.test_rag
python -m rag.tests.test_agents
```

---

## 🖥️ Streamlit Application

### Two Modes

#### 1️⃣ Chat Assistant
- Natural language queries
- Agent-based routing
- Strictly grounded responses
- Source citations

#### 2️⃣ Retrieval Debugger
- Visualize retrieved chunks
- Inspect metadata and scores
- Diagnose missing/incorrect answers
- Understand retrieval decisions

This dual interface makes the system both **user-friendly** and **engineer-friendly**.

---

## 📊 Optional: LangSmith Tracing

Enable production observability with LangSmith:

```bash
# In .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=grounded-rag
```

Track:
- Query classification accuracy
- Retrieval performance
- Agent routing decisions
- End-to-end latency

---

## 🛠️ Configuration

Edit `src/rag/settings.py` to customize:

- **Chunk size** and overlap
- **Number of retrieved chunks**
- **LLM model** and temperature
- **Embedding model**
- **Vector store persistence path**

---

## 🎯 Use Cases

This architecture is ideal for:

- **Enterprise knowledge bases** - Internal documentation, policies, procedures
- **Regulated industries** - Healthcare, finance, legal (where accuracy is critical)
- **Technical support** - Product manuals, troubleshooting guides
- **Compliance systems** - Policy enforcement, audit trail requirements

---

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- Additional document loaders (Word, Excel, etc.)
- Reranking and hybrid fusion strategies
- Multi-language support
- Advanced metadata filtering
- Performance benchmarking suite

---

## 📄 License

[Choose your license - MIT, Apache 2.0, etc.]

---

## 🙏 Acknowledgments

Built with:
- [LangChain](https://langchain.com/) - RAG orchestration
- [Chroma](https://www.trychroma.com/) - Vector database
- [Streamlit](https://streamlit.io/) - UI framework
- [OpenAI](https://openai.com/) - LLM and embeddings

---

**Yajurved Jayavarapu**  
Data Scientist 

📧 yjayavarapu@gmail.com  
💼 [LinkedIn](https://www.linkedin.com/in/yajurved-jayavarapu/)   
📂 [GitHub](https://github.com/yajurved987)


📧 yjayavarapu@gmail.com
💼 LinkedIn
📂 GitHub

---

**⭐ If you find this project useful, please consider giving it a star!**
