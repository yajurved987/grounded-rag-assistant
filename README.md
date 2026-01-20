# Grounded RAG Assistant with Agents & Retrieval Debugger

This project is a **production-style Retrieval-Augmented Generation (RAG) system** that answers questions **strictly from internal documents**.  
If the requested information is not present in the documents, the system **intentionally refuses to answer** to prevent hallucinations.

The system is designed to prioritize **correctness, traceability, and debuggability**, mirroring real-world enterprise and regulated-domain RAG systems.

---

## 🔍 Why This Project

Many RAG demos prioritize “always answering,” even when the underlying documents do not support the response.  
This project intentionally takes a different approach:

- Answers are **fully grounded in retrieved documents**
- Missing coverage is **explicitly surfaced**
- Retrieval behavior is **observable and debuggable**
- System behavior is **predictable and auditable**

This makes the project suitable for **AI Engineer roles** and enterprise-style use cases.

---

## ✨ Key Features

- **Strict document-grounded RAG** (no fallback, no hallucinations)
- **Agent-based query routing** (policy, medical, device, membership)
- **Hybrid retrieval** combining BM25 (lexical) and vector search (semantic)
- **Retrieval Debugger UI** to inspect retrieved chunks and metadata
- **Integration tests** for RAG and agent routing
- **Optional LangSmith tracing** for observability
- **Streamlit application** with chat + debug modes

---

## 🏗️ High-Level Architecture


### Design Principles
- Agents **do not retrieve documents** — they only enforce domain constraints
- Retrieval and generation are **decoupled**
- No general-knowledge fallback is used (intentional design decision)

---

## 📁 Project Structure


---

## 📥 Document Ingestion

- Documents are ingested **locally** and are **not committed to Git**
- The **folder name is treated as the document category**
- Each document is:
  - loaded page-by-page
  - chunked using recursive text splitting
  - enriched with structured metadata

### Example metadata
```json
{
  "category": "policies",
  "document_name": "Privacy_Policy.pdf",
  "chunk_id": "policies__Privacy_Policy.pdf__chunk_12"
}
