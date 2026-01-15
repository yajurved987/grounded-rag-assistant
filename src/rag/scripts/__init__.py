import os
import re
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.settings import settings


# -------------------------------------------------------------
# PATHS
# -------------------------------------------------------------
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CHROMA_DIR = settings.CHROMA_DIR

print("📂 DATA_DIR =", DATA_DIR)
print("📁 Exists?", os.path.exists(DATA_DIR))

summary_llm = ChatOpenAI(model=settings.MODEL_NAME, api_key=settings.OPENAI_API_KEY)

embeddings = OpenAIEmbeddings(
    model=settings.EMBED_MODEL,
    api_key=settings.OPENAI_API_KEY
)


# -------------------------------------------------------------
# CLEANING FUNCTION
# -------------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"Page \d+|page \d+", "", text)
    text = re.sub(r"[^\S\r\n]{2,}", " ", text)
    return text.strip()


# -------------------------------------------------------------
# METADATA EXTRACTION
# -------------------------------------------------------------
def extract_metadata(root: str, file_name: str):
    """
    Extract metadata based on folder structure:
    data/<category>/<sub_category>/<year>/file.pdf
    """

    parts = Path(root).parts
    # Example parts = (..., "data", "Device", "Specs", "2024")

    category = parts[-3] if len(parts) >= 3 else ""
    sub_category = parts[-2] if len(parts) >= 2 else ""
    year = parts[-1] if parts[-1].isdigit() else ""

    return {
        "category": category,
        "sub_category": sub_category,
        "year": year,
        "file_name": file_name,
    }


# -------------------------------------------------------------
# LOAD PDF / TXT DOCUMENTS
# -------------------------------------------------------------
def load_all_documents():
    all_docs = []

    print("🔍 Scanning data directory...")

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            ext = file.lower()

            if ext.endswith(".pdf"):
                loader = PyPDFLoader(full_path)
            elif ext.endswith(".txt"):
                loader = TextLoader(full_path)
            else:
                continue

            print(f"📄 Loading: {full_path}")
            docs = loader.load()
            metadata = extract_metadata(root, file)

            for d in docs:
                page_num = d.metadata.get("page", None)
                d.metadata.update({**metadata, "page_number": page_num})

            all_docs.extend(docs)

    return all_docs


# -------------------------------------------------------------
# LLM SUMMARY (SAFE MODE)
# -------------------------------------------------------------
def summarize_text(text: str):
    try:
        response = summary_llm.invoke(
            f"Summarize this text in 1–2 sentences:\n\n{text}"
        )
        return response.content
    except Exception as e:
        print("⚠ Summary failed:", str(e))
        return ""


# -------------------------------------------------------------
# CHUNKING FUNCTION
# -------------------------------------------------------------
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "?", "!", ";", ","],
    )

    chunks = []

    for doc in documents:
        cleaned = clean_text(doc.page_content)
        if not cleaned.strip():
            continue

        text_chunks = splitter.split_text(cleaned)
        parent_id = str(uuid.uuid4())

        for i, chunk_text in enumerate(text_chunks):
            chunk_id = str(uuid.uuid4())

            summary = summarize_text(chunk_text)

            metadata = {
                **doc.metadata,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "parent_id": parent_id,
                "summary": summary,
            }

            chunks.append(Document(page_content=chunk_text, metadata=metadata))

    return chunks


# -------------------------------------------------------------
# STORE IN CHROMA
# -------------------------------------------------------------
def store_in_chroma(chunks):
    print(f"💽 Storing {len(chunks)} chunks into ChromaDB...")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    vectordb.persist()
    print("✅ ChromaDB updated successfully!")


# -------------------------------------------------------------
# MAIN INGEST PIPELINE
# -------------------------------------------------------------
def ingest():
    print("\n🚀 Loading documents...")
    docs = load_all_documents()
    print(f"📚 Loaded {len(docs)} raw pages")

    print("\n🔪 Chunking + summarizing...")
    chunks = chunk_documents(docs)
    print(f"🧩 Created {len(chunks)} chunks")

    print("\n💾 Saving embeddings...")
    store_in_chroma(chunks)

    print("\n🎉 INGESTION COMPLETE!\n")


# -------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------
if __name__ == "__main__":
    ingest()
