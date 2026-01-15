import os
from pathlib import Path
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from rag.settings import settings


DATA_DIR = Path("./data") 
CHROMA_DIR = settings.CHROMA_DIR

#Load documents
def load_documents(data_dir: Path) -> List[Document]:
    documents: List[Document] = []

    if not data_dir.exists():
        raise RuntimeError(f"Data directory not found: {data_dir}")

    for category_dir in data_dir.iterdir():
        if not category_dir.is_dir():
            continue  # ignore files at root level

        category = category_dir.name.lower()

        for file_path in category_dir.glob("*.pdf"):
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()

            for doc in docs:
                doc.metadata.update({
                    "category": category,
                    "document_name": file_path.name,
                    "source": "local",
                })

            documents.extend(docs)

    return documents

#chunking
def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = (
            f"{chunk.metadata['category']}__"
            f"{chunk.metadata['document_name']}__"
            f"chunk_{idx}"
        )

    return chunks

def persist_chunks(chunks: List[Document]) -> None:
    embeddings = OpenAIEmbeddings(
        model=settings.EMBED_MODEL,
        api_key=settings.OPENAI_API_KEY
    )

    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    db.add_documents(chunks)

def main():
    print("Starting ingestion")

    documents = load_documents(DATA_DIR)
    print(f"Loaded {len(documents)} document pages")

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    persist_chunks(chunks)
    print("Ingestion complete — ChromaDB updated")


if __name__ == "__main__":
    main()
