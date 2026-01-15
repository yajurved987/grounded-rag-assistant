from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def build_bm25_retriever(documents: List[Document], k: int = 6) -> BM25Retriever:
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = k
    return bm25
