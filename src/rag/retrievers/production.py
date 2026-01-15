from typing import List, Optional

from langchain_core.documents import Document

from rag.retrievers.vectorstore import load_vectorstore
from rag.retrievers.hybrid import build_bm25_retriever


class ProductionRetriever:
    """
    Enterprise-grade retriever:
    - Hybrid search (BM25 + Vector)
    - Optional category filtering
    - Deterministic and debuggable
    """

    def __init__(self):
        self.vectorstore = load_vectorstore()

        # Load all documents ONCE for BM25
        raw = self.vectorstore.get()
        self.documents: List[Document] = [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(raw["documents"], raw["metadatas"])
        ]

        self.bm25 = build_bm25_retriever(self.documents)

    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        k: int = 6
    ) -> List[Document]:

        #Vector search (semantic)
        vector_results = self.vectorstore.similarity_search(
            query,
            k=k,
            filter={"category": category} if category else None
        )

        #BM25 search (keyword)
        bm25_results = self.bm25.invoke(query)

        if category:
            bm25_results = [
                d for d in bm25_results
                if d.metadata.get("category") == category
            ]

        #Merge (simple + deterministic)
        seen = set()
        combined: List[Document] = []

        for doc in vector_results + bm25_results:
            uid = doc.metadata.get("chunk_id")
            if uid and uid not in seen:
                seen.add(uid)
                combined.append(doc)

        return combined[:k]
