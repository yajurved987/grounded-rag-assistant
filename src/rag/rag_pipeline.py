from typing import Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from rag.settings import settings
from rag.retrievers.production import ProductionRetriever


class RAGPipeline:
    """
    RAG pipeline:
    - Uses ProductionRetriever for retrieval
    - Builds grounded prompt
    - Calls LLM for final answer
    """

    def __init__(self):
        self.retriever = ProductionRetriever()

        self.llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            temperature=0,
            api_key=settings.OPENAI_API_KEY
        )

    def _build_context(self, docs: List[Document]) -> str:
        """
        Build context string from retrieved documents.
        """
        context_blocks = []

        for d in docs:
            block = (
                f"[Source: {d.metadata.get('document_name')} | "
                f"Category: {d.metadata.get('category')}]\n"
                f"{d.page_content}"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def run(
        self,
        query: str,
        category: Optional[str] = None,
        k: int = 6
    ) -> Dict[str, Any]:

        #Retrieve documents
        retrieved_docs = self.retriever.retrieve(
            query=query,
            category=category,
            k=k
        )

        if not retrieved_docs:
            return {
                "answer": "I could not find relevant information in the documents.",
                "documents": [],
                "context": ""
            }

        #Build context
        context = self._build_context(retrieved_docs)

        #Build prompt (grounded)
        prompt = f"""
You are an AI assistant answering questions strictly from internal documents.

Rules:
- Use ONLY the provided context.
- Do NOT use outside knowledge.
- If the answer is not present, say:
  "I cannot find this information in the provided documents."

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

        #LLM call
        response = self.llm.invoke(prompt)

        return {
            "answer": response.content.strip(),
            "documents": [d.metadata for d in retrieved_docs],
            "context": context
        }
