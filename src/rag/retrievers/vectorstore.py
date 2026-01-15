from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from rag.settings import settings


def load_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model=settings.EMBED_MODEL,
        api_key=settings.OPENAI_API_KEY
    )

    return Chroma(
        persist_directory=settings.CHROMA_DIR,
        embedding_function=embeddings
    )
