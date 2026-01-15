import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT","RAG")


    ENVIRONMENT:str = os.getenv("ENVIRONMENT","Development")
    CHROMA_DIR:str = os.getenv("CHROMA_DIR","./storage/chroma_db")
    MODEL_NAME:str = os.getenv("MODEL_NAME","gpt-3.5-turbo")
    EMBED_MODEL:str = os.getenv("EMBED_MODEL","text-embedding-3-small")
settings= Settings()


