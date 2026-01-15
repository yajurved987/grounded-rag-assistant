from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI

from rag.settings import settings
from rag.rag_pipeline import RAGPipeline



# Base Agent
class BaseAgent:
    """
    Base agent that wraps the RAG pipeline
    with an optional category constraint.
    """

    def __init__(self, category: Optional[str] = None):
        self.category = category
        self.rag = RAGPipeline()

    def run(self, query: str) -> Dict[str, Any]:
        return self.rag.run(
            query=query,
            category=self.category
        )



# Domain Agents
class PolicyAgent(BaseAgent):
    def __init__(self):
        super().__init__(category="policies")


class MedicalAgent(BaseAgent):
    def __init__(self):
        super().__init__(category="medical")


class DeviceAgent(BaseAgent):
    def __init__(self):
        super().__init__(category="device")


class MembershipAgent(BaseAgent):
    def __init__(self):
        super().__init__(category="membership")


# Router
class QueryRouter:
    """
    Routes a query to the appropriate domain agent.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            temperature=0,
            api_key=settings.OPENAI_API_KEY
        )

        self.agents = {
            "policies": PolicyAgent(),
            "medical": MedicalAgent(),
            "device": DeviceAgent(),
            "membership": MembershipAgent(),
        }

    def route(self, query: str) -> Optional[BaseAgent]:
        prompt = f"""
Classify the user query into ONE category:

- policies
- medical
- device
- membership

Return ONLY the category name.

Query:
{query}
"""

        category = self.llm.invoke(prompt).content.strip().lower()
        return self.agents.get(category)



# Agent System (Single Entry Point)
class AgentSystem:
    """
    High-level agent system that routes queries
    and executes the correct agent.
    """

    def __init__(self):
        self.router = QueryRouter()

    def run(self, query: str) -> Dict[str, Any]:
        agent = self.router.route(query)

        if agent is None:
            return {
                "answer": "I could not determine the domain of the question.",
                "documents": [],
                "context": ""
            }

        return agent.run(query)
