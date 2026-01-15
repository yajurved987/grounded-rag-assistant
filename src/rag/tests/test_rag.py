from rag.rag_pipeline import RAGPipeline

rag = RAGPipeline()

result = rag.run(
    query="what are the privacy policies?",
    category="policies"
)

print("\nANSWER:\n", result["answer"])
print("\nDOCUMENTS:\n", result["documents"])
