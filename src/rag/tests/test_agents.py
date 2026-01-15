from rag.agents import AgentSystem


def test_agent_routing_and_execution():
    """
    Integration test for router + agents + RAG pipeline.
    """

    agent_system = AgentSystem()

    test_cases = [
        {
            "query": "What is the privacy policy?",
            "expected_category": "policies"
        },
        {
            "query": "How is medical data processed?",
            "expected_category": "medical"
        },
        {
            "query": "Tell me about the device accuracy",
            "expected_category": "device"
        },
        {
            "query": "What are the membership benefits?",
            "expected_category": "membership"
        }
    ]

    for case in test_cases:
        print("\n" + "=" * 60)
        print(f"QUERY: {case['query']}")

        result = agent_system.run(case["query"])

        # Basic assertions
        assert "answer" in result
        assert "documents" in result

        print("ANSWER:")
        print(result["answer"])

        if result["documents"]:
            categories = {d["category"] for d in result["documents"]}
            print("CATEGORIES FOUND:", categories)

            # Category constraint check
            assert case["expected_category"] in categories
        else:
            print(" No documents retrieved (acceptable if data missing)")

if __name__ == "__main__":
    test_agent_routing_and_execution()