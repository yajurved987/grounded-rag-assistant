import streamlit as st

from rag.tracing import init_tracing
from rag.rag_pipeline import RAGPipeline
from rag.agents import AgentSystem



# Init tracing ONCE at startup
init_tracing()



# App Config
st.set_page_config(
    page_title="RAG Assistant",
    layout="wide"
)

st.title("🤖 RAG Assistant")
st.caption("Chat assistant with retrieval debugger & tracing")



# Load systems (cached)
@st.cache_resource
def load_rag():
    return RAGPipeline()

@st.cache_resource
def load_agents():
    return AgentSystem()

rag = load_rag()
agent_system = load_agents()



# Sidebar controls
mode = st.sidebar.radio(
    "Select Mode",
    ["Chat Assistant", "Retrieval Debugger"]
)

category = st.sidebar.selectbox(
    "Optional category filter",
    options=[None, "policies", "medical", "device", "membership"],
    index=0
)


# Chat Assistant Mode (Agents)
if mode == "Chat Assistant":
    st.subheader("💬 Chat Assistant")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    user_query = st.chat_input("Ask a question...")

    if user_query:
        st.session_state["messages"].append(
            {"role": "user", "content": user_query}
        )

        result = agent_system.run(user_query)

        st.session_state["messages"].append(
            {"role": "assistant", "content": result["answer"]}
        )

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


# Retrieval Debugger Mode (Raw RAG)
else:
    st.subheader("🔍 Retrieval Debugger")

    query = st.text_input(
        "Enter your question",
        placeholder="e.g. What is the privacy policy?"
    )

    if st.button("Run Debugger") and query:
        result = rag.run(query=query, category=category)

        st.subheader("🧠 Answer")
        st.info(result["answer"])

        st.subheader("📄 Retrieved Chunks")

        if not result["documents"]:
            st.warning("No documents retrieved.")
        else:
            for i, meta in enumerate(result["documents"], 1):
                with st.expander(
                    f"Chunk {i} — {meta['document_name']}"
                ):
                    st.write(f"**Category:** {meta['category']}")
                    st.write(f"**Chunk ID:** {meta['chunk_id']}")
