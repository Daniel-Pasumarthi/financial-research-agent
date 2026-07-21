import streamlit as st
from app.agent import build_agent
from app.evaluation import evaluate_run
import os
from app.ingestion import ingest_filing, CHROMA_DIR

def ensure_store_exists():
    """Build the Chroma store on first run if it doesn't exist yet (fresh clone/container)."""
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        with st.spinner("First run: building vector store from filing..."):
            ingest_filing("data/filings/AAPL_10K_primary.html", "AAPL")

ensure_store_exists()

st.set_page_config(page_title="Financial Research Agent", layout="wide")
st.title("Financial Research Agent")

@st.cache_resource
def get_agent():
    return build_agent()

agent = get_agent()

ticker = st.text_input("Ticker", value="AAPL")
query = st.text_area("Research question", value="What are the main revenue risks?")
backend = st.radio("Retrieval backend", ["custom", "bedrock"], horizontal=True)

if st.button("Run research"):
    with st.spinner("Agent working..."):
        result = agent.invoke({
            "ticker": ticker,
            "query": query,
            "retrieval_backend": backend,
            "context": [],
            "financials": {},
            "answer": "",
        })
    st.session_state["agent_result"] = result
    st.session_state["agent_query"] = query
    st.session_state.pop("eval_scores", None)

if "agent_result" in st.session_state:
    result = st.session_state["agent_result"]

    st.subheader("Research summary")
    st.write(result["answer"])

    with st.expander("Retrieved filing context"):
        for c in result["context"]:
            st.markdown(f"> {c[:300]}...")

    if st.button("Evaluate this response"):
        with st.spinner("Scoring with RAGAS..."):
            scores = evaluate_run(
                query=st.session_state["agent_query"],
                answer=result["answer"],
                contexts=result["context"],
            )
        st.session_state["eval_scores"] = scores

    if "eval_scores" in st.session_state:
        scores = st.session_state["eval_scores"]
        col1, col2 = st.columns(2)
        col1.metric("Faithfulness", f"{scores['faithfulness']:.3f}")
        col2.metric("Context Precision", f"{scores['context_precision']:.3f}")