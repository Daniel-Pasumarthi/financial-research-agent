from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from app.retriever_custom import retrieve_custom
from app.tools import get_financials


class AgentState(TypedDict):
    ticker: str
    query: str
    retrieval_backend: str  # "custom" or "bedrock"
    context: List[str]
    financials: dict
    answer: str

def retrieve_node(state: AgentState) -> dict:
    """Route to the chosen retrieval backend."""
    if state["retrieval_backend"] == "bedrock":
        from app.retriever_bedrock import retrieve_bedrock
        context = retrieve_bedrock(state["query"])
    else:
        context = retrieve_custom(state["query"], ticker=state["ticker"], top_n=4)
    return {"context": context}


def financials_node(state: AgentState) -> dict:
    return {"financials": get_financials(state["ticker"])}


def synthesize_node(state: AgentState) -> dict:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    context_text = "\n\n".join(state["context"])

    prompt = f"""You are a financial research assistant.

Using ONLY the filing context and financial data below, write a
concise research summary covering revenue trends, key risks, and
overall sentiment. If something is not supported by the context,
say so explicitly.

FILING CONTEXT:
{context_text}

FINANCIAL DATA:
{state["financials"]}

QUESTION: {state["query"]}"""

    return {"answer": llm.invoke(prompt).content}

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("financials", financials_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "financials")
    graph.add_edge("financials", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()