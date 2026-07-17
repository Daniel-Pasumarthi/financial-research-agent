import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, ContextPrecisionWithoutReference

load_dotenv()

_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_judge_llm = llm_factory("gpt-4o", client=_client, max_tokens=4096)


async def _score_run_async(query: str, answer: str, contexts: list[str]) -> dict:
    """Score one agent run for faithfulness and context precision."""
    faithfulness = Faithfulness(llm=_judge_llm)
    context_precision = ContextPrecisionWithoutReference(llm=_judge_llm)

    faith_result = await faithfulness.ascore(
        user_input=query,
        response=answer,
        retrieved_contexts=contexts,
    )
    precision_result = await context_precision.ascore(
        user_input=query,
        response=answer,
        retrieved_contexts=contexts,
    )

    return {
        "faithfulness": faith_result.value,
        "faithfulness_reason": faith_result.reason,
        "context_precision": precision_result.value,
        "context_precision_reason": precision_result.reason,
    }


def evaluate_run(query: str, answer: str, contexts: list[str]) -> dict:
    """Sync wrapper — call this from anywhere that isn't already async (e.g. Streamlit)."""
    return asyncio.run(_score_run_async(query, answer, contexts))


if __name__ == "__main__":
    test_query = "What are the main revenue risks?"
    test_answer = "Apple faces risks from foreign exchange fluctuations and reliance on iPhone sales."
    test_contexts = [
        "The Company's business is subject to numerous risks, including foreign exchange rate fluctuations...",
        "A significant portion of the Company's net sales has come from the iPhone...",
    ]
    scores = evaluate_run(test_query, test_answer, test_contexts)
    print(scores)