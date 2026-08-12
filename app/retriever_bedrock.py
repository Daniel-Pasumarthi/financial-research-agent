import os
import boto3
from dotenv import load_dotenv

load_dotenv()

_client = boto3.client(
    "bedrock-agent-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
)
KB_ID = os.getenv("BEDROCK_KB_ID")


def retrieve_bedrock(query: str, ticker: str, top_n: int = 4):
    """Retrieve chunks from a Bedrock Knowledge Base, filtered to one ticker."""
    try:
        response = _client.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "managedSearchConfiguration": {
                    "numberOfResults": top_n,
                    "filter": {
                        "equals": {"key": "ticker", "value": ticker}
                    },
                }
            },
        )
        return [
            r["content"]["text"]
            for r in response["retrievalResults"]
        ]
    except Exception as e:
        return [f"Bedrock retrieval error: {e}"]