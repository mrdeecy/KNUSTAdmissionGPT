"""Fixtures for production RAG tests; optional evaluators load only for live tests."""
import json
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rag_deploy  # Initialize Windows certificate trust before evaluator SDKs.

load_dotenv(REPO_ROOT / ".env")

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "llm: calls a live LLM/vector DB (slow, costs money; run explicitly with -m llm)"
    )


def load_golden_set():
    return json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def golden_set():
    return load_golden_set()


@pytest.fixture(scope="session")
def retriever():
    from rag_deploy import retrieval_pipeline as rag

    return rag.retriever


@pytest.fixture(scope="session")
def api_url():
    import requests

    url = os.getenv("KNUST_API_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        response = requests.get(f"{url}/health", timeout=5)
        response.raise_for_status()
        assert response.json().get("status") == "ok"
    except (requests.RequestException, ValueError, AssertionError) as exc:
        pytest.fail(f"Start the production FastAPI backend before live evaluation: {url}. {exc}")
    return url


@pytest.fixture(scope="session")
def ragas_metrics():
    """Build each RAGAS metric once per test session — the judge LLM and
    embedding model have real setup cost we don't want to pay per test."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall
    from ragas.metrics._faithfulness import Faithfulness

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=os.getenv("CHAT_MODEL", "gpt-4o-mini"), temperature=0)
    )
    evaluator_emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    )
    return {
        "context_precision": LLMContextPrecisionWithReference(llm=evaluator_llm),
        "context_recall": LLMContextRecall(llm=evaluator_llm),
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_emb),
    }
