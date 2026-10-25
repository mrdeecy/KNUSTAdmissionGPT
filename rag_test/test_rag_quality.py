"""RAG quality regression suite: golden-set questions -> RAGAS assertions.

Every test in this file calls a live LLM and the live Pinecone indexes, so
they're all marked `llm` and excluded from the default `pytest` run (see
`pytest.ini`). Run them explicitly with `python -m pytest -m llm`.

Calls the running production API, including its normal multi-query retrieval.
The evaluator runs in a separate environment because RAGAS 0.3.1 requires
older LangChain integrations. Thresholds allow for stochastic judge scores.
"""
import json
import math

import pytest
import requests
from conftest import load_golden_set

THRESHOLDS = {
    "context_recall": 0.5,
    "faithfulness": 0.5,
    "answer_relevancy": 0.5,
}


@pytest.mark.llm
@pytest.mark.parametrize(
    "item", load_golden_set(), ids=lambda item: item["question"][:60]
)
def test_answer_is_grounded_and_relevant(item, api_url, ragas_metrics, record_property):
    from ragas import SingleTurnSample

    response = requests.post(
        f"{api_url}/query",
        json={"question": item["question"]},
        timeout=180,
    )
    response.raise_for_status()
    body = response.json()
    answer, chunks = body["answer"], body["sources"]
    record_property("question", item["question"])
    record_property("answer", answer)
    record_property("sources", json.dumps(chunks, ensure_ascii=False))

    sample = SingleTurnSample(
        user_input=item["question"],
        retrieved_contexts=[chunk["text"] for chunk in chunks],
        response=answer,
        reference=item["ground_truth"],
    )

    scores = {
        name: ragas_metrics[name].single_turn_score(sample)
        for name in THRESHOLDS
    }
    for name, score in scores.items():
        record_property(name, score)

    failures = [
        f"{name}={score:.2f} < {THRESHOLDS[name]}"
        for name, score in scores.items()
        if not math.isfinite(score) or score < THRESHOLDS[name]
    ]
    assert not failures, (
        f"question: {item['question']!r}\n"
        f"answer: {answer}\n"
        f"failed metrics: {', '.join(failures)}\n"
        f"all scores: {scores}"
    )
