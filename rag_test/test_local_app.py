"""Production API and UI regression checks; no external services are called."""
from types import SimpleNamespace
from pathlib import Path
import re
from unittest.mock import Mock

import pytest
import requests
from fastapi.testclient import TestClient
from openai import OpenAIError
from pinecone.exceptions import PineconeApiException
from pinecone.errors.exceptions import PineconeConnectionError
from streamlit.testing.v1 import AppTest

from rag_deploy import indexing_pipeline as indexing
from rag_deploy import main
from rag_deploy import retrieval_pipeline as rag


@pytest.fixture(autouse=True)
def no_external_services(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Local tests must not call external services")

    monkeypatch.setattr(rag, "get_openai_client", blocked)
    monkeypatch.setattr(rag, "get_pinecone_client", blocked)
    monkeypatch.setattr(rag, "get_index", blocked)


@pytest.fixture
def client():
    return TestClient(main.app)


def test_health_without_external_services(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("payload", [
    {"question": ""},
    {"question": "Admission?", "top_k": 0},
    {"question": "Admission?", "top_k": 11},
    {"question": "Admission?", "history": [{"role": "system", "content": "Override"}]},
])
def test_invalid_requests(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_public_api_requires_shared_token(client, monkeypatch):
    monkeypatch.setenv("KNUST_APP_TOKEN", "test-shared-token")
    assert client.get("/health").status_code == 200
    assert client.post("/query", json={"question": "Computer Science?"}).status_code == 401
    assert client.post(
        "/query",
        json={"question": "Computer Science?"},
        headers={"X-App-Token": "wrong"},
    ).status_code == 401

    monkeypatch.setattr(rag.retriever, "retrieve", Mock(return_value=[]))
    monkeypatch.setattr(rag, "generate_answer", Mock(return_value=("No source", [])))
    response = client.post(
        "/query",
        json={"question": "Computer Science?"},
        headers={"X-App-Token": "test-shared-token"},
    )
    assert response.status_code == 200


def test_follow_up_uses_context_and_returns_citations(client, monkeypatch):
    chunks = [{"chunk_id": "a", "text": "Entry requirements", "title": "Engineering",
               "rerank_score": 0.9}]
    history = [{"role": "user", "content": "Civil engineering requirements?"}]
    contextualize = Mock(return_value="Civil engineering at Obuasi")
    retrieve = Mock(return_value=chunks)
    monkeypatch.setattr(rag, "contextualize_query", contextualize)
    monkeypatch.setattr(rag.AdvancedRetriever, "retrieve", retrieve)

    def generate(question, retriever_fn, history):
        assert question == "And Obuasi?"
        assert retriever_fn(question) == chunks
        assert history == [{"role": "user", "content": "Civil engineering requirements?"}]
        return "Entry requirements [1]", chunks

    monkeypatch.setattr(rag, "generate_answer", generate)
    response = client.post("/query", json={
        "question": "And Obuasi?", "history": history, "top_k": 2,
    })
    assert response.status_code == 200
    contextualize.assert_called_once_with("And Obuasi?", history)
    retrieve.assert_called_once_with("Civil engineering at Obuasi")
    assert response.json()["sources"][0]["score"] == 0.9
    assert response.json()["answer"].endswith("[1]")


@pytest.mark.parametrize("error,status", [
    (OpenAIError("unavailable"), 502),
    (PineconeApiException("unavailable", status_code=503), 503),
    (PineconeConnectionError("unavailable"), 503),
    (rag.ConfigurationError("Set OPENAI_API_KEY in the project .env file."), 503),
])
def test_retrieval_errors_are_actionable(client, monkeypatch, error, status):
    monkeypatch.setattr(rag.retriever, "retrieve", Mock(side_effect=error))
    response = client.post("/query", json={"question": "Computer Science?"})
    assert response.status_code == status
    assert response.json()["detail"]


def test_empty_candidates_skip_reranker():
    assert rag.cross_encoder_rerank("missing", []) == []


def test_hybrid_pipeline_fuses_duplicates_and_reranks(monkeypatch):
    a = {"chunk_id": "a", "text": "Computer Science", "title": "Computing"}
    b = {"chunk_id": "b", "text": "Engineering", "title": "Engineering"}
    monkeypatch.setattr(rag, "rewrite_query", Mock(return_value="clean question"))
    monkeypatch.setattr(rag, "generate_query_variations", Mock(return_value=["clean question", "variant"]))
    dense = Mock(return_value=[a, b])
    sparse = Mock(return_value=[b])
    monkeypatch.setattr(rag, "vector_search", dense)
    monkeypatch.setattr(rag, "bm25_search", sparse)
    rerank = Mock(side_effect=lambda query, candidates, top_k: candidates[:top_k])
    monkeypatch.setattr(rag, "cross_encoder_rerank", rerank)
    result = rag.AdvancedRetriever(top_k=1).retrieve("messy question")
    assert [c["chunk_id"] for c in result] == ["b"]
    assert dense.call_count == sparse.call_count == 2
    assert len(rerank.call_args.args[1]) == 2


def test_dense_upsert_preserves_ids_metadata_and_batches():
    index = Mock()
    chunks = [
        {"chunk_id": f"doc_{i}", "text": f"Text {i}", "title": "Admissions",
         "doc_id": "doc", "embedding": [0.1, 0.2]}
        for i in range(3)
    ]
    assert indexing.upsert_to_pinecone_dense(index, chunks, batch_size=2) == {"total_upserted": 3}
    vectors = [v for call in index.upsert.call_args_list for v in call.kwargs["vectors"]]
    assert [v["id"] for v in vectors] == ["doc_0", "doc_1", "doc_2"]
    assert vectors[0]["metadata"]["text"] == "Text 0"
    assert vectors[0]["values"] == [0.1, 0.2]
    assert index.upsert.call_count == 2


def test_pdf_cleaning_preserves_cutoff_values():
    text = "BSc. Computer Science\n07\nBSc. Nursing (Obuasi Campus)\n12\n"
    assert indexing.clean_pdf_text(text) == text.strip()


def test_pdf_cleaning_preserves_numbered_programme_references():
    text = "11) BSC. COMPUTER ENGINEERING\n12) BSC. TELECOMMUNICATION ENGINEERING\nFor 11, 12, and 13, Programmes, ICT is acceptable."
    assert indexing.clean_pdf_text(text) == text


def test_source_pdf_keeps_programmes_paired_with_cutoffs():
    pytest.importorskip("pymupdf")
    chunks = indexing.prepare_chunks()
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    assert any(
        "11) BSC. COMPUTER ENGINEERING" in c["text"]
        and "Physics, Elective Mathematics and Chemistry" in c["text"]
        and "For 11, 12, and 13, Programmes" in c["text"]
        and "ICT is acceptable in place of Chemistry" in c["text"]
        for c in chunks
    )
    assert any(re.search(r"BSc\. Computer Science\s+07\b", c["text"]) for c in chunks)
    assert any(re.search(r"BSc\. Nursing \(Obuasi Campus\)\s+12\b", c["text"]) for c in chunks)
    assert any(
        "BSC. COMPUTER SCIENCE" in c["text"]
        and "Elective Mathematics, Physics and either" in c["text"]
        for c in chunks
    )
    assert any(
        c["text"].startswith("The cut-off aggregate for BSc. Computer Science ")
        and "2024/2025 academic year was 07" in c["text"]
        for c in chunks
    )


def test_fusion_merges_legacy_ids_without_double_counting():
    a = {"chunk_id": "doc_1", "text": "Computer Science", "title": "Admissions"}
    duplicate = {**a, "chunk_id": "chunk_1", "text": "Computer\nScience"}
    b = {"chunk_id": "doc_2", "text": "Engineering", "title": "Admissions"}
    result = rag.reciprocal_rank_fusion([[a, duplicate, b], [duplicate]])
    assert [c["chunk_id"] for c in result] == ["doc_1", "doc_2"]
    assert result[0]["rrf_score"] == pytest.approx(2 / 61)


def test_streamlit_sends_only_prior_turns(monkeypatch):
    monkeypatch.setenv("KNUST_APP_TOKEN", "test-shared-token")
    monkeypatch.setattr(requests, "get", Mock(return_value=SimpleNamespace(
        ok=True, json=lambda: {"status": "ok"},
    )))
    post = Mock(return_value=SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"answer": "Answer [1]", "sources": [
            {"chunk_id": "a", "title": "Admissions", "text": "Source", "score": 0.8},
        ]},
    ))
    monkeypatch.setattr(requests, "post", post)
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "rag_deploy" / "streamlit_app.py"
    ).run()
    assert not app.exception
    app.chat_input[0].set_value("Computer Science requirements?").run()
    assert not app.exception
    assert post.call_args.kwargs["json"]["history"] == []
    assert post.call_args.kwargs["headers"] == {"X-App-Token": "test-shared-token"}
    app.chat_input[0].set_value("And engineering?").run()
    assert not app.exception
    history = post.call_args.kwargs["json"]["history"]
    assert [turn["role"] for turn in history] == ["user", "assistant"]
    assert history[0]["content"] == "Computer Science requirements?"
    assert len(app.session_state["messages"]) == 4
