"""FastAPI backend for the KNUST Admissions Assistant.
Wraps `retrieval_pipeline.py` (AdvancedRetriever + generate_answer)
behind a small REST API: `POST /query` and `GET /health`. This is the only
thing the deployment platform runs in production, and the only thing the
frontend UI is allowed to talk to — the UI never imports `retrieval_pipeline`
directly.
"""
import logging
import os
from secrets import compare_digest
from typing import List, Literal, Optional
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAIError
from pinecone.exceptions import PineconeApiException
from pinecone.errors.exceptions import PineconeConnectionError
from pydantic import BaseModel, Field
from . import retrieval_pipeline as rag

logger = logging.getLogger("KNUST_ADMISSIONS_api")
app = FastAPI(title="KNUST Admissions Assistant API", version="1.0.0")

# The frontend UI runs on a different origin (localhost during development,
# a different domain once deployed), so the browser needs CORS headers to
# allow the cross-origin fetch from the UI -> this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="'user' or 'assistant'.")
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The applicant's question.")
    top_k: Optional[int] = Field(
        default=None, ge=1, le=10,
        description="Number of cited chunks to return. Defaults to the retriever's configured top_k.",
    )
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Prior turns in the conversation, most recent last. "
                     "Used to resolve follow-up questions like 'what about the Obuasi campus version?'",
    )


class Source(BaseModel):
    chunk_id: str
    title: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


class HealthResponse(BaseModel):
    status: str


@app.exception_handler(rag.ConfigurationError)
async def configuration_error_handler(request, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, x_app_token: Optional[str] = Header(default=None)) -> QueryResponse:
    expected_token = os.getenv("KNUST_APP_TOKEN")
    if expected_token and not compare_digest(x_app_token or "", expected_token):
        raise HTTPException(status_code=401, detail="Invalid application token.")
    retriever = rag.AdvancedRetriever(top_k=request.top_k) if request.top_k else rag.retriever
    history = [h.model_dump() for h in request.history] if request.history else None

    # Resolve conversational references ("what about the Obuasi campus
    # version?") into a standalone question BEFORE retrieval, so the
    # retriever isn't searching for the literal follow-up phrase.
    try:
        search_query = (
            rag.contextualize_query(request.question, history)
            if history else request.question
        )
    except OpenAIError as exc:
        logger.exception("Query contextualization failed")
        raise HTTPException(
            status_code=502,
            detail="The language model backend failed to respond.",
        ) from exc

    try:
        chunks = retriever.retrieve(search_query)
    except OpenAIError as exc:
        logger.exception("Query rewriting or embedding failed")
        raise HTTPException(
            status_code=502,
            detail="The language model backend failed to respond.",
        ) from exc
    except (PineconeApiException, PineconeConnectionError) as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(
            status_code=503,
            detail="Retrieval backend unavailable. Try again shortly.",
        ) from exc

    try:
        # Hand the already-retrieved chunks back through generate_answer's own
        # retriever_fn hook so the grounded-answer prompt and citation format
        # stay in one place (retrieval_pipeline.py) instead of being copied here.
        # Note: the ORIGINAL question (not search_query) goes to generation,
        # so the model responds naturally to what the user actually typed;
        # `history` gives it the conversational context to do so correctly.
        answer, _ = rag.generate_answer(
            request.question, retriever_fn=lambda _q: chunks, history=history,
        )
    except OpenAIError as exc:
        logger.exception("Generation failed")
        raise HTTPException(
            status_code=502,
            detail="The language model backend failed to respond.",
        ) from exc

    sources = [
        Source(
            chunk_id=c["chunk_id"],
            title=c.get("title", ""),
            text=c["text"],
            score=c.get("rerank_score", c.get("rrf_score", c.get("score", 0.0))),
        )
        for c in chunks
    ]

    return QueryResponse(answer=answer, sources=sources)
