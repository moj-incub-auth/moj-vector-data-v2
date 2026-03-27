# Python imports
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_health import health
from milvus_lib import MilvusKnowledgeBase, ScoredSearchComponent
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

logger = logging.getLogger(f"uvicorn.{__name__}")


def create_knowledge_base() -> MilvusKnowledgeBase:
    """Create a MilvusKnowledgeBase from environment variables.

    Returns:
        A configured MilvusKnowledgeBase instance. Connection is established
        separately via lifespan.
    """
    host = os.getenv("MILVUS_HOST", "localhost")
    port = int(os.getenv("MILVUS_PORT", "19530"))
    collection_name = os.getenv("MILVUS_COLLECTION", "knowledge_base")
    embedding_model = os.getenv(
        "MILVUS_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"
    )
    embedding_dim = int(os.getenv("MILVUS_EMBEDDING_DIM", "1024"))
    max_batch_size = int(os.getenv("MILVUS_MAX_BATCH_SIZE", "2"))
    return MilvusKnowledgeBase(
        host, port, collection_name, embedding_model, embedding_dim, max_batch_size
    )


knowledge_base = create_knowledge_base()


def knowledge_base_status() -> bool:
    """Check if the Milvus knowledge base is healthy and ready for search."""
    return knowledge_base.is_healthy()


async def health_handler(**kwargs) -> Dict[str, Any]:
    """Format health check results for the /health endpoint response."""
    is_success = all(kwargs.values())
    return {
        "status": "success" if is_success else "failure",
        "results": kwargs.items(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage knowledge base connection for the lifetime of the FastAPI app."""
    knowledge_base.connect()
    yield
    knowledge_base.close()


app = FastAPI(
    title="MOJ Design System Search", description="Vector Search API", lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_api_route(
    "/health",
    health(
        [knowledge_base_status],
        success_handler=health_handler,
        failure_handler=health_handler,
    ),
)
Instrumentator(
    excluded_handlers=["/health", "/metrics"],
).instrument(app).expose(app)


@app.get("/")
def read_root():
    """Root endpoint returning a simple greeting."""
    return {"Hello": "World"}


# Based on https://gist.github.com/Aron-v1/f6e58554acf9ef0f328ac93d74dcb9ca
class SearchRequest(BaseModel):
    """Request body for the /search endpoint."""

    message: str
    limit: int = 10
    min_score: float = 0.20


class SearchResponse(BaseModel):
    """Response body for the /search endpoint."""

    message: str
    components: list[ScoredSearchComponent] = []


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """Perform semantic search over design system components."""

    input_guardrails_fired = guardrails(request.message)
    if input_guardrails_fired:
        logger.info("INPUT GUARDRAILS FAILURE")
        return SearchResponse(
            message="Failed Guardrails. I cannot help you with this request. Please be more specific or more polite."
        )

    logger.info(f"Searching for: {request.message}")
    results = knowledge_base.search_components(
        request.message, request.limit, min_score=request.min_score
    )
    logger.info(f"Search results: {results}")
    return SearchResponse(message="Search successful", components=results)


def guardrails(prompt: str) -> bool:

    guardrails_enabled = os.getenv("GUARDRAILS_ENABLED", "False")

    if str(guardrails_enabled).strip().lower() == "true":
        logger.info(f"GUARDRAILS ENABLED: {guardrails_enabled}")

        guardrails_url = os.getenv(
            "GUARDRAILS_GATEWAY", "http://127.0.0.1:8090"
        )  # eg. guardrails-gateway.vllm-serving.svc.cluster.local:8090
        guardrail_type = os.getenv("GUARDRAILS_TYPE", "all")  # all, hype etc.
        guardrails_context = (
            f"{guardrails_url}/{guardrail_type}{'/v1/chat/completions'}"
        )
        guardrails_api_key = os.getenv("GUARDRAILS_API_KEY", "no_api_key")
        guardrails_model = os.getenv("INFERENCE_MODEL", "qwen3-14b-llm")

        logger.debug(f"GUARDRAILS_GATEWAY: {guardrails_url}")
        logger.debug(f"GUARDRAILS_TYPE: {guardrail_type}")
        logger.info("GUARDRAILS LOCATION: " + f"{guardrails_context}")
        logger.debug(f"GUARDRAILS_API_KEY: {guardrails_api_key}")
        logger.debug(f"INFERENCE_MODEL: {guardrails_model}")

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": guardrails_api_key,
            }
            payload = {
                "model": guardrails_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }

            response = requests.post(guardrails_context, headers=headers, json=payload)
        except requests.exceptions.RequestException as e:
            logger.error(f"GUARDRAILS {guardrails_url} CONNECTION or RESPONSE ERROR")
            raise e

        logger.debug("--- Guardrails Response Details ---")
        logger.debug(f"URL: {response.url}")
        logger.debug(f"Status: {response.status_code} ({response.reason})")
        logger.debug("\n--- Body ---")
        if response.ok:
            logger.debug(response.json())
        else:
            logger.debug(f"Error occurred: {response.text}")

        data = response.json()
        detections = data.get("detections")

        if str(detections).strip().lower() != "none":
            logger.warning("Detections found!")
            logger.warning(data["detections"])
            return True
        else:
            return False

    else:
        logger.info(f"GUARDRAILS ENABLED: {guardrails_enabled}")
        return False
