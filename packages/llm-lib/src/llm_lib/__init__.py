# Standard library imports
from dataclasses import dataclass
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# Third party imports
from openai import OpenAI
from pydantic import BaseModel, Field

from milvus_lib import ComponentEntry

logger = logging.getLogger(__name__)


class LLMComponentDataResponse(BaseModel):
    """Component data model matching the required JSON format."""

    title: str
    url: str
    description: str
    parent: str
    accessibility: str
    created_at: str
    updated_at: str
    has_research: bool
    needs_research: bool
    views: int = 0
    status: str

    class Config:
        populate_by_name = True


class LLMIngestionAssistantBase:
    """An LLM-based ingestion system

    Manages connection, schema, indexing, and semantic search over component
    documentation. Uses OpenAI-compatible embeddings via the configured model.
    """
    base_url: str  #http://127.0.0.1:8090/v1
    inference_model: str
    inference_api_key: int
    context_size: int
    client: OpenAI


    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8090/v1",
        inference_model = "none",
        inference_api_key = "not-needed",
        context_size = 65000,
    ):
        """Initialize the llm base client.

        Args:
            base_url: OPEAN AI Base URL eg. http://host:port/v1
            collection_name: Name of the vector collection.
            inference_model: Model identifier inference.
            inference_api_key: Authentication Key.
            context_size: Maximum context.
        """
        self.base_url = base_url
        self.inference_model = inference_model
        self.inference_api_key = inference_api_key
        self.context_size = context_size

        self.client = OpenAI(
            base_url=base_url
            or os.getenv(
                "OPENAI_BASE_URL",
                "http://127.0.0.1:8090/v1", #oc port-forward qwen3-14b-llm-predictor-67c95fbd5-r4vs5 8090:8080
            ),            
            api_key=inference_api_key or os.getenv("OPENAI_API_KEY", "not-needed"),
        )

@dataclass
class LLMComponentEntry:
    """Dataclass for LLM Design System component entries."""

    component_path: Path
    llm_structured_output: LLMComponentDataResponse
    full_content: str


    def to_component_entry(self) -> ComponentEntry:
        """Convert LLMComponentDataResponse to ComponentEntry for Milvus storage."""
        title = self.llm_structured_output.title
        description = self.llm_structured_output.description
        status = "N/A"
        created_at= self.llm_structured_output.created_at
        updated_at = self.llm_structured_output.updated_at
        has_research = self.llm_structured_output.has_research
        needs_research = self.llm_structured_output.needs_research
        accessibility = self.llm_structured_output.accessibility
        parent = self.llm_structured_output.parent
        url = self.llm_structured_output.url
        status = self.llm_structured_output.status


        content = f"""
Title: {title}
Description: {description}
Parent: {parent}
Content: {self.full_content}
        """[:65000].strip()

        logger.debug(f"TRANSFORMED FOR MILVUS COMPONENT [{title}]")


        return ComponentEntry(
            component_id=url,
            title=title,
            description=description,
            url=url,
            parent=parent,
            status=status,                  #needs to be aligned to Adam's
            accessibility=accessibility,
            has_research=has_research,      #needs to be aligned to Adam's
            needs_research=needs_research,           #needs to be added and aligned to Adam's
            created_at=created_at,
            updated_at=updated_at,
            views=0,
            content=content,
            full_content=self.full_content,

        )


__all__ = ["LLMComponentDataResponse", "LLMIngestionAssistantBase"]
