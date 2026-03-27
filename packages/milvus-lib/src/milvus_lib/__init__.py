import itertools
import logging
from typing import Any, Dict, Iterable, List

import requests
from pydantic import BaseModel

# PyMilvus imports
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    connections,
    utility,
)
from pymilvus.client.types import LoadState

logger = logging.getLogger(f"uvicorn.{__name__}")


class SearchComponent(BaseModel):
    """A design system component as returned from search results.

    Represents the core metadata of a component stored in the knowledge base,
    excluding the relevance score. Used as the base for search result models.
    """

    title: str
    url: str
    description: str
    parent: str
    accessibility: str
    status: str
    created_at: str
    updated_at: str
    has_research: bool
    needs_research: bool
    views: int


class ScoredSearchComponent(SearchComponent):
    """A search result combining component metadata with its relevance score.

    Extends SearchComponent with the similarity score returned by vector search,
    indicating how closely the component matches the query.
    """

    score: float


class ComponentEntry(BaseModel):
    """A component ready for ingestion into the Milvus knowledge base.

    Contains all metadata and content fields required for indexing. The views
    field is excluded when upserting to allow Milvus to manage counters.
    """

    component_id: str
    title: str
    description: str
    url: str
    parent: str
    accessibility: str
    status: str
    has_research: bool
    needs_research: bool
    created_at: str
    updated_at: str
    views: int
    content: str
    full_content: str

    def validate_url(self):
        """Check if the component url is valid."""
        response = requests.head(self.url)
        if response.status_code != 200:
            logger.error(
                f"Component URL not found: {response.status_code} - {self.url}"
            )

    def upsert_dump(self) -> Dict[str, Any]:
        """Serialize the component for Milvus upsert, excluding the views field.

        Returns:
            A dictionary suitable for Collection.upsert(), with views omitted
            so Milvus can manage view counts separately.
        """
        self.validate_url()
        return self.model_dump(exclude={"views"})


class MilvusKnowledgeBase:
    """A Milvus-backed vector knowledge base for design system components.

    Manages connection, schema, indexing, and semantic search over component
    documentation. Uses OpenAI-compatible embeddings via the configured model.
    """

    host: str
    port: int
    collection_name: str
    embedding_model: str
    embedding_dim: int
    max_batch_size: int
    collection: Collection

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "knowledge_base",
        embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
        embedding_dim: int = 1024,
        max_batch_size: int = 2,
    ):
        """Initialize the knowledge base client.

        Args:
            host: Milvus server host.
            port: Milvus server port.
            collection_name: Name of the vector collection.
            embedding_model: Model identifier for text embeddings.
            embedding_dim: Dimension of embedding vectors.
            max_batch_size: Maximum components per upsert batch.
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.max_batch_size = max_batch_size

    def __schema(self) -> CollectionSchema:
        """Build the collection schema with fields and embedding function."""
        return CollectionSchema(
            fields=self.__fields(),
            functions=self.__functions(),
            description="Knowledge base for design system component",
        )

    def __index(self) -> Dict[str, Any]:
        """Build the IVF_FLAT index parameters for cosine similarity."""
        return {
            "metric_type": "COSINE",  # Use cosine similarity
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }

    def __fields(self) -> List[FieldSchema]:
        """Define the schema fields for the collection."""
        return [
            FieldSchema(
                name="component_id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=256,
                description="Unique component identifier, Primary key",
            ),
            FieldSchema(
                name="title",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="Component title",
            ),
            FieldSchema(
                name="description",
                dtype=DataType.VARCHAR,
                max_length=4096,
                description="Component description",
            ),
            FieldSchema(
                name="url",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="Component URL",
            ),
            FieldSchema(
                name="parent",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="Parent design system",
            ),
            FieldSchema(
                name="status",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="Component status",
            ),
            FieldSchema(
                name="accessibility",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="Accessibility level (e.g., AA)",
            ),
            FieldSchema(
                name="has_research",
                dtype=DataType.BOOL,
                default_value=False,
                description="Whether component has research",
            ),
            FieldSchema(
                name="needs_research",
                dtype=DataType.BOOL,
                default_value=False,
                description="Whether component needs research",
            ),
            FieldSchema(
                name="created_at",
                dtype=DataType.VARCHAR,
                max_length=128,
                description="Creation timestamp",
            ),
            FieldSchema(
                name="updated_at",
                dtype=DataType.VARCHAR,
                max_length=128,
                description="Update timestamp",
            ),
            FieldSchema(
                name="views", dtype=DataType.INT64, description="views", default_value=0
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="Markdown content to be embedded",
            ),
            FieldSchema(
                name="content_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dim,
                description="Vector embedding of component content",
            ),
            FieldSchema(
                name="full_content",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="Full markdown content (truncated if needed)",
            ),
        ]

    def __functions(self) -> List[Function]:
        """Define the embedding function for content to vector conversion."""
        params = {  # Provider-specific configuration (highest priority)
            "provider": "openai",  # Embedding model provider
            "model_name": self.embedding_model,  # Embedding model
            # "dim": self.embedding_dim,
            # Optional parameters:
            # "credential": "apikey_dev",               # Optional: Credential label specified in milvus.yaml
            # "user": "user123"                         # Optional: identifier for API tracking
        }
        return [
            Function(
                name="content_embedding",  # Unique identifier for this embedding function
                function_type=FunctionType.TEXTEMBEDDING,  # Type of embedding function
                input_field_names=["content"],  # Scalar field to embed
                output_field_names=[
                    "content_embedding"
                ],  # Vector field to store embeddings
                params=params,
            ),
        ]

    def is_healthy(self) -> bool:
        """Check if the collection is loaded and has a valid embedding index.

        Returns:
            True if the collection is ready for search, False otherwise.
            Attempts to reload the collection if the index check fails.
        """
        utility.has_collection
        if self.collection is None:
            return False
        return utility.has_collection(self.collection_name)

    def connect(self, drop_existing: bool = False):
        """Connect to Milvus and load or create the collection.

        Args:
            drop_existing: If True, drop the existing collection before
                creating a new one. Use with caution as it erases all data.
        """
        connections.connect(
            alias="default", host=self.host, port=self.port, keep_alive=True
        )

        if drop_existing and utility.has_collection(self.collection_name):
            logger.warning(f"Dropping existing collection: {self.collection_name}")
            utility.drop_collection(self.collection_name)

        # Get collection and load it
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
        else:
            logger.info(f"Creating new collection: {self.collection_name}")
            self.collection = Collection(self.collection_name, schema=self.__schema())
            self.collection.create_index(
                field_name="content_embedding", index_params=self.__index()
            )
        self.collection.load()
        logger.info(f"Connected to Milvus collection: {self.collection_name}")

    def close(self):
        """Release the collection and disconnect from Milvus."""
        self.collection.release()
        self.collection = None
        connections.disconnect("default")
        logger.info(f"Disconnected from Milvus collection: {self.collection_name}")

    def add_components(self, components: Iterable[ComponentEntry]):
        """Insert or update components in the knowledge base.

        Components are batched according to max_batch_size and upserted.
        Embeddings are computed automatically by Milvus.

        Args:
            components: Iterable of ComponentEntry instances to add.
        """
        for batch in itertools.batched(
            map(lambda x: x.upsert_dump(), components), self.max_batch_size
        ):
            self.collection.upsert([*batch])
        self.collection.flush()

    def search_components(
        self,
        query: List[float] | str,
        limit: int = 10,
        min_count: int = 3,
        min_score: float = 0.2,
    ) -> List[ScoredSearchComponent]:
        """Perform semantic search over component content.

        Args:
            query: Search query as a text string (embedding computed automatically)
                or a pre-computed embedding vector.
            limit: Maximum number of results to return.

        Returns:
            List of ScoredSearchComponent results ordered by relevance.
        """
        # Search parameters
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        if utility.load_state(self.collection_name) == LoadState.NotLoad:
            self.collection.load()

        # Perform search
        results = self.collection.search(
            data=[query],
            anns_field="content_embedding",
            param=search_params,
            limit=limit,
            output_fields=[
                "title",
                "description",
                "url",
                "parent",
                "accessibility",
                "status",
                "has_research",
                "needs_research",
                "created_at",
                "updated_at",
                "views",
            ],
        )

        count = 0

        # Format results
        formatted_results = []
        filtered_results = []
        for hits in results:
            for hit in hits:
                result = ScoredSearchComponent(
                    score=hit.score,
                    title=hit.entity.get("title"),
                    description=hit.entity.get("description"),
                    url=hit.entity.get("url"),
                    parent=hit.entity.get("parent"),
                    status=hit.entity.get("status"),
                    accessibility=hit.entity.get("accessibility"),
                    has_research=hit.entity.get("has_research"),
                    needs_research=hit.entity.get("needs_research"),
                    created_at=hit.entity.get("created_at"),
                    updated_at=hit.entity.get("updated_at"),
                    views=hit.entity.get("views"),
                )
                if count >= min_count and hit.score < min_score:
                    filtered_results.append(result)
                else:
                    count += 1
                    formatted_results.append(result)
        logger.info(f"Filtered results: {filtered_results}")
        return formatted_results
