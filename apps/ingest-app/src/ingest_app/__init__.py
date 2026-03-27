import logging
from pathlib import Path

import configargparse
from ingest_lib.dwp_designs import DWPComponentsIngestor
from ingest_lib.govuk_design import GovUkComponentsIngestor
from ingest_lib.hmrc_designs import HMRCComponentsIngestor
from ingest_lib.moj_frontend import MojFrontendIngestor
from milvus_lib import MilvusKnowledgeBase
from llm_lib import LLMIngestionAssistantBase
from llm_lib.nhs_designs import NHSComponentsIngestorAI
from llm_lib.dwp_designs import DWPComponentsIngestorAI
from llm_lib.moj_designs import MOJComponentsIngestorAI

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def main():
    """CLI entry point for ingesting data into Milvus or searching the knowledge base.

    Supports two subcommands:
        ingest - Ingest design system components from moj-frontend into Milvus
        search - Run a semantic search query and print results
    """
    main_parser = configargparse.ArgParser(
        description="Interact with the Milvus Knowledge Base"
    )

    parent_parser = configargparse.ArgParser(add_help=False)

    parent_parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        env_var="MILVUS_HOST",
        help="Milvus server host",
    )

    parent_parser.add_argument(
        "--port",
        type=int,
        default=19530,
        env_var="MILVUS_PORT",
        help="Milvus server port",
    )

    parent_parser.add_argument(
        "--embedding-dim",
        type=int,
        default=1024,
        env_var="MILVUS_EMBEDDING_DIM",
        help="Milvus Embedding Dimension",
    )

    parent_parser.add_argument(
        "--max-batch-size",
        type=int,
        default=2,
        env_var="MILVUS_MAX_BATCH_SIZE",
        help="Milvus Max Batch Size",
    )

    parent_parser.add_argument(
        "--collection-name",
        type=str,
        default="knowledge_base",
        env_var="MILVUS_COLLECTION_NAME",
        help="Milvus collection name",
    )

    parent_parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-ai/nomic-embed-text-v1.5",
        env_var="MILVUS_EMBEDDING_MODEL",
        help="Milvus Embedding Model",
    )

    subparsers = main_parser.add_subparsers(dest="command", help="Command to execute")

    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest data into Milvus", parents=[parent_parser]
    )

    ingest_parser.add_argument(
        "--ingest-dir",
        type=Path,
        default=Path("ingest"),
        env_var="INGEST_DIR",
        help="Ingest directory",
    )

    ingest_parser.add_argument(
        "--drop",
        action="store_true",
        default=False,
        env_var="DROP_COLLECTIONS",
        help="Drop existing collections",
    )

    ingest_ai_parser = subparsers.add_parser(
        "ingest-ai", help="Ingest data into Milvus with AI Prompt", parents=[parent_parser]
    )


    ingest_ai_parser.add_argument(
        "--ingest-dir",
        type=Path,
        default=Path("ingest"),
        env_var="INGEST_DIR",
        help="Ingest directory",
    )

    ingest_ai_parser.add_argument(
        "--drop",
        action="store_true",
        default=False,
        env_var="DROP_COLLECTIONS",
        help="Drop existing collections",
    )

    ingest_ai_parser.add_argument(
        "--llm-base-url",
        type=str,
        default="http://127.0.0.1:8080/v1",
        env_var="OPENAI_BASE_URL",
        help="The LLM Base URL until /v1",
    )

    ingest_ai_parser.add_argument(
        "--inference-model",
        type=str,
        default="qwen3-14b-llm",
        env_var="INFERENCE_MODEL",
        help="The LLM model for inference",
    )

    ingest_ai_parser.add_argument(
        "--inference-api-key",
        type=str,
        default="not-needed",
        env_var="INFERENCE_API_KEY",
        help="The api key to authenticate with the LLM model for inference",
    )    


    search_parser = subparsers.add_parser(
        "search", help="Search Milvus", parents=[parent_parser]
    )

    search_parser.add_argument(
        "--search-query",
        type=str,
        required=True,
        help="Search query",
    )

    args = main_parser.parse_args()

    if args.command == "ingest":
        print(args.host)
        ingest_dir = args.ingest_dir
        if not ingest_dir.exists() or not ingest_dir.is_dir():
            raise FileNotFoundError(f"Ingest directory not found: {ingest_dir}")
        print(ingest_dir)

        milvus_client = MilvusKnowledgeBase(
            args.host,
            args.port,
            args.collection_name,
            args.embedding_model,
            args.embedding_dim,
            args.max_batch_size,
        )
        milvus_client.connect(drop_existing=args.drop)         

        for component_ingest in [
            MojFrontendIngestor(ingest_dir / "moj-frontend"),
            DWPComponentsIngestor(ingest_dir / "dwp-design-system"),
            GovUkComponentsIngestor(ingest_dir / "govuk-design-system"),
            HMRCComponentsIngestor(ingest_dir / "hmrc-design-system"),
        ]:
            if not component_ingest.project_exists():
                logger.warning(f"Project not found: {component_ingest.project_root}")
                continue
            component_count = component_ingest.component_count()
            if component_count > 0:
                logger.info(
                    f"Processing {component_ingest.component_count()} components in {component_ingest.project_root}"
                )
                milvus_client.add_components(component_ingest.extract_components())
                logger.info(
                    f"Added {component_count} components to Milvus collection: {args.collection_name}"
                )

        milvus_client.close()

    elif args.command == "ingest-ai":
        print(args.host)
        ingest_dir = args.ingest_dir
        if not ingest_dir.exists() or not ingest_dir.is_dir():
            raise FileNotFoundError(f"Ingest directory not found: {ingest_dir}")
        print(ingest_dir)

        milvus_client = MilvusKnowledgeBase(
            args.host,
            args.port,
            args.collection_name,
            args.embedding_model,
            args.embedding_dim,
            args.max_batch_size,
        )
        milvus_client.connect(drop_existing=args.drop)

        print(f"LLM URL:     {args.llm_base_url}")
        print(f"LLM Model:   {args.inference_model}")
        print(f"LLM API KEY: {args.inference_api_key}")

        llm_assistant = LLMIngestionAssistantBase(
            args.llm_base_url,
            args.inference_model,
            args.inference_api_key,
        )  

        for ai_component_ingest in [
            NHSComponentsIngestorAI(llm_assistant, ingest_dir / "nhsuk-service-manual"),
            # DWPComponentsIngestorAI(llm_assistant, ingest_dir / "design-system"),
            # MOJComponentsIngestorAI(llm_assistant, ingest_dir / "moj-frontend"),            
        ]:

            if not ai_component_ingest.project_exists():
                logger.warning(f"Project not found: {ai_component_ingest.project_root}")
                continue
            component_count = ai_component_ingest.component_count()
            print(f"ingest-ai - Processing # components [{component_count}]")
            if component_count > 0:
                logger.info(
                    f"Processing {ai_component_ingest.component_count()} components in {ai_component_ingest.project_root}"
                )
                milvus_client.add_components(ai_component_ingest.extract_components())
                logger.info(
                    f"Added {component_count} components to Milvus collection: {args.collection_name}"
                )        


        milvus_client.close()

    elif args.command == "search":
        milvus_client = MilvusKnowledgeBase(
            args.host,
            args.port,
            args.collection_name,
            args.embedding_model,
            args.embedding_dim,
            args.max_batch_size,
        )
        milvus_client.connect()
        search_results = milvus_client.search_components(args.search_query)
        for idx, result in enumerate(search_results):
            print(f"Result #{idx} (Similarity: {result.score:.4f})")
            print(f"  Title: {result.title}")
            print(f"  Description: {result.description[:100]}...")
            print(f"  URL: {result.url}")
            print(f"  Parent: {result.parent}")
            print(f"  Status: {result.status}")
            print(f"  Accessibility: {result.accessibility}")
            print(f"  Has Research: {result.has_research}")
            print(f"  Created At: {result.created_at}")
            print(f"  Updated At: {result.updated_at}")
            print(f"  Views: {result.views:.4f}")
        milvus_client.close()


if __name__ == "__main__":
    main()
