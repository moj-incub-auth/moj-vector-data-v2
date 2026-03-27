# MOJ Vector Data v2

Vector search and ingestion for the MOJ Design System. Uses QWEN3-14B to ingest via Prompt and Milvus for embedding storage and semantic search.

## Setup

```bash
uv sync --all-packages --all-extras --all-groups
```

## Packages (libraries)

| Package       | Description                                 |
|---------------|---------------------------------------------|
| `milvus-lib`  | Milvus client, schema, and search utilities |
| `ingest-lib`  | Component extraction from moj-frontend docs |

## Apps

| App           | Description                                  |
|---------------|----------------------------------------------|
| `search-app`  | FastAPI vector search API                    |
| `ingest-app`  | CLI for ingesting components into Milvus    |

## Usage

### ingest-app

CLI with two subcommands. Requires `uv sync` from the project root.

```bash
# Ingest moj-frontend components (expects ingest/moj-frontend/)
uv run ingest-app ingest --ingest-dir ingest

# Ingest and drop existing collection
uv run ingest-app ingest --ingest-dir ingest --drop

# Search from the command line
uv run ingest-app search --search-query "button component"
```

Options (e.g. `--host`, `--port`, `--embedding-model`) are available for both subcommands. See `uv run ingest-app --help`.

### search-app

FastAPI web server. Start with:

```bash
uv run fastapi run -e search_app:app --host 0.0.0.0 --port 8080
```

For development with auto-reload:

```bash
uv run fastapi dev -e search_app:app
```

Requires a running Milvus instance (see `MILVUS_HOST`, `MILVUS_PORT` env vars).


## GuardRails

| Env Var               | Description                                                        |
|-----------------------|-------------------------------------------------                   |
| `GUARDRAILS_ENABLED`  | Enable by setting to True (default False)                          |
| `GUARDRAILS_GATEWAY`  | The service of the Gateway eg. http://guardrails-gateway.vllm-serving.svc.cluster.local:8090  |
| `GUARDRAILS_TYPE`     | The type of guardrail for various guardrails behaviors eg. (`all`, `passthrough`, `hap`) 
| `GUARDRAILS_API_KEY`  | The API KEY                                                        |
| `INFERENCE_MODEL`     | Model we are hitting for inference qwen3-14b-llm                   |

