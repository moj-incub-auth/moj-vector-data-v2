# llm-lib

LLM integration library for extracting structured component data from design system documentation.

## Features

- **LLM-powered extraction**: Uses qwen3-14b-llm to extract structured data from component files
- **Research detection**: Automatically detects research sections with specific criteria
- **Pydantic models**: Type-safe data structures for component information
- **Flexible input**: Extract from files or content strings
- **JSON output**: Returns structured JSON matching the required format

## Packages (libraries)

| Package       | Description                                 |
|---------------|---------------------------------------------|
| `milvus-lib`  | Milvus client, schema, and search utilities |
| `llm-lib`     | Component extraction with LLM Prompts       |

## Apps

| App           | Description                                  |
|---------------|----------------------------------------------|
| `search-app`  | FastAPI vector search API                    |
| `ingest-app`  | CLI for ingesting components into Milvus     |

## Setup

### Env Variables

| Package            | Mandatory | Description                                           |
|--------------------|-----------|-------------------------------------------------------|
| `INGEST_DIR`       | YES       | Directory where Design Systems will be cloned at      |
| `OPENAI_BASE_URL`  | YES       | URL for chatCompletions eg. http://127.0.0.1:8090/v1" |
| `INFERENCE_MODEL`  | NO        | Name of model (defaults to qwen3-14b-llm)             |
| `INFERENCE_API_KEY`| NO        | API KEY (defaults to not-needed)                      |
| `INGEST_DIR`       | YES       | Directory where Design Systems will be cloned at      |

## Dependencies
```bash
uv sync --all-packages --all-extras --all-groups
```

## Usage

### ingest-app-ai

CLI with two subcommands. Requires `uv sync` from the project root.

```bash
# Ingest moj-frontend components (expects ingest/moj-frontend/)
uv run ingest-app ingest-ai --ingest-dir ingest --llm-base-url "http://127.0.0.1:8080/v1"

# Ingest and drop existing collection
uv run ingest-app ingest-ai --ingest-dir ingest --drop --llm-base-url "http://127.0.0.1:8080/v1"

```

Options (e.g. `--host`, `--port`, `--embedding-model`) are available for both subcommands. See `uv run ingest-app --help`.


