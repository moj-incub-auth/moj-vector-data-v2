## Prereqs

1. Setup ENV vars

```bash
export MILVUS_HOST=127.0.0.1
export MILVUS_PORT=19530
export MILVUS_EMBEDDING_DIM=768
export MILVUS_COLLECTION_NAME=knowledge_base
export MILVUS_EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5"
export INGEST_DIR=eg. /home/data

echo MILVUS_HOST=:            $MILVUS_HOST
echo MILVUS_PORT=:            $MILVUS_PORT
echo MILVUS_EMBEDDING_DIM:    $MILVUS_EMBEDDING_DIM
echo MILVUS_COLLECTION_NAME:  $MILVUS_COLLECTION_NAME
echo MILVUS_EMBEDDING_MODEL:  $MILVUS_EMBEDDING_MODEL
echo INGEST_DIR:              $INGEST_DIR
```

2. Install all requirements

```bash
cd cd moj-vector-data
uv sync --all-packages --all-extras --all-groups
```

3. Run locally all components

IMPORTANT: Read https://huggingface.co/docs/text-embeddings-inference/intel_container and update accordingly the following section in [docker-compose.yaml](../../deploy/docker/docker-compose.yaml)

```
  text-embeddings-inference:
    container_name: tei
    #image:
```

Execute:

```bash
docker-compose up
# Linux docker-compose -f docker-compose.cpu-ipex.yaml up/down
```

4. Verify `milvus` is running by connecting to [http://0.0.0.0:9091/webui](http://0.0.0.0:9091/webui)

## Use Ingest & Search Utility

1. Download/Clone the design systems repos under `INGEST_DIR`

```bash
cd $INGEST_DIR 
git clone https://github.com/ministryofjustice/moj-frontend
```

2. Ingest

```bash
cd moj-vector-data
uv run ingest-app ingest
```


3. Search

```bash
cd moj-vector-data
uv run ingest-app search --search-query "Can you give me a date picker?"
```
