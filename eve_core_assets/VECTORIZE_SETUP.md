# Vectorize Setup for Eve

This Worker now supports Vectorize endpoints for inserting and querying embeddings. Follow these steps to create an index and test vector search.

## 1) Create Vector Index (Wrangler)

Pick a name (we use `eve-knowledge`), and choose dimensions to match your embeddings model. For Workers AI `@cf/baai/bge-base-en-v1.5`, use 768 dims and cosine metric:

```bash
wrangler vectorize index create eve-knowledge --dimensions 768 --metric cosine
```

Alternatively, if your Wrangler supports presets (varies by version):

```bash
wrangler vectorize index create eve-knowledge --preset bge-base-en-v1.5
```

Verify index:

```bash
wrangler vectorize index list
```

## 2) Bind Vectorize and AI in Worker

`wrangler.toml` already includes:

```toml
[ai]
binding = "AI"

[[vectorize]]
binding = "VECTORIZE"
index_name = "eve-knowledge"
```

Deploy the Worker:

```bash
wrangler deploy
```

## 3) Insert Text into the Index

You can insert raw text; the Worker will create embeddings using Workers AI and upsert into Vectorize:

```bash
curl -s -X POST "https://d1-template.jeffgreen311.workers.dev/v/insert" \
  -H "Content-Type: application/json" \
  -d '{
        "id": "doc-1",
        "text": "Eve is an AI companion that creates music, dreams, and memories.",
        "metadata": {"source": "eve-docs", "lang": "en"}
      }'
```

If you already have embeddings, send them directly:

```bash
curl -s -X POST "https://d1-template.jeffgreen311.workers.dev/v/insert" \
  -H "Content-Type: application/json" \
  -d '{
        "id": "doc-2",
        "embedding": [0.01, 0.02, ...],
        "metadata": {"source": "precomputed"}
      }'
```

## 4) Query the Index

Query with natural language; the Worker embeds the text and performs vector search:

```bash
curl -s -X POST "https://d1-template.jeffgreen311.workers.dev/v/query" \
  -H "Content-Type: application/json" \
  -d '{
        "text": "What is Eve and how does she store sessions?",
        "topK": 5,
        "returnMetadata": true
      }'
```

You can also query with a precomputed embedding by passing `embedding` instead of `text`.

## 5) Next: Build RAG with AutoRAG

- Use Vectorize for retrieval.
- Generate answers with Workers AI LLMs or your preferred model.
- Optionally integrate [AutoRAG] to manage pipelines. When ready, bind your AutoRAG Worker similarly and call it from this Worker.

## Tips

- Ensure `wrangler.toml` uses the same `index_name` you created.
- If you set a preview URL, you can test against it by swapping the hostname in the curl commands.
- For large batch inserts, consider a custom `/v/batchInsert` endpoint or use Wrangler CLI bulk upload tools.
