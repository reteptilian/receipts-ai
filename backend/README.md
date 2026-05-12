# Receipts_ai Backend

This includes a set of utility scripts for ingesting, reviewing, uploading and outputting transaction data. It also includes library functionality that can be called from a UI to read and modify transaction data in firestore.

## Generating embeddings

There are features in this app that use embeddings vectors. The embeddings need to be pre-generated for the product taxonomy dataset. This can be done with the following command:

```
uv run python devtools/build_taxonomy_embeddings.py
```



