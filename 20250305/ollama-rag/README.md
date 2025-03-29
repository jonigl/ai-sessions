# Trying to create a RAG system

## Mismatch in dimensionality

Sometimes an error of the kind

`chromadb.errors.InvalidDimensionException: Embedding dimension 384 does not match collection dimensionality 768`

both when adding documents or querying chroma. The thing is that the model and chromaDB don't seem to accept a forced size of vector. Instead, I've used a function to truncate or fill with 0s to the amount of dimensions in the embeds.


Then other error happened:
`ValueError: Unequal lengths for fields: ids: 1, embeddings: 24, documents: 24 in add.`

This happened because when there are several chunks, you have to associate them to different IDs. So I added a suffix to the ids, so those are unique.
