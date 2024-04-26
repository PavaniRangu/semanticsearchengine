**Part 1: Ingesting Documents**
1 . Read the given data.
2 . Apply appropriate cleaning steps on subtitle documents (whatever is required)
3 . Experiment with the following to generate text vectors of subtitle documents:
4 .  A very important step to improve the performance: Document Chunker.
5 . Store embeddings in a ChromaDB database. 

**Part 2: Retrieving Documents**
Take the user's search query.
Preprocess the query (if required).
Create query embedding.
Using cosine distance, calculate the similarity score between embeddings of documents and user search query embedding.
These cosine similarity scores will help in returning the most relevant candidate documents as per user’s search query.
