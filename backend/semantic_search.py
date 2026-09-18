import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ==================================================
# 1. Load chunks
# ==================================================

CHUNKS_FILE = Path("data/chunks/chunks.json")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")


# ==================================================
# 2. Load embedding model
# ==================================================

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


# ==================================================
# 3. Create cosine-similarity FAISS index
# ==================================================

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

# Convert embeddings to float32
embeddings = embeddings.astype("float32")

# Normalize embeddings
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

# Inner Product on normalized vectors = Cosine Similarity
index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"FAISS index created with {index.ntotal} vectors")


# ==================================================
# 4. Save index
# ==================================================

FAISS_FOLDER = Path("data/faiss")

FAISS_FOLDER.mkdir(parents=True, exist_ok=True)

FAISS_INDEX = FAISS_FOLDER / "index.faiss"

faiss.write_index(index, str(FAISS_INDEX))

print(f"Index saved to: {FAISS_INDEX}")


# ==================================================
# 5. Search
# ==================================================

query = input("\nEnter your question: ")

query_embedding = model.encode(
    [query],
    convert_to_numpy=True
).astype("float32")

# Normalize query vector
faiss.normalize_L2(query_embedding)

top_k = 5

similarities, indices = index.search(
    query_embedding,
    top_k
)


# ==================================================
# 6. Display results
# ==================================================

print("\n" + "=" * 70)
print("SEMANTIC SEARCH RESULTS - COSINE SIMILARITY")
print("=" * 70)

for rank, (idx, similarity) in enumerate(
    zip(indices[0], similarities[0]),
    start=1
):

    chunk = chunks[idx]

    print(f"\nRESULT {rank}")
    print("-" * 70)

    print(f"Document: {chunk['document']}")
    print(f"Page: {chunk['page']}")
    print(f"Cosine Similarity: {similarity:.4f}")

    print("\nText:")
    print(chunk["text"])