import json
import re
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ==================================================
# 1. Load chunks
# ==================================================

CHUNKS_FILE = Path("data/chunks/chunks.json")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")


# ==================================================
# 2. BM25 setup
# ==================================================

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


tokenized_corpus = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_corpus)


# ==================================================
# 3. Semantic search setup
# ==================================================

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

FAISS_INDEX = Path("data/faiss/index.faiss")

index = faiss.read_index(str(FAISS_INDEX))


# ==================================================
# 4. Get user question
# ==================================================

query = input("\nEnter your question: ")


# ==================================================
# 5. BM25 search
# ==================================================

query_tokens = tokenize(query)

bm25_scores = bm25.get_scores(query_tokens)

bm25_results = sorted(
    enumerate(bm25_scores),
    key=lambda x: x[1],
    reverse=True
)[:10]


# ==================================================
# 6. Semantic search
# ==================================================

query_embedding = model.encode(
    [query],
    convert_to_numpy=True
).astype("float32")

# Normalize query embedding
faiss.normalize_L2(query_embedding)

cosine_scores, semantic_indices = index.search(
    query_embedding,
    10
)


# ==================================================
# 7. Normalize BM25 scores
# ==================================================

bm25_max = max(score for _, score in bm25_results)

if bm25_max > 0:

    bm25_normalized = {
        idx: score / bm25_max
        for idx, score in bm25_results
    }

else:

    bm25_normalized = {
        idx: 0
        for idx, score in bm25_results
    }


# ==================================================
# 8. Normalize cosine similarity scores
# ==================================================

semantic_scores = cosine_scores[0]

semantic_min = min(semantic_scores)
semantic_max = max(semantic_scores)

semantic_normalized = {}

for rank, idx in enumerate(semantic_indices[0]):

    score = semantic_scores[rank]

    if semantic_max != semantic_min:

        normalized_score = (
            score - semantic_min
        ) / (
            semantic_max - semantic_min
        )

    else:

        normalized_score = 1.0

    semantic_normalized[idx] = normalized_score


# ==================================================
# 9. Combine BM25 + Semantic scores
# ==================================================

hybrid_scores = {}

all_indices = set(
    bm25_normalized.keys()
).union(
    semantic_normalized.keys()
)


for idx in all_indices:

    keyword_score = bm25_normalized.get(idx, 0)

    semantic_score = semantic_normalized.get(idx, 0)

    hybrid_score = (
        0.5 * keyword_score
        +
        0.5 * semantic_score
    )

    hybrid_scores[idx] = hybrid_score


# ==================================================
# 10. Sort Hybrid results
# ==================================================

hybrid_results = sorted(
    hybrid_scores.items(),
    key=lambda x: x[1],
    reverse=True
)[:5]


# ==================================================
# 11. Display results
# ==================================================

print("\n" + "=" * 70)
print("HYBRID SEARCH RESULTS")
print("=" * 70)

for rank, (idx, score) in enumerate(
    hybrid_results,
    start=1
):

    chunk = chunks[idx]

    print(f"\nRESULT {rank}")
    print("-" * 70)

    print(f"Document: {chunk['document']}")
    print(f"Page: {chunk['page']}")
    print(f"Hybrid Score: {score:.4f}")

    print("\nText:")
    print(chunk["text"])