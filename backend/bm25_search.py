import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


# --------------------------------------------------
# 1. Load chunks
# --------------------------------------------------

CHUNKS_FILE = Path("data/chunks/chunks.json")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")


# --------------------------------------------------
# 2. Tokenization function
# --------------------------------------------------

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# --------------------------------------------------
# 3. Prepare documents for BM25
# --------------------------------------------------

tokenized_corpus = []

for chunk in chunks:
    tokens = tokenize(chunk["text"])
    tokenized_corpus.append(tokens)


# --------------------------------------------------
# 4. Create BM25 model
# --------------------------------------------------

bm25 = BM25Okapi(tokenized_corpus)


# --------------------------------------------------
# 5. Get user's question
# --------------------------------------------------

query = input("\nEnter your question: ")

query_tokens = tokenize(query)


# --------------------------------------------------
# 6. Calculate BM25 scores
# --------------------------------------------------

scores = bm25.get_scores(query_tokens)


# --------------------------------------------------
# 7. Get top 5 results
# --------------------------------------------------

top_k = 5

top_results = sorted(
    enumerate(scores),
    key=lambda x: x[1],
    reverse=True
)[:top_k]


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\n" + "=" * 70)
print("BM25 KEYWORD SEARCH RESULTS")
print("=" * 70)


for rank, (index, score) in enumerate(top_results, start=1):

    chunk = chunks[index]

    print(f"\nRESULT {rank}")
    print("-" * 70)

    print(f"Document: {chunk['document']}")
    print(f"Page: {chunk['page']}")
    print(f"BM25 Score: {score}")

    print("\nText:")
    print(chunk["text"])