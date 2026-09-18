import json
import re
import os
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv


# ==================================================
# 1. Load environment variables
# ==================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Please check your .env file."
    )

client = genai.Client(api_key=API_KEY)


# ==================================================
# 2. Load document chunks
# ==================================================

CHUNKS_FILE = Path("data/chunks/chunks.json")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")


# ==================================================
# 3. BM25 Keyword Search Setup
# ==================================================

def tokenize(text):
    """
    Convert text into lowercase words.
    Example:
    'Minimum Attendance Requirement'
    becomes:
    ['minimum', 'attendance', 'requirement']
    """
    return re.findall(r"\b\w+\b", text.lower())


tokenized_corpus = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_corpus)


# ==================================================
# 4. Semantic Search Setup
# ==================================================

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

FAISS_INDEX = Path("data/faiss/index.faiss")

index = faiss.read_index(str(FAISS_INDEX))


# ==================================================
# 5. Hybrid Search Function
# ==================================================

def hybrid_search(query, top_k=5):

    # ------------------------------------------------
    # BM25 Search
    # ------------------------------------------------

    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_results = sorted(
        enumerate(bm25_scores),
        key=lambda x: x[1],
        reverse=True
    )[:10]


    # ------------------------------------------------
    # Semantic Search using Cosine Similarity
    # ------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Normalize query vector
    faiss.normalize_L2(query_embedding)

    cosine_scores, semantic_indices = index.search(
        query_embedding,
        10
    )


    # ------------------------------------------------
    # Normalize BM25 scores
    # ------------------------------------------------

    bm25_max = max(
        score for _, score in bm25_results
    )

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


    # ------------------------------------------------
    # Normalize Semantic Scores
    # ------------------------------------------------

    semantic_scores = cosine_scores[0]

    semantic_min = min(semantic_scores)
    semantic_max = max(semantic_scores)

    semantic_normalized = {}

    for rank, idx in enumerate(
        semantic_indices[0]
    ):

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


    # ------------------------------------------------
    # Combine BM25 + Semantic Scores
    # ------------------------------------------------

    hybrid_scores = {}

    all_indices = set(
        bm25_normalized.keys()
    ).union(
        semantic_normalized.keys()
    )


    for idx in all_indices:

        keyword_score = bm25_normalized.get(
            idx,
            0
        )

        semantic_score = semantic_normalized.get(
            idx,
            0
        )

        # 50% BM25 + 50% Semantic
        hybrid_score = (
            0.5 * keyword_score
            +
            0.5 * semantic_score
        )

        hybrid_scores[idx] = hybrid_score


    # ------------------------------------------------
    # Select Top Results
    # ------------------------------------------------

    hybrid_results = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]


    # Return chunks only
    return [
        chunks[idx]
        for idx, score in hybrid_results
    ]


# ==================================================
# 6. Generate Answer Using Gemini
# ==================================================

def generate_answer(
    query,
    retrieved_chunks
):

    context_parts = []


    # ------------------------------------------------
    # Build context from retrieved chunks
    # ------------------------------------------------

    for chunk in retrieved_chunks:

        context_parts.append(
            f"""
Document: {chunk['document']}
Page: {chunk['page']}

Content:
{chunk['text']}
"""
        )


    context = "\n".join(context_parts)


    # ------------------------------------------------
    # RAG Prompt
    # ------------------------------------------------

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

IMPORTANT RULES:

1. Do not make up information.

2. If the answer cannot be found in the provided context,
say:

"I could not find this information in the provided documents."

3. Treat each document as a separate source.

4. Do NOT combine rules, policies, percentages, dates,
or requirements from different documents as if they belong
to the same institution.

5. If different documents give different answers, clearly
mention the difference and identify the document associated
with each answer.

6. Always mention the document name when the answer depends
on a specific document.

7. Do not assume that two documents belong to the same
institution.

8. Give a clear and concise answer.

9. Do not use outside knowledge. Use only the retrieved
document context.

USER QUESTION:
{query}

RETRIEVED DOCUMENT CONTEXT:
{context}

ANSWER:
"""


    # ------------------------------------------------
    # Send prompt to Gemini
    # ------------------------------------------------

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )


    return response.text


# ==================================================
# 7. Main Program
# ==================================================

def main():

    # ------------------------------------------------
    # Get question
    # ------------------------------------------------

    query = input(
        "\nEnter your question: "
    )


    # ------------------------------------------------
    # Hybrid retrieval
    # ------------------------------------------------

    print("\nSearching documents...")

    retrieved_chunks = hybrid_search(
        query,
        top_k=5
    )


    # ------------------------------------------------
    # Generate answer
    # ------------------------------------------------

    print("Generating answer...")

    answer = generate_answer(
        query,
        retrieved_chunks
    )


    # =================================================
    # Display Answer
    # =================================================

    print("\n" + "=" * 70)
    print("RAG ANSWER")
    print("=" * 70)

    print("\n" + answer)


    # =================================================
    # Display Sources
    # =================================================

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)


    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        print(
            f"{i}. {chunk['document']} "
            f"(Page {chunk['page']})"
        )


# ==================================================
# 8. Start Program
# ==================================================

if __name__ == "__main__":
    main()