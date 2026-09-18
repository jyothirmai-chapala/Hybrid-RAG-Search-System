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
# 2. Tokenizer
# ==================================================

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# ==================================================
# 3. BM25 Setup
# ==================================================

tokenized_corpus = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_corpus)


# ==================================================
# 4. Semantic Search Setup
# ==================================================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

index = faiss.read_index(
    "data/faiss/index.faiss"
)


# ==================================================
# 5. Evaluation Dataset
# ==================================================
#
# Relevance is defined using DOCUMENT + PAGE.
#
# This means a retrieved chunk is considered relevant
# only when it comes from an expected document AND page.
#

evaluation_questions = [

    {
        "query": "What is the minimum attendance requirement?",
        "relevant_pages": [
            (
                "IAR-Attendance-Policy.pdf",
                2
            ),
            (
                "UG_Regulations_(2024-25)_2024-7-17-11-36-22.pdf",
                5
            )
        ]
    },

    {
        "query": "What happens if a student has shortage of attendance?",
        "relevant_pages": [
            (
                "IAR-Attendance-Policy.pdf",
                3
            )
        ]
    },

    {
        "query": "What are the examination rules for students?",
        "relevant_pages": [
            (
                "Examination_Rules_for_Students.pdf",
                1
            ),
            (
                "Examination_Rules_for_Students.pdf",
                2
            )
        ]
    },

    {
        "query": "What are the regulations for B.Tech students?",
        "relevant_pages": [
            (
                "R26_Regulations_B.Tech.pdf",
                1
            ),
            (
                "R26_Regulations_B.Tech.pdf",
                2
            ),
            (
                "R26_Regulations_B.Tech.pdf",
                3
            )
        ]
    },

    {
        "query": "What is the curriculum and instruction for B.Ed?",
        "relevant_pages": [
            (
                "__UG_B.Ed._Education_70122-Curriculum and Instruction_1747.pdf",
                1
            )
        ]
    }
]


# ==================================================
# 6. BM25 Search
# ==================================================

def bm25_search(query, top_k=5):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(
        query_tokens
    )

    results = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    return [
        idx
        for idx, score in results
    ]


# ==================================================
# 7. Semantic Search
# ==================================================

def semantic_search(query, top_k=5):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(
        query_embedding
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    return indices[0].tolist()


# ==================================================
# 8. Hybrid Search
# ==================================================

def hybrid_search(query, top_k=5):

    # ------------------------------------------------
    # BM25 top 10
    # ------------------------------------------------

    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    bm25_results = sorted(
        enumerate(bm25_scores),
        key=lambda x: x[1],
        reverse=True
    )[:10]


    # ------------------------------------------------
    # Semantic top 10
    # ------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(
        query_embedding
    )

    semantic_scores, semantic_indices = index.search(
        query_embedding,
        10
    )


    # ------------------------------------------------
    # Normalize BM25
    # ------------------------------------------------

    bm25_max = max(
        score
        for _, score in bm25_results
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
    # Normalize Semantic
    # ------------------------------------------------

    semantic_scores = semantic_scores[0]

    semantic_min = min(
        semantic_scores
    )

    semantic_max = max(
        semantic_scores
    )

    semantic_normalized = {}

    for rank, idx in enumerate(
        semantic_indices[0]
    ):

        score = semantic_scores[rank]

        if semantic_max != semantic_min:

            normalized = (
                score - semantic_min
            ) / (
                semantic_max - semantic_min
            )

        else:

            normalized = 1.0

        semantic_normalized[idx] = normalized


    # ------------------------------------------------
    # Combine
    # ------------------------------------------------

    all_indices = set(
        bm25_normalized.keys()
    ).union(
        semantic_normalized.keys()
    )

    hybrid_scores = {}

    for idx in all_indices:

        keyword_score = bm25_normalized.get(
            idx,
            0
        )

        semantic_score = semantic_normalized.get(
            idx,
            0
        )

        hybrid_scores[idx] = (
            0.5 * keyword_score
            +
            0.5 * semantic_score
        )


    # ------------------------------------------------
    # Top K
    # ------------------------------------------------

    results = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    return [
        idx
        for idx, score in results
    ]


# ==================================================
# 9. Check Relevance
# ==================================================

def is_relevant(
    chunk,
    relevant_pages
):

    page_key = (
        chunk["document"],
        chunk["page"]
    )

    return page_key in relevant_pages


# ==================================================
# 10. Precision
# ==================================================

def calculate_precision(
    retrieved_indices,
    relevant_pages
):

    if not retrieved_indices:
        return 0.0

    relevant_retrieved = 0

    for idx in retrieved_indices:

        if is_relevant(
            chunks[idx],
            relevant_pages
        ):
            relevant_retrieved += 1

    return (
        relevant_retrieved
        /
        len(retrieved_indices)
    )


# ==================================================
# 11. Recall
# ==================================================

def calculate_recall(
    retrieved_indices,
    relevant_pages
):

    # Find all chunks in the relevant pages
    total_relevant_chunks = []

    for idx, chunk in enumerate(chunks):

        if is_relevant(
            chunk,
            relevant_pages
        ):
            total_relevant_chunks.append(idx)


    if not total_relevant_chunks:
        return 0.0


    retrieved_relevant = 0

    for idx in retrieved_indices:

        if idx in total_relevant_chunks:
            retrieved_relevant += 1


    return (
        retrieved_relevant
        /
        len(total_relevant_chunks)
    )


# ==================================================
# 12. Evaluate One Method
# ==================================================

def evaluate_method(
    method_name,
    search_function,
    query,
    relevant_pages
):

    retrieved = search_function(
        query,
        top_k=5
    )

    precision = calculate_precision(
        retrieved,
        relevant_pages
    )

    recall = calculate_recall(
        retrieved,
        relevant_pages
    )

    return precision, recall, retrieved


# ==================================================
# 13. Main Evaluation
# ==================================================

def main():

    print("\n")
    print("=" * 70)
    print("HYBRID RAG PRECISION / RECALL EVALUATION")
    print("=" * 70)


    methods = {
        "BM25": bm25_search,
        "Semantic": semantic_search,
        "Hybrid": hybrid_search
    }


    results = {
        method: {
            "precision": [],
            "recall": []
        }
        for method in methods
    }


    # ------------------------------------------------
    # Evaluate each question
    # ------------------------------------------------

    for question_number, item in enumerate(
        evaluation_questions,
        start=1
    ):

        query = item["query"]

        relevant_pages = set(
            item["relevant_pages"]
        )


        print("\n" + "-" * 70)

        print(
            f"Question {question_number}: "
            f"{query}"
        )

        print("-" * 70)


        for method_name, search_function in methods.items():

            precision, recall, retrieved = evaluate_method(
                method_name,
                search_function,
                query,
                relevant_pages
            )


            results[method_name]["precision"].append(
                precision
            )

            results[method_name]["recall"].append(
                recall
            )


            print(
                f"{method_name:<10} -> "
                f"Precision: {precision:.2f}, "
                f"Recall: {recall:.2f}"
            )


    # =================================================
    # Average Results
    # =================================================

    print("\n")
    print("=" * 70)
    print("AVERAGE RESULTS")
    print("=" * 70)


    for method_name in methods:

        avg_precision = sum(
            results[method_name]["precision"]
        ) / len(
            results[method_name]["precision"]
        )


        avg_recall = sum(
            results[method_name]["recall"]
        ) / len(
            results[method_name]["recall"]
        )


        print(
            f"\n{method_name}"
        )

        print(
            f"Average Precision: "
            f"{avg_precision:.2f}"
        )

        print(
            f"Average Recall: "
            f"{avg_recall:.2f}"
        )


# ==================================================
# 14. Run
# ==================================================

if __name__ == "__main__":
    main()