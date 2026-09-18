import os
import json
import re
from pathlib import Path

import faiss
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Please check your .env file."
    )

client = genai.Client(api_key=API_KEY)


# ============================================================
# 2. FASTAPI APP
# ============================================================

app = FastAPI(
    title="Hybrid RAG Search System",
    description="BM25 + Semantic Search + FAISS + Gemini",
    version="1.0"
)


# ============================================================
# 3. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 4. LOAD CHUNKS
# ============================================================

CHUNKS_PATH = Path("data/chunks/chunks.json")
FAISS_PATH = Path("data/faiss/index.faiss")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")


# ============================================================
# 5. BM25 SETUP
# ============================================================

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


tokenized_chunks = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# ============================================================
# 6. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# 7. LOAD FAISS COSINE SIMILARITY INDEX
# ============================================================

if not FAISS_PATH.exists():
    raise FileNotFoundError(
        "FAISS index not found. Please run "
        "backend/semantic_search.py first."
    )

faiss_index = faiss.read_index(str(FAISS_PATH))

print("Search system ready")


# ============================================================
# 8. REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# 9. HYBRID RETRIEVAL
# ============================================================

def hybrid_search(query, top_k=5):

    # --------------------------------------------------------
    # BM25 SEARCH
    # --------------------------------------------------------

    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:10]


    # --------------------------------------------------------
    # SEMANTIC SEARCH
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    semantic_scores, semantic_indices = faiss_index.search(
        query_embedding,
        10
    )

    semantic_scores = semantic_scores[0]
    semantic_indices = semantic_indices[0]


    # --------------------------------------------------------
    # NORMALIZE BM25 SCORES
    # --------------------------------------------------------

    max_bm25 = max(bm25_scores)

    if max_bm25 > 0:
        bm25_normalized = {
            i: bm25_scores[i] / max_bm25
            for i in bm25_top_indices
        }
    else:
        bm25_normalized = {
            i: 0.0
            for i in bm25_top_indices
        }


    # --------------------------------------------------------
    # NORMALIZE SEMANTIC SCORES
    # --------------------------------------------------------

    semantic_min = min(semantic_scores)
    semantic_max = max(semantic_scores)

    semantic_normalized = {}

    for i, score in zip(
        semantic_indices,
        semantic_scores
    ):

        if semantic_max > semantic_min:
            normalized_score = (
                (float(score) - float(semantic_min))
                / (float(semantic_max) - float(semantic_min))
            )
        else:
            normalized_score = 0.0

        semantic_normalized[int(i)] = normalized_score


    # --------------------------------------------------------
    # COMBINE RESULTS
    # --------------------------------------------------------

    candidate_indices = set(
        bm25_top_indices
    ).union(
        int(i) for i in semantic_indices
    )


    hybrid_results = []

    for index in candidate_indices:

        keyword_score = bm25_normalized.get(
            index,
            0.0
        )

        semantic_score = semantic_normalized.get(
            index,
            0.0
        )

        # 50% keyword + 50% semantic
        hybrid_score = (
            0.5 * keyword_score
            + 0.5 * semantic_score
        )

        hybrid_results.append(
            (
                index,
                hybrid_score
            )
        )


    # --------------------------------------------------------
    # SORT BY HYBRID SCORE
    # --------------------------------------------------------

    hybrid_results.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # --------------------------------------------------------
    # RETURN TOP K
    # --------------------------------------------------------

    results = []

    for index, score in hybrid_results[:top_k]:

        results.append({
            "index": index,
            "document": chunks[index]["document"],
            "page": chunks[index]["page"],
            "text": chunks[index]["text"],
            "score": float(score)
        })

    return results


# ============================================================
# 10. GENERATE ANSWER USING GEMINI
# ============================================================

def generate_answer(query, retrieved_chunks):

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):

        context_parts.append(
            f"""
SOURCE {i}
Document: {chunk['document']}
Page: {chunk['page']}

Content:
{chunk['text']}
"""
        )

    context = "\n".join(context_parts)


    # --------------------------------------------------------
    # RAG PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an AI assistant for a Hybrid RAG Search System.

Answer the user's question using ONLY the information
provided in the CONTEXT below.

USER QUESTION:
{query}

CONTEXT:
{context}

IMPORTANT RULES:

1. Use only the provided context.
2. Do not make up information.
3. If the answer cannot be found in the context, say:
   "I could not find this information in the provided documents."
4. Treat each document as a separate source.
5. Do not combine rules, policies, percentages, dates,
   or requirements from different documents unless the
   context clearly supports doing so.
6. If different documents contain different requirements,
   clearly mention the difference and identify the
   document.
7. When an answer depends on a specific document,
   mention the document name.
8. Give a concise and easy-to-understand answer.
"""


    # --------------------------------------------------------
    # GEMINI API CALL
    # --------------------------------------------------------

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response.text:
            return response.text

        return (
            "I could not generate an answer from the "
            "provided documents."
        )


    # --------------------------------------------------------
    # HANDLE GEMINI ERRORS
    # --------------------------------------------------------

    except Exception as e:

        print(f"Gemini API error: {e}")

        error_message = str(e)

        # Gemini temporary server overload
        if "503" in error_message or "UNAVAILABLE" in error_message:

            return (
                "Gemini is temporarily unavailable because "
                "the AI service is experiencing high demand. "
                "Please try again in a few seconds."
            )

        # Rate-limit error
        if "429" in error_message:

            return (
                "The Gemini API rate limit has been reached. "
                "Please wait a little and try again."
            )

        # Other API errors
        return (
            "The answer generation service encountered an "
            "error. Please try again."
        )


# ============================================================
# 11. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Hybrid RAG Search API is running"
    }


# ============================================================
# 12. ASK ENDPOINT
# ============================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    # --------------------------------------------------------
    # EMPTY QUESTION CHECK
    # --------------------------------------------------------

    if not question:

        return {
            "question": "",
            "answer": "Please enter a question.",
            "sources": []
        }


    # --------------------------------------------------------
    # HYBRID RETRIEVAL
    # --------------------------------------------------------

    retrieved_chunks = hybrid_search(
        question,
        top_k=5
    )


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    answer = generate_answer(
        question,
        retrieved_chunks
    )


    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    sources = []

    for chunk in retrieved_chunks:

        sources.append({
            "document": chunk["document"],
            "page": chunk["page"]
        })


    # --------------------------------------------------------
    # RETURN RESPONSE
    # --------------------------------------------------------

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }