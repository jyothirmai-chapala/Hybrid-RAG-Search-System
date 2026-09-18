# Hybrid RAG Search System

> A document question-answering system that combines **BM25 keyword retrieval**, **semantic vector search using Sentence Transformers + FAISS**, and **Google Gemini** to generate grounded answers from PDF documents.

---

## 1. Project Overview

The **Hybrid RAG Search System** is a Retrieval-Augmented Generation (RAG) application designed to answer questions from a collection of PDF documents.

Traditional keyword search works well when the user's query contains the exact words present in a document. However, it may fail when the user asks the same question using different words.

Semantic search solves this problem by understanding the meaning of the query using embeddings. However, semantic search may sometimes miss important exact terms such as regulations, percentages, names, or specific keywords.

This project combines both approaches:

- **BM25** for keyword-based retrieval
- **Sentence Transformers** for semantic embeddings
- **FAISS** for efficient vector similarity search
- **Hybrid Retrieval** to combine keyword and semantic results
- **Google Gemini** for generating the final answer
- **React** for the user interface
- **FastAPI** for the backend API

The system also displays the source documents and page numbers used to generate the answer.

---

# 2. Problem Statement

Searching through large collections of academic and educational PDF documents manually can be time-consuming.

A simple keyword search has limitations:

- It requires matching words between the query and document.
- It may fail when the user uses different wording.
- It does not understand the semantic meaning of a question.

Pure semantic search also has limitations:

- Important exact keywords may receive less importance.
- Specific terms, numbers, regulations, and policy names may not always be retrieved accurately.

Therefore, this project implements a **Hybrid Retrieval system** that combines both keyword and semantic search to improve document retrieval.

---

# 3. Objectives

The main objectives of this project are:

1. Extract text from PDF documents.
2. Divide the extracted text into manageable chunks.
3. Implement keyword retrieval using BM25.
4. Generate semantic embeddings using Sentence Transformers.
5. Build a FAISS vector index.
6. Implement semantic similarity search using cosine similarity.
7. Combine BM25 and semantic search into a hybrid retrieval system.
8. Use an LLM to generate answers from retrieved document context.
9. Display the source document and page number for each retrieved result.
10. Evaluate BM25, semantic search, and hybrid retrieval using precision and recall.
11. Provide a simple web interface using React.

---

# 4. System Architecture

The complete system follows this pipeline:

```text
                    PDF Documents
                         |
                         v
                  Text Extraction
                         |
                         v
                      Chunking
                         |
              +----------+----------+
              |                     |
              v                     v
         BM25 Index            Embeddings
              |                     |
              v                     v
       Keyword Search             FAISS
              |                     |
              |              Semantic Search
              |                     |
              +----------+----------+
                         |
                         v
                 Hybrid Retrieval
                         |
                         v
                  Top Relevant Chunks
                         |
                         v
                    Context Builder
                         |
                         v
                    Google Gemini
                         |
                         v
                  Generated Answer
                         |
                         v
              Answer + Source Pages
