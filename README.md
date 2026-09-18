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


              5. How RAG Works in This Project

RAG stands for Retrieval-Augmented Generation.

Instead of asking the LLM to answer a question only from its internal knowledge, this system first retrieves relevant information from the provided documents.

The process is:

User Question
      |
      v
Retrieve relevant document chunks
      |
      v
Build context from retrieved chunks
      |
      v
Send context + question to Gemini
      |
      v
Generate grounded answer

This helps the system answer questions based on the uploaded document collection instead of relying only on the LLM's general knowledge.

6. Dataset / Documents

The project uses a collection of academic and educational PDF documents.

The documents include:

Academic Registration Guidelines
Examination Rules
Attendance Policy
R26 B.Tech Regulations
UG Regulations
B.Ed Curriculum and Instruction

The documents are stored in:

data/documents/
Important Note

One of the PDF files is image/scanned based and did not contain extractable text using the current PDF text extraction method.

OCR has not been implemented in this version of the project.

Therefore, that particular document may produce zero text chunks.

7. Text Extraction

PDF text is extracted using PyMuPDF.

File:

backend/pdf_processor.py

The extracted information includes:

Document name
Page number
Extracted text

The page information is preserved so that the system can later show the source page to the user.

Example metadata:

{
    "document": "IAR-Attendance-Policy.pdf",
    "page": 2,
    "text": "A minimum of 75% attendance is required..."
}
8. Text Chunking

Large documents are divided into smaller pieces called chunks.

File:

backend/chunker.py

The current configuration is:

Chunk size: 1000 characters
Overlap: 200 characters
Why chunking?

Sending an entire PDF to the LLM is inefficient.

Instead:

Large PDF
    |
    v
Small text chunks
    |
    v
Retrieve only relevant chunks
    |
    v
Send relevant context to Gemini

The overlap helps preserve information that may occur at the boundary between two chunks.

9. BM25 Keyword Retrieval

File:

backend/bm25_search.py

BM25 is a ranking algorithm commonly used for keyword-based information retrieval.

It considers factors such as:

Term frequency
Inverse document frequency
Document length

For example, if the user searches:

What is the minimum attendance requirement?

BM25 gives higher scores to chunks containing important matching words such as:

minimum
attendance
requirement
BM25 Advantage

BM25 is useful for queries containing:

Exact keywords
Names
Regulations
Specific terms
Numbers
Policy terminology
BM25 Limitation

BM25 mainly depends on lexical matching.

For example:

How much attendance should students maintain?

may not match as strongly with a document containing:

A minimum of 75% attendance is required.

even though both sentences have similar meanings.

10. Semantic Search

File:

backend/semantic_search.py

The project uses the Sentence Transformers model:

all-MiniLM-L6-v2

The model converts text into numerical vectors called embeddings.

Example:

"What is the minimum attendance requirement?"
                    |
                    v
              Text Embedding
                    |
                    v
        [0.12, -0.34, 0.56, ...]

The same process is applied to document chunks.

The system then compares the query embedding with document embeddings.

11. FAISS Vector Search

FAISS is used for efficient similarity search over embeddings.

The project uses:

FAISS IndexFlatIP

The embeddings are normalized before indexing.

Because normalized vectors are used with inner product similarity, the resulting score corresponds to cosine similarity.

Embedding
    |
    v
Normalize
    |
    v
FAISS IndexFlatIP
    |
    v
Cosine Similarity Search
Cosine Similarity

Cosine similarity measures how similar two vectors are in terms of their direction.

The value generally ranges from:

-1 to 1

A higher similarity indicates that the vectors are more similar.

12. Hybrid Retrieval

File:

backend/hybrid_search.py

The main feature of this project is combining:

BM25 Keyword Search
        +
Semantic Vector Search
        |
        v
Hybrid Retrieval

The system retrieves the top results from both methods.

The current implementation uses equal weighting:

Hybrid Score =
0.5 × Keyword Score
+
0.5 × Semantic Score

Before combining the scores, they are normalized so that the two retrieval methods can be combined more reasonably.

Why Hybrid Search?

BM25 is strong at:

Exact keyword matching
Specific terminology
Names
Numbers
Regulations

Semantic search is strong at:

Meaning
Paraphrased questions
Conceptually similar text
Different wording

Hybrid retrieval attempts to use the strengths of both approaches.

13. RAG Generation

File:

backend/rag.py

After hybrid retrieval, the top relevant chunks are collected and provided as context to Google Gemini.

The prompt instructs the model to:

Use only the retrieved document context.
Avoid inventing information.
State when the information cannot be found.
Keep different documents separate.
Mention document differences when conflicting information exists.
Identify the relevant source document when necessary.

For example, if a question cannot be answered from the available documents, the system can return:

I could not find this information in the provided documents.

This is an important part of preventing hallucinated answers.

14. Google Gemini

The project uses:

Google Gemini

Model:

gemini-2.5-flash

Gemini is used only after retrieval.

The basic flow is:

Question
   |
   v
Hybrid Retrieval
   |
   v
Top 5 Chunks
   |
   v
Context + Question
   |
   v
Gemini
   |
   v
Answer

The Gemini API key is stored in an environment file:

.env

Example:

GEMINI_API_KEY=your_api_key_here

The actual API key should never be committed to GitHub.

15. Source Attribution

The system does not only return an answer.

It also returns the documents and pages used during retrieval.

Example:

Sources

1. IAR-Attendance-Policy.pdf
   Page 2

2. UG_Regulations_(2024-25)_2024-7-17-11-36-22.pdf
   Page 5

This makes the answer easier to verify.

16. Backend

The backend is implemented using FastAPI.

Main API file:

backend/api.py

The backend provides:

POST /ask
Request
{
    "question": "What is the minimum attendance requirement?"
}
Response

The response contains:

{
    "answer": "Generated answer...",
    "sources": [
        {
            "document": "IAR-Attendance-Policy.pdf",
            "page": 2
        }
    ]
}

The backend also enables CORS so that the React frontend can communicate with the FastAPI server.

17. Frontend

The frontend is implemented using:

React
Vite
React Markdown
CSS

The frontend provides:

Question input
Ask Question button
Clear button
Loading indicator
Generated answer
Source documents
Source page numbers
Error handling
Markdown rendering

Frontend directory:

frontend/

The frontend communicates with:

http://127.0.0.1:8000/ask
18. User Interface Flow

The user enters a question:

What is the minimum attendance requirement?

Then:

React Frontend
      |
      v
FastAPI /ask
      |
      v
Hybrid Retrieval
      |
      +---- BM25
      |
      +---- Semantic Search + FAISS
      |
      v
Top Relevant Chunks
      |
      v
Gemini
      |
      v
Answer + Sources
      |
      v
React UI
19. Project Structure
hybrid-rag-search/
│
├── .env
├── requirements.txt
│
├── data/
│   ├── documents/
│   │   ├── Academic Registration PDF
│   │   ├── Examination Rules PDF
│   │   ├── Attendance Policy PDF
│   │   ├── R26 Regulations PDF
│   │   ├── UG Regulations PDF
│   │   └── B.Ed Curriculum PDF
│   │
│   ├── chunks/
│   │   └── chunks.json
│   │
│   └── faiss/
│       └── index.faiss
│
├── backend/
│   ├── pdf_processor.py
│   ├── chunker.py
│   ├── semantic_search.py
│   ├── bm25_search.py
│   ├── hybrid_search.py
│   ├── rag.py
│   └── api.py
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── style.css
│   │
│   ├── package.json
│   └── index.html
│
└── evaluation/
    ├── evaluation.py
    └── inspect_chunks.py
20. Technologies Used
Technology	Purpose
Python	Backend and RAG implementation
PyMuPDF	PDF text extraction
NumPy	Numerical operations
Sentence Transformers	Text embeddings
all-MiniLM-L6-v2	Embedding model
FAISS	Vector similarity search
rank-bm25	BM25 keyword retrieval
FastAPI	Backend REST API
Uvicorn	FastAPI server
Google Gemini	Answer generation
python-dotenv	Environment variable management
React	Frontend
Vite	Frontend development/build tool
React Markdown	Markdown answer rendering
CSS	User interface styling
21. Installation
Prerequisites

Install the following:

Python 3.12
Node.js
npm
Git
Step 1: Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>

Then:

cd hybrid-rag-search
22. Python Virtual Environment

Create a virtual environment:

python -m venv .venv

Activate it on Windows PowerShell:

.venv\Scripts\Activate.ps1

The terminal should show:

(.venv)
23. Install Python Dependencies

Run:

pip install -r requirements.txt
24. Configure Gemini API Key

Create a file named:

.env

in the project root.

Add:

GEMINI_API_KEY=YOUR_GEMINI_API_KEY

Replace the placeholder with your actual Gemini API key.

Security

Do not upload .env to GitHub.

The .gitignore file should contain:

.env
.venv/
__pycache__/
*.pyc
frontend/node_modules/
frontend/dist/
25. Prepare the Documents

Place PDF files inside:

data/documents/

Run the PDF extraction script:

python backend/pdf_processor.py

Then create chunks:

python backend/chunker.py

This creates:

data/chunks/chunks.json
26. Create the FAISS Index

Run:

python backend/semantic_search.py

This generates:

data/faiss/index.faiss

The current project contains approximately:

995 text chunks

from the available extractable PDF content.

27. Test BM25 Search

Run:

python backend/bm25_search.py

Enter a question such as:

What is the minimum attendance requirement?

The program displays the top BM25 results and their scores.

28. Test Hybrid Search

Run:

python backend/hybrid_search.py

Enter:

What is the minimum attendance requirement?

The program combines:

BM25 Score
+
Semantic Similarity

and displays the top hybrid results.

29. Test RAG

Run:

python backend/rag.py

Enter a question.

The system will:

Retrieve relevant chunks.
Combine BM25 and semantic results.
Build the context.
Send the context to Gemini.
Generate an answer.
Display source documents and pages.
30. Start the FastAPI Backend

From the project root:

uvicorn backend.api:app --reload

The backend runs at:

http://127.0.0.1:8000

FastAPI documentation is available at:

http://127.0.0.1:8000/docs
31. Start the React Frontend

Open another terminal.

Go to the frontend directory:

cd frontend

Install frontend dependencies:

npm install

Start the development server:

npm run dev

Vite will provide a local URL, usually similar to:

http://localhost:5173

Open that URL in a browser.

32. Example Questions

The following questions can be used to test the system.

Exact Keyword Query
What is the minimum attendance required for each module?

This tests keyword retrieval.

Semantic / Paraphrased Query
How much class attendance does a student need to maintain?

This tests whether semantic retrieval can handle different wording.

Policy Query
What action is taken when a student has insufficient attendance?

This tests retrieval of policy-related information.

Examination Query
What rules should students follow during examinations?

This tests retrieval from the examination rules document.

B.Tech Query
What does the B.Tech regulation document say about attendance?

This tests retrieval from the B.Tech regulations.

B.Ed Query
What is covered under curriculum and instruction in the B.Ed program?

This tests semantic retrieval over the large B.Ed document.

Unanswerable Query
What is the hostel mess fee for students?

The documents do not contain this information.

The expected behavior is:

I could not find this information in the provided documents.

This demonstrates that the system does not intentionally invent information when the required information is unavailable in the retrieved context.

33. Evaluation

The project evaluates three retrieval approaches:

BM25
Semantic Search
Hybrid Search

The evaluation uses a small manually created set of five questions.

Relevance is determined using the relevant document + page.

The evaluation calculates:

Precision = Relevant Retrieved Results / Total Retrieved Results

and

Recall = Relevant Retrieved Results / Total Relevant Results
34. Evaluation Results

The current evaluation produced the following results:

Query	Method	Precision	Recall
Minimum attendance	BM25	0.60	0.38
Minimum attendance	Semantic	0.60	0.38
Minimum attendance	Hybrid	0.80	0.50
Shortage of attendance	BM25	0.20	0.33
Shortage of attendance	Semantic	0.20	0.33
Shortage of attendance	Hybrid	0.20	0.33
Examination rules	BM25	0.20	0.17
Examination rules	Semantic	0.20	0.17
Examination rules	Hybrid	0.20	0.17
B.Tech regulations	BM25	0.00	0.00
B.Tech regulations	Semantic	0.00	0.00
B.Tech regulations	Hybrid	0.00	0.00
B.Ed curriculum	BM25	0.00	0.00
B.Ed curriculum	Semantic	0.20	0.33
B.Ed curriculum	Hybrid	0.00	0.00
Average Results
Method	Average Precision	Average Recall
BM25	0.20	0.17
Semantic Search	0.24	0.24
Hybrid Search	0.24	0.20
35. Interpretation of Evaluation

The evaluation shows that the retrieval performance varies depending on the type of query.

For the minimum attendance query, hybrid retrieval achieved:

Precision = 0.80
Recall = 0.50

which was higher than the corresponding BM25 and semantic results for that query.

However, the overall five-question average does not show hybrid retrieval outperforming semantic search on every metric.

Therefore, the results should be interpreted as an experiment on this particular document collection and question set rather than as a general claim that hybrid retrieval is always better.
