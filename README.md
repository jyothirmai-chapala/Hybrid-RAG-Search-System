
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

## 2. Problem Statement

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

## 3. Objectives

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
10. Evaluate BM25, semantic search, and hybrid retrieval using Precision and Recall.
11. Provide a simple web interface using React.

---

## 4. System Architecture

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
````

---

## 5. How RAG Works in This Project

RAG stands for **Retrieval-Augmented Generation**.

Instead of asking the LLM to answer a question only from its internal knowledge, this system first retrieves relevant information from the provided documents.

The process is:

```text
User Question
      |
      v
Retrieve Relevant Document Chunks
      |
      v
Build Context from Retrieved Chunks
      |
      v
Send Context + Question to Gemini
      |
      v
Generate Grounded Answer
```

This helps the system answer questions based on the provided document collection instead of relying only on the LLM's general knowledge.

---

## 6. Dataset / Documents

The project uses a collection of academic and educational PDF documents.

The documents include:

* Academic Registration Guidelines
* Examination Rules
* Attendance Policy
* R26 B.Tech Regulations
* UG Regulations
* B.Ed Curriculum and Instruction

The documents are stored in:

```text
data/documents/
```

### Important Note

One of the PDF files is image/scanned based and did not contain extractable text using the current PDF text extraction method.

OCR has not been implemented in this version of the project.

Therefore, that particular document may produce zero text chunks.

---

## 7. Text Extraction

PDF text is extracted using **PyMuPDF**.

File:

```text
backend/pdf_processor.py
```

The extracted information includes:

* Document name
* Page number
* Extracted text

The page information is preserved so that the system can later show the source page to the user.

Example metadata:

```json
{
    "document": "IAR-Attendance-Policy.pdf",
    "page": 2,
    "text": "A minimum of 75% attendance is required..."
}
```

---

## 8. Text Chunking

Large documents are divided into smaller pieces called **chunks**.

File:

```text
backend/chunker.py
```

The current configuration is:

```text
Chunk size: 1000 characters
Overlap: 200 characters
```

### Why Chunking?

Sending an entire PDF to the LLM is inefficient.

Instead:

```text
Large PDF
    |
    v
Small Text Chunks
    |
    v
Retrieve Only Relevant Chunks
    |
    v
Send Relevant Context to Gemini
```

The overlap helps preserve information that may occur at the boundary between two chunks.

The current document collection produces approximately:

```text
995 chunks
```

The chunks are stored in:

```text
data/chunks/chunks.json
```

---

## 9. BM25 Keyword Retrieval

File:

```text
backend/bm25_search.py
```

BM25 is a ranking algorithm commonly used for keyword-based information retrieval.

It considers factors such as:

* Term frequency
* Inverse document frequency
* Document length

For example, if the user searches:

```text
What is the minimum attendance requirement?
```

BM25 gives higher scores to chunks containing important matching words such as:

```text
minimum
attendance
requirement
```

### BM25 Advantages

BM25 is useful for queries containing:

* Exact keywords
* Names
* Regulations
* Specific terms
* Numbers
* Policy terminology

### BM25 Limitation

BM25 mainly depends on lexical matching.

For example:

```text
How much attendance should students maintain?
```

may not match as strongly with a document containing:

```text
A minimum of 75% attendance is required.
```

even though both sentences have similar meanings.

---

## 10. Semantic Search

File:

```text
backend/semantic_search.py
```

The project uses the Sentence Transformers model:

```text
all-MiniLM-L6-v2
```

The model converts text into numerical vectors called **embeddings**.

Example:

```text
"What is the minimum attendance requirement?"
                    |
                    v
              Text Embedding
                    |
                    v
        [0.12, -0.34, 0.56, ...]
```

The same process is applied to document chunks.

The system then compares the query embedding with document embeddings to find semantically similar content.

Semantic search is useful when the user uses different words but the meaning is similar to the document content.

---

## 11. FAISS Vector Search

FAISS is used for efficient similarity search over embeddings.

The project uses:

```text
FAISS IndexFlatIP
```

The embeddings are normalized before indexing.

Because normalized vectors are used with inner product similarity, the resulting score corresponds to **cosine similarity**.

The process is:

```text
Document Embedding
       |
       v
L2 Normalization
       |
       v
FAISS IndexFlatIP
       |
       v
Cosine Similarity Search
```

For a user query:

```text
User Question
       |
       v
Query Embedding
       |
       v
Normalization
       |
       v
FAISS Search
       |
       v
Most Similar Chunks
```

The FAISS index is stored in:

```text
data/faiss/index.faiss
```

---

## 12. Cosine Similarity

Cosine similarity measures how similar two vectors are based on the angle between them.

The value generally ranges from:

```text
-1 to 1
```

A higher cosine similarity indicates that two vectors are more similar in direction.

In this project, document and query embeddings are normalized before FAISS search. Therefore, the inner product returned by `IndexFlatIP` corresponds to cosine similarity.

---

## 13. Hybrid Retrieval

File:

```text
backend/hybrid_search.py
```

The main feature of this project is combining:

```text
BM25 Keyword Search
        +
Semantic Vector Search
        |
        v
Hybrid Retrieval
```

The system retrieves candidate results from both methods.

The current implementation uses equal weighting:

```text
Hybrid Score =
0.5 × Normalized BM25 Score
+
0.5 × Normalized Semantic Score
```

Before combining the scores, they are normalized so that the two retrieval methods can be combined more reasonably.

### Why Hybrid Search?

BM25 is strong at:

* Exact keyword matching
* Specific terminology
* Names
* Numbers
* Regulations
* Policy terms

Semantic search is strong at:

* Meaning
* Paraphrased questions
* Conceptually similar text
* Different wording

Hybrid retrieval combines these two retrieval approaches.

---

## 14. RAG Generation

File:

```text
backend/rag.py
```

After hybrid retrieval, the top relevant chunks are collected and provided as context to Google Gemini.

The prompt instructs the model to:

* Use only the retrieved document context.
* Avoid inventing information.
* State when the information cannot be found.
* Keep different documents separate.
* Mention document differences when conflicting information exists.
* Identify the relevant source document when necessary.

For example, if a question cannot be answered from the available documents, the system can return:

```text
I could not find this information in the provided documents.
```

This helps reduce unsupported or hallucinated answers.

---

## 15. Google Gemini

The project uses:

```text
Google Gemini
```

Model:

```text
gemini-2.5-flash
```

Gemini is used after the retrieval stage.

The basic flow is:

```text
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
```

The Gemini API key is stored in an environment file:

```text
.env
```

Example:

```text
GEMINI_API_KEY=your_api_key_here
```

The actual API key should **never** be committed to GitHub.

---

## 16. Source Attribution

The system does not only return an answer.

It also returns the documents and pages used during retrieval.

Example:

```text
Sources

1. IAR-Attendance-Policy.pdf
   Page 2

2. UG_Regulations_(2024-25)_2024-7-17-11-36-22.pdf
   Page 5
```

This makes the generated answer easier to verify against the original documents.

---

## 17. Backend

The backend is implemented using **FastAPI**.

Main API file:

```text
backend/api.py
```

The main endpoint is:

```text
POST /ask
```

### Request

```json
{
    "question": "What is the minimum attendance requirement?"
}
```

### Response

```json
{
    "answer": "Generated answer...",
    "sources": [
        {
            "document": "IAR-Attendance-Policy.pdf",
            "page": 2
        }
    ]
}
```

The backend runs on:

```text
http://127.0.0.1:8000
```

FastAPI documentation is available at:

```text
http://127.0.0.1:8000/docs
```

The backend also enables CORS so that the React frontend can communicate with the FastAPI server.

Temporary Gemini service errors such as rate limits and service unavailability are handled with user-friendly error messages.

---

## 18. Frontend

The frontend is implemented using:

* React
* Vite
* React Markdown
* CSS

The frontend provides:

* Question input
* Ask Question button
* Clear button
* Loading indicator
* Generated answer
* Source documents
* Source page numbers
* Error handling
* Markdown rendering

Frontend directory:

```text
frontend/
```

The frontend communicates with:

```text
http://127.0.0.1:8000/ask
```

---

## 19. User Interface Flow

The complete user interaction is:

```text
User enters a question
        |
        v
React Frontend
        |
        v
FastAPI /ask
        |
        v
Hybrid Retrieval
        |
        +------ BM25 Keyword Search
        |
        +------ Semantic Search + FAISS
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
```

The application also supports:

```text
Ctrl + Enter
```

for submitting the question.

---

## 20. Project Structure

```text
hybrid-rag-search/
│
├── .env
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── documents/
│   │   ├── Detailed Guidelines - Academic Registration
│   │   ├── Examination_Rules_for_Students.pdf
│   │   ├── IAR-Attendance-Policy.pdf
│   │   ├── R26_Regulations_B.Tech.pdf
│   │   ├── UG_Regulations_(2024-25)_2024-7-17-11-36-22.pdf
│   │   └── __UG_B.Ed._Education_70122-Curriculum and Instruction_1747.pdf
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
```

---

## 21. Technologies Used

| Technology            | Purpose                             |
| --------------------- | ----------------------------------- |
| Python                | Backend and RAG implementation      |
| PyMuPDF               | PDF text extraction                 |
| NumPy                 | Numerical operations                |
| Sentence Transformers | Text embeddings                     |
| all-MiniLM-L6-v2      | Embedding model                     |
| FAISS                 | Vector similarity search            |
| rank-bm25             | BM25 keyword retrieval              |
| FastAPI               | Backend REST API                    |
| Uvicorn               | FastAPI server                      |
| Google Gemini         | Answer generation                   |
| python-dotenv         | Environment variable management     |
| React                 | Frontend                            |
| Vite                  | Frontend development and build tool |
| React Markdown        | Markdown answer rendering           |
| CSS                   | User interface styling              |

---

## 22. Installation

### Prerequisites

Install the following:

* Python 3.12
* Node.js
* npm
* Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/jyothirmai-chapala/Hybrid-RAG-Search-System.git
cd Hybrid-RAG-Search-System
```

### Step 2: Create a Python Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

After activation, the terminal should show:

```text
(.venv)
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 23. Configure Gemini API Key

Create a file named:

```text
.env
```

in the project root.

Add:

```text
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

Replace the placeholder with your actual Gemini API key.

### Security

Do not upload `.env` to GitHub.

The `.gitignore` file should contain:

```text
.env
.venv/
__pycache__/
*.pyc
frontend/node_modules/
frontend/dist/
```

---

## 24. Prepare the Documents

Place the PDF files inside:

```text
data/documents/
```

Run the PDF extraction script:

```bash
python backend/pdf_processor.py
```

Then create the chunks:

```bash
python backend/chunker.py
```

This creates:

```text
data/chunks/chunks.json
```

The current document collection produces approximately:

```text
995 chunks
```

from the available extractable PDF content.

---

## 25. Create the FAISS Index

Run:

```bash
python backend/semantic_search.py
```

This generates:

```text
data/faiss/index.faiss
```

The FAISS index contains the embeddings of the document chunks and is used for semantic similarity search.

---

## 26. Test BM25 Search

Run:

```bash
python backend/bm25_search.py
```

Enter a question such as:

```text
What is the minimum attendance requirement?
```

The program displays the top BM25 results and their relevance scores.

---

## 27. Test Semantic Search

The semantic search implementation is contained in:

```text
backend/semantic_search.py
```

It uses:

```text
all-MiniLM-L6-v2
```

to generate embeddings and FAISS to retrieve semantically similar chunks.

---

## 28. Test Hybrid Search

Run:

```bash
python backend/hybrid_search.py
```

Enter:

```text
What is the minimum attendance requirement?
```

The program combines:

```text
BM25 Score
+
Semantic Similarity
```

and displays the top hybrid results.

---

## 29. Test RAG

Run:

```bash
python backend/rag.py
```

Enter a question.

The system will:

1. Retrieve relevant chunks.
2. Combine BM25 and semantic results.
3. Build the context.
4. Send the context to Gemini.
5. Generate an answer.
6. Display source documents and pages.

---

## 30. Start the FastAPI Backend

From the project root, run:

```bash
uvicorn backend.api:app --reload
```

The backend runs at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 31. Start the React Frontend

Open another terminal.

Go to the frontend directory:

```bash
cd frontend
```

Install frontend dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Vite will provide a local URL, usually similar to:

```text
http://localhost:5173
```

Open the displayed URL in a browser.

---

## 32. Example Questions

The following questions can be used to test the system.

### Attendance Query

```text
What is the minimum attendance required for each module?
```

### Semantic / Paraphrased Query

```text
How much class attendance does a student need to maintain?
```

### Attendance Policy Query

```text
What action is taken when a student has insufficient attendance?
```

### Examination Query

```text
What rules should students follow during examinations?
```

### B.Tech Query

```text
What does the B.Tech regulation document say about attendance?
```

### B.Ed Query

```text
What is covered under curriculum and instruction in the B.Ed program?
```

### Unanswerable Query

```text
What is the hostel mess fee for students?
```

The provided documents do not contain this information.

The expected behavior is:

```text
I could not find this information in the provided documents.
```

This demonstrates that the system does not intentionally invent information when the required information is unavailable in the provided documents.

---

## 33. Evaluation

The project evaluates three retrieval approaches:

1. BM25
2. Semantic Search
3. Hybrid Search

The evaluation uses:

* Precision
* Recall
* Page-level relevance

The evaluation contains five manually selected questions based on the available documents.

The questions cover:

* Minimum attendance requirement
* Attendance shortage
* Examination rules
* B.Tech regulations
* B.Ed curriculum and instruction

---

## 34. Evaluation Method

For every test question:

1. BM25 retrieves the top results.
2. Semantic search retrieves the top results.
3. Hybrid search retrieves the top results.
4. Retrieved pages are compared with manually identified relevant pages.
5. Precision and Recall are calculated.

### Precision

Precision measures how many of the retrieved results are relevant.

```text
Precision =
Relevant Retrieved Results
--------------------------
Total Retrieved Results
```

### Recall

Recall measures how many of the relevant results were retrieved.

```text
Recall =
Relevant Retrieved Results
--------------------------
Total Relevant Results
```

---

## 35. Evaluation Results

### Query-Level Results

| Query                  | Method   | Precision | Recall |
| ---------------------- | -------- | --------: | -----: |
| Minimum attendance     | BM25     |      0.60 |   0.38 |
| Minimum attendance     | Semantic |      0.60 |   0.38 |
| Minimum attendance     | Hybrid   |      0.80 |   0.50 |
| Shortage of attendance | BM25     |      0.20 |   0.33 |
| Shortage of attendance | Semantic |      0.20 |   0.33 |
| Shortage of attendance | Hybrid   |      0.20 |   0.33 |
| Examination rules      | BM25     |      0.20 |   0.17 |
| Examination rules      | Semantic |      0.20 |   0.17 |
| Examination rules      | Hybrid   |      0.20 |   0.17 |
| B.Tech regulations     | BM25     |      0.00 |   0.00 |
| B.Tech regulations     | Semantic |      0.00 |   0.00 |
| B.Tech regulations     | Hybrid   |      0.00 |   0.00 |
| B.Ed curriculum        | BM25     |      0.00 |   0.00 |
| B.Ed curriculum        | Semantic |      0.20 |   0.33 |
| B.Ed curriculum        | Hybrid   |      0.00 |   0.00 |

### Average Results

| Method          | Average Precision | Average Recall |
| --------------- | ----------------: | -------------: |
| BM25            |              0.20 |           0.17 |
| Semantic Search |              0.24 |           0.24 |
| Hybrid Search   |              0.24 |           0.20 |

---

## 36. Interpretation of Evaluation

The evaluation shows that retrieval performance varies depending on the type of query.

For the **minimum attendance** query, hybrid retrieval achieved:

```text
Precision = 0.80
Recall = 0.50
```

while BM25 and semantic retrieval achieved:

```text
Precision = 0.60
Recall = 0.38
```

for that query.

However, the overall five-question average does not show hybrid retrieval outperforming semantic search on every metric.

The average results were:

```text
BM25:
Precision = 0.20
Recall = 0.17

Semantic Search:
Precision = 0.24
Recall = 0.24

Hybrid Search:
Precision = 0.24
Recall = 0.20
```

Therefore, the results should be interpreted as an experiment on this particular document collection and question set rather than as a general claim that hybrid retrieval is always better.

The evaluation demonstrates that different retrieval methods behave differently depending on the query and document content.

---

## 37. Limitations

The current version of the project has the following limitations:

1. One scanned PDF does not currently provide extractable text.
2. OCR is not implemented.
3. The evaluation contains only five questions.
4. Relevance labels are manually created.
5. The evaluation uses page-level relevance.
6. Hybrid weights are fixed at 0.5 and 0.5.
7. The system retrieves a fixed number of top results.
8. No dedicated reranking model is currently used.
9. Very broad questions may retrieve less relevant chunks.
10. Gemini availability and API rate limits can temporarily affect answer generation.

---

## 38. Future Enhancements

The system can be improved in the future by adding:

### OCR Support

OCR can be added to process scanned PDFs.

```text
Scanned PDF
     |
     v
    OCR
     |
     v
Extracted Text
     |
     v
Chunking
```

### Better Chunking

The current system uses character-based chunking.

Future versions can use:

* Sentence-based chunking
* Paragraph-based chunking
* Section-aware chunking

### Improved Hybrid Retrieval

Different BM25 and semantic weights can be tested to determine how retrieval performance changes.

For example:

```text
BM25 = 0.3
Semantic = 0.7
```

or:

```text
BM25 = 0.7
Semantic = 0.3
```

### Larger Evaluation Dataset

More questions and manually verified relevance labels can be added to make the evaluation more reliable.

### Reranking

A reranking model can be added after initial retrieval to improve the ordering of retrieved chunks.

### Document Upload

A future version can allow users to upload their own documents directly through the web interface.

### Additional Document Formats

The system can be extended to support:

* DOCX
* TXT
* HTML
* Markdown

### Conversation History

The application can support multi-turn conversations and follow-up questions.

### Evaluation Dashboard

A dashboard can be added to visualize:

* Precision
* Recall
* Retrieval scores
* Query-level results

---

## 39. Security

The Gemini API key is stored in the `.env` file.

Example:

```text
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

The actual API key should never be written directly inside the Python source code.

The `.env` file should be included in `.gitignore`:

```text
.env
```

Before pushing the project to GitHub, verify that `.env` is not being tracked by Git.

---

## 40. Conclusion

The **Hybrid RAG Search System** is an end-to-end document question-answering application that combines traditional information retrieval, semantic vector search, hybrid retrieval, and generative AI.

The complete workflow is:

```text
PDF Documents
      |
      v
Text Extraction
      |
      v
Text Chunking
      |
      +-------------------+
      |                   |
      v                   v
    BM25             Embeddings
      |                   |
      v                   v
Keyword Search          FAISS
      |            Semantic Search
      |                   |
      +---------+---------+
                |
                v
        Hybrid Retrieval
                |
                v
       Relevant Chunks
                |
                v
             Gemini
                |
                v
        Grounded Answer
                |
                v
         Answer + Sources
```

The key idea of the project is to combine **exact keyword matching** with **semantic understanding**.

BM25 helps retrieve information containing important exact terms, while semantic search helps retrieve information with similar meaning even when different words are used.

The retrieved information is then passed to Google Gemini to generate a natural-language answer based on the available document context.

The project demonstrates the practical implementation of:

* Retrieval-Augmented Generation (RAG)
* BM25
* Text Embeddings
* Semantic Search
* Cosine Similarity
* FAISS
* Hybrid Retrieval
* Large Language Models
* Grounded Answer Generation
* FastAPI
* React
* Google Gemini

---

## 41. Repository

GitHub Repository:

[https://github.com/jyothirmai-chapala/Hybrid-RAG-Search-System](https://github.com/jyothirmai-chapala/Hybrid-RAG-Search-System)

```
```
