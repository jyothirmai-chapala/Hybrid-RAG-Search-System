import { useState } from "react";
import ReactMarkdown from "react-markdown";
import "./style.css";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const askQuestion = async () => {
    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }

    setLoading(true);
    setAnswer("");
    setSources([]);
    setError("");

    try {
      const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error("Backend request failed");
      }

      const data = await response.json();

      setAnswer(data.answer || "No answer was generated.");
      setSources(data.sources || []);
    } catch (error) {
      console.error(error);

      setError(
        "Could not connect to the backend. Please make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && event.ctrlKey) {
      askQuestion();
    }
  };

  return (
    <div className="app">

      {/* Header */}
      <header className="header">
        <h1>Hybrid RAG Search System</h1>

        <p>
          BM25 + Semantic Search + FAISS + Gemini
        </p>
      </header>

      {/* Main Content */}
      <main className="container">

        {/* Question Section */}
        <section className="search-card">

          <h2>Ask a Question</h2>

          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Example: What is the minimum attendance requirement?"
            rows="5"
          />

          <div className="button-row">
            <button
              onClick={askQuestion}
              disabled={loading}
            >
              {loading ? "Searching..." : "Ask Question"}
            </button>

            <button
              className="clear-button"
              onClick={() => {
                setQuestion("");
                setAnswer("");
                setSources([]);
                setError("");
              }}
              disabled={loading}
            >
              Clear
            </button>
          </div>

          <p className="shortcut">
            Tip: Press Ctrl + Enter to ask
          </p>

          {/* Error */}
          {error && (
            <div className="error">
              {error}
            </div>
          )}

        </section>

        {/* Loading */}
        {loading && (
          <section className="result-card loading-card">
            <div className="loader"></div>

            <p>
              Searching documents and generating an answer...
            </p>
          </section>
        )}

        {/* Answer */}
        {!loading && answer && (
          <section className="result-card">

            <h2>Answer</h2>

            <div className="answer">
              <ReactMarkdown>
                {answer}
              </ReactMarkdown>
            </div>

          </section>
        )}

        {/* Sources */}
        {!loading && sources.length > 0 && (
          <section className="result-card">

            <h2>Sources</h2>

            <div className="sources">

              {sources.map((source, index) => (
                <div
                  className="source"
                  key={`${source.document}-${source.page}-${index}`}
                >

                  <div className="source-info">

                    <span className="source-number">
                      {index + 1}
                    </span>

                    <div>
                      <strong>
                        {source.document}
                      </strong>
                    </div>

                  </div>

                  <span className="page">
                    Page {source.page}
                  </span>

                </div>
              ))}

            </div>

          </section>
        )}

      </main>

      {/* Footer */}
      <footer>
        <p>
          Hybrid RAG Search System
        </p>

        <span>
          BM25 • FAISS • Sentence Transformers • Gemini
        </span>
      </footer>

    </div>
  );
}

export default App;