import React, { useState } from "react";
import "./App.css";

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const predictDifficulty = async () => {
    if (!text.trim()) {
      setError("Please paste a problem statement.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="card">
        <h1 className="title">AutoJudge</h1>
        <p className="subtitle">
          Machine Learning based programming problem difficulty predictor
        </p>

        <textarea
          className="input"
          placeholder="Paste the full problem statement here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <button className="btn" onClick={predictDifficulty} disabled={loading}>
          {loading ? "Analyzing..." : "Predict Difficulty"}
        </button>

        {error && <p className="error">{error}</p>}

        {result && (
          <div className="result">
            <div className={`pill ${result.difficulty}`}>
              {result.difficulty.toUpperCase()}
            </div>

            <div className="score">
              <span>{result.problem_score}</span>
              <small>/ 10</small>
            </div>

            <div className="scale">
              <div
                className="fill"
                style={{ width: `${result.problem_score * 10}%` }}
              />
            </div>

            <p className="hint">
              Estimated complexity score based on algorithms, constraints, and
              structure.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
