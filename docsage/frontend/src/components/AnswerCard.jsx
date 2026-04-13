// frontend/src/components/AnswerCard.jsx
import { useState } from "react";

const CONFIDENCE_COLORS = {
  high: "confidence--high",
  medium: "confidence--medium",
  low: "confidence--low",
  uncertain: "confidence--uncertain",
};

const RISK_ICONS = { low: "✓", medium: "⚠", high: "⚠" };

export default function AnswerCard({ answer, meta }) {
  const [showSources, setShowSources] = useState(false);
  const [showDebug, setShowDebug] = useState(false);

  const confidenceLabel = meta?.confidence_label || "uncertain";
  const confidencePct = meta?.confidence != null
    ? Math.round(meta.confidence * 100)
    : null;
  const adversarialRisk = meta?.adversarial_risk || "low";
  const sources = meta?.sources || [];
  const latency = meta?.total_latency_ms;

  return (
    <div className="answer-card">
      {/* Answer text */}
      <div className="answer-text">
        {meta?.is_impossible ? (
          <span className="answer-impossible">{answer}</span>
        ) : (
          answer
        )}
      </div>

      {/* Metadata row */}
      {meta && (
        <div className="answer-meta">
          {/* Confidence */}
          {confidencePct !== null && (
            <span className={`confidence-badge ${CONFIDENCE_COLORS[confidenceLabel]}`}>
              {confidencePct}% {confidenceLabel}
            </span>
          )}

          {/* Adversarial risk */}
          {adversarialRisk !== "low" && (
            <span className={`risk-badge risk-badge--${adversarialRisk}`}>
              {RISK_ICONS[adversarialRisk]} {adversarialRisk} adversarial risk
            </span>
          )}

          {/* Sources button */}
          {sources.length > 0 && (
            <button
              className="meta-btn"
              onClick={() => setShowSources((s) => !s)}
            >
              {showSources ? "Hide" : `${sources.length} source${sources.length !== 1 ? "s" : ""}`}
            </button>
          )}

          {/* Latency */}
          {latency != null && (
            <span className="latency-label">{Math.round(latency)} ms</span>
          )}

          {/* Debug toggle */}
          {meta.latency_breakdown && (
            <button
              className="meta-btn meta-btn--subtle"
              onClick={() => setShowDebug((s) => !s)}
            >
              {showDebug ? "Hide" : "Debug"}
            </button>
          )}
        </div>
      )}

      {/* Sources panel */}
      {showSources && sources.length > 0 && (
        <div className="sources-panel">
          {sources.map((src, i) => (
            <div key={i} className="source-item">
              <div className="source-header">
                <span className="source-index">{i + 1}</span>
                <span className="source-doc">{src.doc_id.slice(0, 8)}…</span>
                {src.page && (
                  <span className="source-page">p. {src.page}</span>
                )}
                {src.section && (
                  <span className="source-section">{src.section}</span>
                )}
              </div>
              <p className="source-snippet">"{src.snippet}"</p>
            </div>
          ))}
        </div>
      )}

      {/* Debug panel */}
      {showDebug && meta.latency_breakdown && (
        <div className="debug-panel">
          <table className="debug-table">
            <tbody>
              {Object.entries(meta.latency_breakdown).map(([key, val]) => (
                <tr key={key}>
                  <td className="debug-key">{key}</td>
                  <td className="debug-val">{Math.round(val)} ms</td>
                </tr>
              ))}
              <tr>
                <td className="debug-key">answer_type</td>
                <td className="debug-val">{meta.answer_type}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}