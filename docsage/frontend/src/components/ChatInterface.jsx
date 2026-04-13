// frontend/src/components/ChatInterface.jsx
import { useState, useRef, useEffect } from "react";
import { askQuestion, clearHistory } from "../services/api";
import AnswerCard from "./AnswerCard";

const PLACEHOLDER_QUESTIONS = [
  "What are the main contributions of this research?",
  "Summarize the key findings.",
  "What limitations are mentioned?",
  "What datasets were used?",
  "Compare the methods described.",
];

export default function ChatInterface({ sessionId, activeDocIds, hasDocuments }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleAsk = async (question = input.trim()) => {
    if (!question || isLoading) return;
    setError(null);
    setInput("");

    const userMsg = { id: Date.now(), role: "user", content: question };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const response = await askQuestion(question, sessionId, activeDocIds);
      const assistantMsg = {
        id: Date.now() + 1,
        role: "assistant",
        content: response.answer,
        meta: response,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (e) {
      setError(e.message);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "error",
          content: `Error: ${e.message}`,
        },
      ]);
    } finally {
      setIsLoading(false);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  const handleClear = async () => {
    try {
      await clearHistory(sessionId);
    } catch (_) {}
    setMessages([]);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  return (
    <div className="chat-container">
      {/* Header */}
      <div className="chat-header">
        <div>
          <h1 className="chat-title">Ask your documents</h1>
          {activeDocIds && (
            <p className="chat-subtitle">
              Searching {activeDocIds.length} selected document
              {activeDocIds.length !== 1 ? "s" : ""}
            </p>
          )}
        </div>
        {messages.length > 0 && (
          <button className="btn-ghost" onClick={handleClear}>
            Clear chat
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="messages-area">
        {messages.length === 0 && (
          <div className="empty-state">
            {!hasDocuments ? (
              <>
                <div className="empty-icon">◈</div>
                <h2>Upload a document to get started</h2>
                <p>Support PDF, DOCX, TXT, and Markdown files up to 50 MB.</p>
              </>
            ) : (
              <>
                <div className="empty-icon">⬡</div>
                <h2>What would you like to know?</h2>
                <p>Ask anything about your documents.</p>
                <div className="suggestion-chips">
                  {PLACEHOLDER_QUESTIONS.slice(0, 3).map((q) => (
                    <button
                      key={q}
                      className="suggestion-chip"
                      onClick={() => handleAsk(q)}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`message message--${msg.role}`}>
            {msg.role === "user" && (
              <div className="message-bubble message-bubble--user">
                {msg.content}
              </div>
            )}
            {msg.role === "assistant" && (
              <AnswerCard answer={msg.content} meta={msg.meta} />
            )}
            {msg.role === "error" && (
              <div className="message-bubble message-bubble--error">
                {msg.content}
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="message message--assistant">
            <div className="typing-indicator">
              <span /><span /><span />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="input-bar">
        <div className="input-wrapper">
          <textarea
            ref={inputRef}
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
            rows={1}
            disabled={isLoading}
          />
          <button
            className={`send-btn ${isLoading || !input.trim() ? "disabled" : ""}`}
            onClick={() => handleAsk()}
            disabled={isLoading || !input.trim()}
            aria-label="Send"
          >
            {isLoading ? (
              <span className="send-spinner" />
            ) : (
              <span className="send-arrow">↑</span>
            )}
          </button>
        </div>
        <p className="input-hint">
          Hybrid retrieval · Cross-encoder reranking · Adversarial filtering
        </p>
      </div>
    </div>
  );
}