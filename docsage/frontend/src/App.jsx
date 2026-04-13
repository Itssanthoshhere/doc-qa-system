// frontend/src/App.jsx
import { useState, useEffect } from "react";
import DocumentUploader from "./components/DocumentUploader";
import ChatInterface from "./components/ChatInterface";
import DocumentLibrary from "./components/DocumentLibrary";
import SessionBadge from "./components/SessionBadge";
import { createSession } from "./services/api";
import "./styles/globals.css";

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [activeDocIds, setActiveDocIds] = useState([]);
  const [view, setView] = useState("chat"); // "chat" | "library"
  const [isCreatingSession, setIsCreatingSession] = useState(true);

  useEffect(() => {
    const initSession = async () => {
      try {
        const session = await createSession([]);
        setSessionId(session.session_id);
      } catch (e) {
        console.error("Session creation failed:", e);
        // Use a local fallback session ID
        setSessionId(`local_${Date.now()}`);
      } finally {
        setIsCreatingSession(false);
      }
    };
    initSession();
  }, []);

  const handleDocumentReady = (doc) => {
    setDocuments((prev) => {
      const exists = prev.find((d) => d.doc_id === doc.doc_id);
      if (exists) return prev.map((d) => (d.doc_id === doc.doc_id ? doc : d));
      return [...prev, doc];
    });
  };

  const handleDocumentDelete = (docId) => {
    setDocuments((prev) => prev.filter((d) => d.doc_id !== docId));
    setActiveDocIds((prev) => prev.filter((id) => id !== docId));
  };

  const toggleDocActive = (docId) => {
    setActiveDocIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  if (isCreatingSession) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner" />
        <p>Initializing DocSage…</p>
      </div>
    );
  }

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <span className="logo-icon">◈</span>
            <span className="logo-text">DocSage</span>
          </div>
          <p className="logo-tagline">Smart Document QA</p>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`nav-item ${view === "chat" ? "active" : ""}`}
            onClick={() => setView("chat")}
          >
            <span className="nav-icon">⬡</span> Chat
          </button>
          <button
            className={`nav-item ${view === "library" ? "active" : ""}`}
            onClick={() => setView("library")}
          >
            <span className="nav-icon">⬢</span> Library
            {documents.length > 0 && (
              <span className="nav-badge">{documents.length}</span>
            )}
          </button>
        </nav>

        <div className="sidebar-upload">
          <DocumentUploader onDocumentReady={handleDocumentReady} />
        </div>

        {/* Active filters */}
        {documents.length > 0 && (
          <div className="active-docs">
            <p className="active-docs-label">Search scope</p>
            {documents
              .filter((d) => d.status === "ready")
              .map((doc) => (
                <button
                  key={doc.doc_id}
                  className={`doc-filter-chip ${
                    activeDocIds.includes(doc.doc_id) ? "selected" : ""
                  }`}
                  onClick={() => toggleDocActive(doc.doc_id)}
                  title={doc.filename}
                >
                  <span className="chip-dot" />
                  {doc.filename.length > 24
                    ? doc.filename.slice(0, 22) + "…"
                    : doc.filename}
                </button>
              ))}
            {activeDocIds.length > 0 && (
              <button
                className="clear-filter"
                onClick={() => setActiveDocIds([])}
              >
                Clear filter
              </button>
            )}
          </div>
        )}

        <div className="sidebar-footer">
          <SessionBadge sessionId={sessionId} />
        </div>
      </aside>

      {/* Main content */}
      <main className="main-content">
        {view === "chat" ? (
          <ChatInterface
            sessionId={sessionId}
            activeDocIds={activeDocIds.length > 0 ? activeDocIds : null}
            hasDocuments={documents.some((d) => d.status === "ready")}
          />
        ) : (
          <DocumentLibrary
            documents={documents}
            activeDocIds={activeDocIds}
            onToggleActive={toggleDocActive}
            onDelete={handleDocumentDelete}
          />
        )}
      </main>
    </div>
  );
}