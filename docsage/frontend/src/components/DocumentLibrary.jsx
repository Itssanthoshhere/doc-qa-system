// frontend/src/components/DocumentLibrary.jsx
import { deleteDocument } from "../services/api";

const STATUS_LABEL = {
  queued: "Queued",
  processing: "Indexing…",
  ready: "Ready",
  failed: "Failed",
};

export default function DocumentLibrary({
  documents,
  activeDocIds,
  onToggleActive,
  onDelete,
}) {
  const handleDelete = async (docId) => {
    if (!confirm("Remove this document from the index?")) return;
    try {
      await deleteDocument(docId);
      onDelete(docId);
    } catch (e) {
      alert(`Delete failed: ${e.message}`);
    }
  };

  if (documents.length === 0) {
    return (
      <div className="library-empty">
        <div className="empty-icon">⬢</div>
        <h2>No documents yet</h2>
        <p>Upload documents from the sidebar to start building your library.</p>
      </div>
    );
  }

  return (
    <div className="library-container">
      <div className="library-header">
        <h1 className="library-title">Document Library</h1>
        <p className="library-subtitle">
          {documents.length} document{documents.length !== 1 ? "s" : ""} ·{" "}
          {activeDocIds.length > 0
            ? `${activeDocIds.length} active in search`
            : "searching all"}
        </p>
      </div>

      <div className="library-grid">
        {documents.map((doc) => (
          <div
            key={doc.doc_id}
            className={`doc-card ${
              activeDocIds.includes(doc.doc_id) ? "doc-card--active" : ""
            } doc-card--${doc.status}`}
          >
            <div className="doc-card-header">
              <span className={`doc-status-dot doc-status-dot--${doc.status}`} />
              <span className="doc-status-label">{STATUS_LABEL[doc.status] || doc.status}</span>
              <button
                className="doc-delete-btn"
                onClick={() => handleDelete(doc.doc_id)}
                title="Delete document"
              >
                ×
              </button>
            </div>

            <div className="doc-card-body">
              <div className="doc-icon">
                {doc.filename?.endsWith(".pdf") ? "⬛" : "⬜"}
              </div>
              <h3 className="doc-name" title={doc.filename}>
                {doc.filename}
              </h3>
              {doc.file_size_kb && (
                <p className="doc-meta">{Math.round(doc.file_size_kb)} KB</p>
              )}
            </div>

            {doc.status === "ready" && (
              <button
                className={`doc-toggle-btn ${
                  activeDocIds.includes(doc.doc_id) ? "active" : ""
                }`}
                onClick={() => onToggleActive(doc.doc_id)}
              >
                {activeDocIds.includes(doc.doc_id)
                  ? "Remove from search"
                  : "Add to search"}
              </button>
            )}

            {doc.status === "processing" && (
              <div className="doc-progress-bar">
                <div className="doc-progress-fill doc-progress-fill--indeterminate" />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}