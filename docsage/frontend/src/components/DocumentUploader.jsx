// frontend/src/components/DocumentUploader.jsx
import { useState, useRef, useCallback } from "react";
import { uploadDocument, pollDocument } from "../services/api";

const POLL_INTERVAL_MS = 2000;
const MAX_POLLS = 60; // 2 minutes max wait

export default function DocumentUploader({ onDocumentReady }) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploads, setUploads] = useState([]); // { id, filename, progress, status }
  const fileInputRef = useRef(null);

  const updateUpload = (id, patch) => {
    setUploads((prev) =>
      prev.map((u) => (u.id === id ? { ...u, ...patch } : u))
    );
  };

  const processFile = useCallback(
    async (file) => {
      const uploadId = `${Date.now()}_${file.name}`;
      setUploads((prev) => [
        ...prev,
        { id: uploadId, filename: file.name, progress: 0, status: "uploading" },
      ]);

      try {
        // Upload
        const doc = await uploadDocument(file, (pct) => {
          updateUpload(uploadId, { progress: pct });
        });

        updateUpload(uploadId, { status: "processing", docId: doc.doc_id });
        onDocumentReady({ ...doc, status: "processing" });

        // Poll until ready
        let polls = 0;
        const poll = async () => {
          polls++;
          if (polls > MAX_POLLS) {
            updateUpload(uploadId, { status: "timeout" });
            return;
          }

          try {
            const updated = await pollDocument(doc.doc_id);
            if (updated.status === "ready") {
              updateUpload(uploadId, { status: "ready" });
              onDocumentReady({ ...updated, status: "ready" });
            } else if (updated.status === "failed") {
              updateUpload(uploadId, { status: "failed" });
            } else {
              setTimeout(poll, POLL_INTERVAL_MS);
            }
          } catch (_) {
            setTimeout(poll, POLL_INTERVAL_MS);
          }
        };

        setTimeout(poll, POLL_INTERVAL_MS);
      } catch (e) {
        updateUpload(uploadId, { status: "error", error: e.message });
      }
    },
    [onDocumentReady]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(false);
      const files = [...e.dataTransfer.files];
      files.forEach(processFile);
    },
    [processFile]
  );

  const handleFileInput = (e) => {
    [...e.target.files].forEach(processFile);
    e.target.value = "";
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const activeUploads = uploads.filter(
    (u) => u.status === "uploading" || u.status === "processing"
  );

  return (
    <div className="uploader">
      {/* Drop zone */}
      <div
        className={`drop-zone ${isDragging ? "drop-zone--active" : ""}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
      >
        <span className="drop-icon">↑</span>
        <span className="drop-label">
          {isDragging ? "Drop to upload" : "Upload document"}
        </span>
        <span className="drop-hint">PDF · DOCX · TXT · MD</span>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.doc,.txt,.md"
          multiple
          onChange={handleFileInput}
          className="visually-hidden"
        />
      </div>

      {/* Upload progress list */}
      {activeUploads.length > 0 && (
        <div className="upload-list">
          {activeUploads.map((u) => (
            <div key={u.id} className="upload-item">
              <span className="upload-name" title={u.filename}>
                {u.filename.length > 22
                  ? u.filename.slice(0, 20) + "…"
                  : u.filename}
              </span>
              <span className={`upload-status upload-status--${u.status}`}>
                {u.status === "uploading" ? `${u.progress}%` : "indexing…"}
              </span>
              {u.status === "uploading" && (
                <div className="progress-track">
                  <div
                    className="progress-fill"
                    style={{ width: `${u.progress}%` }}
                  />
                </div>
              )}
              {u.status === "processing" && (
                <div className="progress-track">
                  <div className="progress-fill progress-fill--indeterminate" />
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}