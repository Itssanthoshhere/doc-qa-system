// frontend/src/services/api.js
const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

// ── Sessions ─────────────────────────────────────────────────────────────────
export const createSession = (docIds = []) =>
  request("/sessions/", {
    method: "POST",
    body: JSON.stringify({ doc_ids: docIds }),
  });

export const getHistory = (sessionId) =>
  request(`/sessions/${sessionId}/history`);

export const clearHistory = (sessionId) =>
  request(`/sessions/${sessionId}/history`, { method: "DELETE" });

// ── Documents ─────────────────────────────────────────────────────────────────
export const uploadDocument = async (file, onProgress) => {
  const formData = new FormData();
  formData.append("file", file);

  // Use XHR for upload progress tracking
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE_URL}/documents/upload`);
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        const err = JSON.parse(xhr.responseText || "{}");
        reject(new Error(err.detail || "Upload failed"));
      }
    });
    xhr.addEventListener("error", () => reject(new Error("Network error")));
    xhr.send(formData);
  });
};

export const pollDocument = (docId) => request(`/documents/${docId}`);

export const listDocuments = () => request("/documents/");

export const deleteDocument = (docId) =>
  request(`/documents/${docId}`, { method: "DELETE" });

// ── QA ────────────────────────────────────────────────────────────────────────
export const askQuestion = (question, sessionId, docIds = null) =>
  request("/qa/ask", {
    method: "POST",
    body: JSON.stringify({
      question,
      session_id: sessionId,
      doc_ids: docIds,
    }),
  });