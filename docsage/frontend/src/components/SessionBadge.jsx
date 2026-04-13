// frontend/src/components/SessionBadge.jsx
export default function SessionBadge({ sessionId }) {
  if (!sessionId) return null;
  return (
    <div className="session-badge">
      <span className="session-dot" />
      <span className="session-label">
        Session {sessionId.slice(0, 8)}
      </span>
    </div>
  );
}