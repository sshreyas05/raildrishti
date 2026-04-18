// ── Rail Drishti Frontend Config ──────────────────────────────────────────
// FIXED: Use relative URL so frontend always calls the same server it's hosted on.
// This works on Railway, Render, Fly.io, or any platform automatically.

const API_BASE = window.location.origin;

// Helper to build API URLs
function apiUrl(path) {
  return `${API_BASE}${path}`;
}

// Export for use in other JS files
window.API_BASE = API_BASE;
window.apiUrl = apiUrl;
