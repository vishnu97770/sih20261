const API_PREFIX = import.meta.env.VITE_API_PREFIX || "/api";
const DATASET_EXTENSIONS = new Set(["csv", "xlsx", "xls"]);
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
const TOKEN_KEY = "mi_token";

// Any file can be uploaded. CSV/XLSX/XLS feed the analytics pipeline; everything
// else (PDF, DOCX, images, text, or anything not listed here) is still accepted
// and routed to the document library by the backend.
export function validateDatasetFile(file) {
  if (!file) return "Choose a file to upload.";
  if (file.size > MAX_UPLOAD_BYTES) {
    return "File is too large. The maximum upload size is 25 MB.";
  }
  return "";
}

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

export function isDatasetFile(file) {
  const extension = file?.name?.split(".").pop()?.toLowerCase();
  return DATASET_EXTENSIONS.has(extension);
}

async function request(path, options = {}) {
  const token = getToken();
  const headers = { ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(`${API_PREFIX}${path}`, { ...options, headers });

  if (res.status === 401) {
    clearToken();
    window.dispatchEvent(new Event("mi-auth-expired"));
  }

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      try {
        detail = await res.text();
      } catch {
        /* ignore */
      }
    }
    throw new Error(detail);
  }

  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return res.json();
  }
  return res.blob();
}

export function buildQuery(params = {}) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value == null) return;
    if (Array.isArray(value)) {
      value.forEach((item) => {
        if (item !== "" && item != null) search.append(key, item);
      });
      return;
    }
    if (value !== "") {
      search.set(key, value);
    }
  });
  const query = search.toString();
  return query ? `?${query}` : "";
}

export function apiGet(path, params = {}, extra = {}) {
  return request(`${path}${buildQuery(params)}`, { method: "GET", ...extra });
}

export function apiJson(path, method = "GET", body, extra = {}) {
  const options = { method, ...extra };
  if (body !== undefined) {
    if (body instanceof FormData) {
      options.body = body;
    } else {
      options.headers = { "Content-Type": "application/json", ...(extra.headers || {}) };
      options.body = JSON.stringify(body);
    }
  }
  return request(path, options);
}

// FastAPI's OAuth2PasswordRequestForm (used by /auth/login) expects a
// form-encoded body, not JSON.
function apiForm(path, fields) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams(fields).toString(),
  });
}

export const api = {
  authLogin: (email, password) => apiForm("/auth/login", { username: email, password }),
  authRegister: (email, password) => apiJson("/auth/register", "POST", { username: email, password }),
  authMe: () => apiGet("/auth/me"),
  health: () => apiGet("/health"),
  session: () => apiGet("/session"),
  filters: () => apiGet("/filters"),
  quality: () => apiGet("/quality"),
  kpis: (params) => apiGet("/kpis", params),
  production: (params) => apiGet("/production", params),
  anomalies: (params) => apiGet("/anomalies", params),
  forecast: (params) => apiGet("/forecast", params),
  askAssistant: (question) => apiJson("/ask", "POST", { question }),
  generateReport: (payload) => apiJson("/report", "POST", payload),
  downloadReportPdf: (params) => apiGet("/report/pdf", params),
  uploadDataset: (file) => {
    const form = new FormData();
    form.append("file", file);
    return apiJson("/upload", "POST", form);
  },
  uploadDocument: (file) => {
    const form = new FormData();
    form.append("file", file);
    return apiJson("/documents/upload", "POST", form);
  },
  loadDemoDataset: () => apiJson("/demo/load", "POST"),
  removeDataset: () => apiJson("/dataset", "DELETE"),
  removeDocument: (id) => apiJson(`/documents/${id}`, "DELETE"),
  documentFile: (id) => request(`/documents/${id}/file`, { method: "GET" }),
  retrainModels: () => apiJson("/train-models", "POST"),
  suggestions: () => apiGet("/assistant/suggestions"),
  documents: () => apiGet("/documents"),
  wordcloud: () => apiGet("/insights/wordcloud"),
  notifications: () => apiGet("/notifications"),
  markNotificationsRead: () => apiJson("/notifications/read", "POST"),
};

export { request };
