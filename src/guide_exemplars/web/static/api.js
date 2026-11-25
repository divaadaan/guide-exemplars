// Thin fetch wrappers for the /api endpoints.

async function request(method, path, { body, headers = {} } = {}) {
  const init = { method, headers: { ...headers } };
  if (body instanceof FormData) {
    init.body = body;
  } else if (body !== undefined) {
    init.headers["content-type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await fetch(path, init);
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const detail = data && data.detail ? data.detail : res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

const keyHeader = (key) => (key ? { "x-llm-api-key": key } : {});

export const api = {
  health: () => request("GET", "/api/health"),
  backbones: () => request("GET", "/api/backbones"),
  presets: () => request("GET", "/api/presets"),
  tasks: () => request("GET", "/api/tasks"),
  datasets: () => request("GET", "/api/datasets"),
  importPreset: (body) => request("POST", "/api/datasets/from-preset", { body }),
  jobs: () => request("GET", "/api/jobs"),
  job: (id) => request("GET", `/api/jobs/${id}`),
  createJob: (body, key) => request("POST", "/api/jobs", { body, headers: keyHeader(key) }),
  cancelJob: (id) => request("POST", `/api/jobs/${id}/cancel`),
  resumeJob: (id, key) => request("POST", `/api/jobs/${id}/resume`, { headers: keyHeader(key) }),
  observations: (id) => request("GET", `/api/jobs/${id}/observations`),
  poolMap: (id, round) => request("GET", `/api/jobs/${id}/pool-map${round ? `?round=${round}` : ""}`),
  exemplarSet: (id) => request("GET", `/api/exemplar-sets/${id}`),
  grade: (body, key) => request("POST", "/api/grade", { body, headers: keyHeader(key) }),
};
