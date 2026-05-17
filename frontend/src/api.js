import { getToken, clearToken } from "./auth.js";

const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, { method = "GET", body, auth = false } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (auth) {
    const token = getToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (res.status === 401 && auth) {
    clearToken();
  }

  let data = null;
  try {
    data = await res.json();
  } catch {
    /* ignore non-JSON */
  }

  if (!res.ok) {
    const msg = data?.detail || res.statusText || "Request failed";
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

export const api = {
  register: (payload) => request("/auth/register", { method: "POST", body: payload }),
  login: (payload) => request("/auth/login", { method: "POST", body: payload }),
  genres: () => request("/genres"),
  allAnime: (limit = 24, offset = 0) =>
    request(`/all_anime?limit=${limit}&offset=${offset}`),
  animeDetail: (id) => request(`/anime/${id}`),
  similarAnime: (id, k = 10) => request(`/similar_anime/${id}?k=${k}`),
  rate: (anime_id, rating) =>
    request("/rate", { method: "POST", body: { anime_id, rating }, auth: true }),
  recommendForMe: () => request("/recommend_for_me", { auth: true }),
};
