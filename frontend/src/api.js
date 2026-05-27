const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function buildHeaders(token) {
  const headers = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

async function handleJson(res) {
  if (!res.ok) {
    let message = "Request failed";
    try {
      const data = await res.json();
      message = data.detail || message;
    } catch { /* ignore */ }
    throw new Error(message);
  }
  return res.json();
}

// ── Anime ──────────────────────────────────────────────────────────────────────
export async function fetchAnimeList({ page, limit, search, genre }) {
  const url = new URL(`${API_BASE}/anime`);
  url.searchParams.set("page", page);
  url.searchParams.set("limit", limit);
  if (search) url.searchParams.set("search", search);
  if (genre)  url.searchParams.set("genre",  genre);
  return handleJson(await fetch(url));
}

export async function fetchAnimeDetail(animeId) {
  return handleJson(await fetch(`${API_BASE}/anime/${animeId}`));
}

export async function fetchSimilarAnime(animeId, token) {
  return handleJson(await fetch(`${API_BASE}/anime/${animeId}/similar`, {
    headers: buildHeaders(token),
  }));
}

// ── Auth (prefix /auth/*) ──────────────────────────────────────────────────────
export async function loginUser({ email, password }) {
  return handleJson(await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: buildHeaders(),
    body: JSON.stringify({ email, password }),
  }));
}

export async function registerUser({ username, email, password }) {
  return handleJson(await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: buildHeaders(),
    body: JSON.stringify({ username, email, password }),
  }));
}

export async function logoutUser(token, threadId) {
  return handleJson(await fetch(`${API_BASE}/auth/logout`, {
    method: "POST",
    headers: buildHeaders(token),
    body: JSON.stringify({ thread_id: threadId || null }),
  }));
}

// ── Chatbot ────────────────────────────────────────────────────────────────────
export async function sendChatMessage(message, threadId, token) {
  return handleJson(await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: buildHeaders(token),
    body: JSON.stringify({ message, thread_id: threadId || null }),
  }));
}

export async function fetchChatHistory(threadId, token) {
  return handleJson(await fetch(`${API_BASE}/chat/${threadId}/history`, {
    headers: buildHeaders(token),
  }));
}

// ── User profile ───────────────────────────────────────────────────────────────
export async function fetchMyProfile(token) {
  return handleJson(await fetch(`${API_BASE}/users/me`, {
    headers: buildHeaders(token),
  }));
}

export async function fetchMyRatings(token) {
  return handleJson(await fetch(`${API_BASE}/users/me/ratings`, {
    headers: buildHeaders(token),
  }));
}

export async function rateAnime(animeId, rating, token) {
  return handleJson(await fetch(`${API_BASE}/anime/${animeId}/rate`, {
    method: "POST",
    headers: buildHeaders(token),
    body: JSON.stringify({ rating }),
  }));
}
