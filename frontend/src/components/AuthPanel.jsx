import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { loginUser, registerUser, logoutUser } from "../api.js";

const AUTH_EVENT = "auth-changed";
function emit() { window.dispatchEvent(new Event(AUTH_EVENT)); }

export default function AuthPanel() {
  const [open, setOpen]     = useState(false);
  const [mode, setMode]     = useState("login");
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState("");
  const [message, setMessage] = useState("");
  const [fields, setFields] = useState({ username: "", email: "", password: "" });
  const [token, setToken]   = useState(() => localStorage.getItem("auth_token") || "");

  function openAs(m) {
    setMode(m);
    setError("");
    setOpen(true);
  }

  useEffect(() => {
    function sync() { setToken(localStorage.getItem("auth_token") || ""); }
    window.addEventListener(AUTH_EVENT, sync);
    return () => window.removeEventListener(AUTH_EVENT, sync);
  }, []);

  // close panel when clicking outside
  useEffect(() => {
    if (!open) return;
    function onOutside(e) {
      if (!e.target.closest(".auth-wrap")) setOpen(false);
    }
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
  }, [open]);

  function set(key, val) { setFields((p) => ({ ...p, [key]: val })); }

  async function handleSubmit(e) {
    e.preventDefault();
    setError(""); setMessage(""); setLoading(true);
    try {
      if (mode === "login") {
        const res = await loginUser({ email: fields.email.trim(), password: fields.password });
        localStorage.setItem("auth_token", res.access_token);
        if (res.chatbot_session_id) localStorage.setItem("chat_thread_id", res.chatbot_session_id);
        setToken(res.access_token);
        emit();
        setOpen(false);
      } else {
        await registerUser({ username: fields.username.trim(), email: fields.email.trim(), password: fields.password });
        // Auto-login immediately after registration
        const res = await loginUser({ email: fields.email.trim(), password: fields.password });
        localStorage.setItem("auth_token", res.access_token);
        if (res.chatbot_session_id) localStorage.setItem("chat_thread_id", res.chatbot_session_id);
        setToken(res.access_token);
        emit();
        setOpen(false);
      }
      setFields({ username: "", email: "", password: "" });
    } catch (err) {
      setError(err.message || "Auth request failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleLogout() {
    const threadId = localStorage.getItem("chat_thread_id") || null;
    try { await logoutUser(token, threadId); } catch { /* best effort */ }
    localStorage.removeItem("auth_token");
    localStorage.removeItem("chat_thread_id");
    setToken("");
    emit();
  }

  return (
    <div className="auth-wrap">
      {token ? (
        <div className="auth-logged">
          <Link to="/profile" className="profile-link" title="Your profile">
            <div className="user-pill">
              <div className="user-dot" />
              Profile
            </div>
          </Link>
          <button className="ghost-btn" onClick={handleLogout}>Logout</button>
        </div>
      ) : open ? (
        <button className="ghost-btn" onClick={() => setOpen(false)}>✕ Close</button>
      ) : (
        <div className="auth-btns">
          <button className="auth-toggle" onClick={() => openAs("login")}>Login</button>
          <button className="auth-toggle auth-toggle--register" onClick={() => openAs("register")}>Register</button>
        </div>
      )}

      {open && !token && (
        <div className="auth-panel" onMouseDown={(e) => e.stopPropagation()}>
          <div className="auth-tabs">
            <button className={mode === "login" ? "tab-btn active" : "tab-btn"} onClick={() => { setMode("login"); setError(""); }}>Login</button>
            <button className={mode === "register" ? "tab-btn active" : "tab-btn"} onClick={() => { setMode("register"); setError(""); }}>Register</button>
          </div>
          <form className="auth-form" onSubmit={handleSubmit}>
            {mode === "register" && (
              <label>
                Username
                <input value={fields.username} onChange={(e) => set("username", e.target.value)} placeholder="Your username" required />
              </label>
            )}
            <label>
              Email
              <input value={fields.email} onChange={(e) => set("email", e.target.value)} placeholder="you@example.com" type="email" required />
            </label>
            <label>
              Password
              <input value={fields.password} onChange={(e) => set("password", e.target.value)} type="password" placeholder="••••••••" required />
            </label>
            {error   && <div className="form-error">{error}</div>}
            {message && <div className="form-message">{message}</div>}
            <button className="primary-btn" type="submit" disabled={loading}>
              {loading ? "Please wait…" : mode === "login" ? "Login" : "Press Enter to Register"}
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
