import { useEffect, useRef, useState } from "react";
import { sendChatMessage } from "../api.js";

const WELCOME = "Hi! I'm YuGen AI — ask me anything about anime. Try: *'recommend me something like Death Note'* or *'what's a good action anime from the 2000s?'*";

const SUGGESTIONS = ["Best 90s anime?", "Similar to Attack on Titan", "Top romance picks", "Hidden gems?"];

function TypingDots() {
  return (
    <div className="chat-row bot">
      <div className="chat-avatar bot-avatar">Y</div>
      <div className="chat-bubble bot-bubble typing-bubble">
        <span className="dot" /><span className="dot" /><span className="dot" />
      </div>
    </div>
  );
}

function parseMarkdown(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/\n/g, "<br/>");
}

export default function Chatbot() {
  // "closed" | "minimized" | "open"
  const [state, setState]       = useState("closed");
  const [input, setInput]       = useState("");
  const [messages, setMessages] = useState([{ from: "bot", text: WELCOME }]);
  const [threadId, setThreadId] = useState(() => localStorage.getItem("chat_thread_id") || "");
  const [token, setToken]       = useState(() => localStorage.getItem("auth_token") || "");
  const [loading, setLoading]   = useState(false);
  const listRef  = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages, state, loading]);

  useEffect(() => {
    if (state === "open" && inputRef.current) inputRef.current.focus();
  }, [state]);

  useEffect(() => {
    function sync() {
      setToken(localStorage.getItem("auth_token") || "");
      setThreadId(localStorage.getItem("chat_thread_id") || "");
    }
    window.addEventListener("auth-changed", sync);
    return () => window.removeEventListener("auth-changed", sync);
  }, []);

  async function handleSend(e) {
    e?.preventDefault();
    const text = input.trim();
    if (!text || loading) return;
    setMessages((p) => [...p, { from: "user", text }]);
    setInput("");
    setLoading(true);
    try {
      const data = await sendChatMessage(text, threadId || null, token);
      if (data.thread_id) {
        setThreadId(data.thread_id);
        localStorage.setItem("chat_thread_id", data.thread_id);
      }
      setMessages((p) => [...p, { from: "bot", text: data.reply }]);
    } catch (err) {
      setMessages((p) => [...p, { from: "bot", text: err.message || "Chatbot unavailable." }]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  }

  function handleReset() {
    setMessages([{ from: "bot", text: WELCOME }]);
    setThreadId("");
    localStorage.removeItem("chat_thread_id");
  }

  const isLoggedIn = Boolean(token);
  const unread = state === "closed" && messages.length > 1;

  return (
    <div className="chatbot-root">
      {/* ── Minimised slot (pill above FAB) ── */}
      {state === "minimized" && (
        <button className="chatbot-minimized-slot" onClick={() => setState("open")}>
          <div className="chatbot-avatar mini-avatar">Y</div>
          <span>YuGen AI</span>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <polyline points="18 15 12 9 6 15" />
          </svg>
        </button>
      )}

      {/* ── Full chat panel ── */}
      <div className={`chatbot-panel${state === "open" ? " panel-open" : ""}`}>
        <div className="chatbot-header">
          <div className="chatbot-header-left">
            <div className="chatbot-avatar">Y</div>
            <div>
              <div className="chatbot-title">YuGen Assistant</div>
              <div className="chatbot-sub">
                {isLoggedIn
                  ? <><span className="status-dot active" /> Personalised mode</>
                  : <><span className="status-dot" /> Guest · login for personal picks</>
                }
              </div>
            </div>
          </div>
          <div className="chatbot-header-actions">
            <button className="icon-btn" onClick={handleReset} title="New chat">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="1 4 1 10 7 10" />
                <path d="M3.51 15a9 9 0 1 0 .49-3.51" />
              </svg>
            </button>
            <button className="icon-btn" onClick={() => setState("minimized")} title="Minimise">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="5" y1="12" x2="19" y2="12" />
              </svg>
            </button>
            <button className="icon-btn" onClick={() => setState("closed")} title="Close">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        </div>

        <div className="chatbot-body" ref={listRef}>
          {messages.map((msg, i) =>
            msg.from === "user" ? (
              <div key={i} className="chat-row user">
                <div className="chat-bubble user-bubble">{msg.text}</div>
                <div className="chat-avatar user-avatar">U</div>
              </div>
            ) : (
              <div key={i} className="chat-row bot">
                <div className="chat-avatar bot-avatar">Y</div>
                <div className="chat-bubble bot-bubble"
                  dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.text) }} />
              </div>
            )
          )}
          {loading && <TypingDots />}
        </div>

        {messages.length === 1 && (
          <div className="chat-suggestions">
            {SUGGESTIONS.map((s) => (
              <button key={s} className="suggestion-chip"
                onClick={() => { setInput(s); inputRef.current?.focus(); }}>
                {s}
              </button>
            ))}
          </div>
        )}

        <form className="chatbot-input-row" onSubmit={handleSend}>
          <textarea
            ref={inputRef}
            className="chatbot-textarea"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about anime…"
            rows={1}
            disabled={loading}
          />
          <button className="send-btn" type="submit" disabled={loading || !input.trim()}>
            <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
            </svg>
          </button>
        </form>
        <div className="chat-hint">Enter to send · Shift+Enter for newline</div>
      </div>

      {/* ── FAB ── */}
      {state !== "open" && state !== "minimized" && (
        <button className="chatbot-fab" onClick={() => setState("open")} title="Open AI chat">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
          {unread && <span className="fab-unread" />}
        </button>
      )}
    </div>
  );
}
