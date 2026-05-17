import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api.js";
import { setToken } from "../auth.js";

export default function Register() {
  const [genres, setGenres] = useState([]);
  const [form, setForm] = useState({ username: "", email: "", password: "" });
  const [selected, setSelected] = useState(new Set());
  const [recs, setRecs] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    api.genres().then((r) => setGenres(r.items)).catch((e) => setErr(e.message));
  }, []);

  function toggle(id) {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelected(next);
  }

  function update(field, value) {
    setForm((f) => ({ ...f, [field]: value }));
  }

  async function onSubmit(e) {
    e.preventDefault();
    setErr(null);
    if (selected.size === 0) {
      setErr("Pick at least one favourite genre.");
      return;
    }
    setBusy(true);
    try {
      const res = await api.register({
        ...form,
        favourite_genres: [...selected],
      });
      setRecs(res.recommended_anime || []);
      // auto-login
      const { access_token } = await api.login({
        email: form.email,
        password: form.password,
      });
      setToken(access_token);
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  if (recs) {
    return (
      <div>
        <h1 className="h3 mb-3">Welcome to YuGen!</h1>
        <p className="text-body-secondary">
          Based on your favourite genres, you might like these:
        </p>
        <div className="row g-3">
          {recs.map((a) => (
            <div className="col-md-6 col-lg-4" key={a.id}>
              <div className="card h-100">
                <div className="card-body">
                  <h6 className="card-title mb-1">{a.title}</h6>
                  <p className="card-text small text-body-secondary mb-1">
                    {a.era} &middot; {a.rating}
                  </p>
                  <p className="card-text small mb-0">
                    Popularity score: {a.popularity_score?.toFixed?.(2) ?? "—"}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4">
          <button className="btn btn-primary" onClick={() => navigate("/for-you")}>
            Go to For You
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="row justify-content-center">
      <div className="col-md-8 col-lg-6">
        <h1 className="h3 mb-4">Register</h1>
        <form onSubmit={onSubmit}>
          <div className="mb-3">
            <label className="form-label">Username</label>
            <input
              className="form-control"
              value={form.username}
              onChange={(e) => update("username", e.target.value)}
              required
            />
          </div>
          <div className="mb-3">
            <label className="form-label">Email</label>
            <input
              type="email"
              className="form-control"
              value={form.email}
              onChange={(e) => update("email", e.target.value)}
              required
            />
          </div>
          <div className="mb-3">
            <label className="form-label">Password</label>
            <input
              type="password"
              className="form-control"
              value={form.password}
              onChange={(e) => update("password", e.target.value)}
              required
            />
          </div>
          <div className="mb-3">
            <label className="form-label">Favourite genres ({selected.size})</label>
            <div className="d-flex flex-wrap gap-2">
              {genres.map((g) => {
                const on = selected.has(g.id);
                return (
                  <button
                    type="button"
                    key={g.id}
                    onClick={() => toggle(g.id)}
                    className={
                      "btn btn-sm " +
                      (on ? "btn-primary" : "btn-outline-secondary")
                    }
                  >
                    {g.genre_name}
                  </button>
                );
              })}
            </div>
          </div>
          {err && <div className="alert alert-danger py-2">{err}</div>}
          <button className="btn btn-primary w-100" type="submit" disabled={busy}>
            {busy ? "Registering..." : "Create account"}
          </button>
        </form>
        <p className="text-body-secondary small mt-3">
          Already have an account? <Link to="/login">Log in</Link>
        </p>
      </div>
    </div>
  );
}
