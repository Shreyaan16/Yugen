import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { fetchMyProfile, fetchMyRatings } from "../api.js";

const STAR_MAP = { 1: "★", 2: "★★", 3: "★★★", 4: "★★★★", 5: "★★★★★",
                   6: "★★★★★★", 7: "★★★★★★★", 8: "★★★★★★★★", 9: "★★★★★★★★★", 10: "★★★★★★★★★★" };

function StarRating({ value }) {
  const pct = (value / 10) * 100;
  return (
    <div className="star-rating" title={`${value}/10`}>
      <div className="stars-bg">{"★★★★★★★★★★"}</div>
      <div className="stars-fg" style={{ width: `${pct}%` }}>{"★★★★★★★★★★"}</div>
    </div>
  );
}

export default function Profile() {
  const navigate = useNavigate();
  const token    = localStorage.getItem("auth_token") || "";

  const [profile,  setProfile]  = useState(null);
  const [ratings,  setRatings]  = useState([]);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState("");
  const [search,   setSearch]   = useState("");
  const [sortKey,  setSortKey]  = useState("rating_desc");

  useEffect(() => {
    if (!token) { navigate("/"); return; }

    Promise.all([fetchMyProfile(token), fetchMyRatings(token)])
      .then(([prof, rats]) => { setProfile(prof); setRatings(rats); })
      .catch((err) => setError(err.message || "Failed to load profile."))
      .finally(() => setLoading(false));
  }, [token]);

  if (loading) return <main className="page"><div className="panel loading-panel">Loading profile…</div></main>;
  if (error)   return <main className="page"><div className="panel error">{error}</div></main>;
  if (!profile) return null;

  // filter + sort
  const filtered = ratings
    .filter((r) => r.title.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      if (sortKey === "rating_desc") return b.rating - a.rating;
      if (sortKey === "rating_asc")  return a.rating - b.rating;
      if (sortKey === "title_asc")   return a.title.localeCompare(b.title);
      return b.title.localeCompare(a.title);
    });

  const avgRating = ratings.length
    ? (ratings.reduce((s, r) => s + r.rating, 0) / ratings.length).toFixed(1)
    : "–";

  // genre breakdown
  const genreCounts = {};
  ratings.forEach((r) => (r.genres || []).forEach((g) => { genreCounts[g] = (genreCounts[g] || 0) + 1; }));
  const topGenres = Object.entries(genreCounts).sort((a, b) => b[1] - a[1]).slice(0, 8);

  return (
    <main className="page">
      {/* ── Profile hero ── */}
      <section className="profile-hero">
        <div className="profile-avatar-wrap">
          <div className="profile-avatar">{profile.username?.[0]?.toUpperCase() || "?"}</div>
          <div className="profile-online-dot" />
        </div>
        <div className="profile-info">
          <div className="eyebrow">Your profile</div>
          <h1 className="profile-name">{profile.username}</h1>
          <div className="profile-email">{profile.email}</div>
          <div className="profile-badges">
            <span className="badge-pill">ID #{profile.user_id}</span>
            <span className="badge-pill">{ratings.length} rated</span>
            <span className="badge-pill accent-pill">Avg {avgRating} / 10</span>
          </div>
        </div>
      </section>

      {/* ── Genre taste ── */}
      {topGenres.length > 0 && (
        <section className="panel profile-genres-panel">
          <div className="panel-title" style={{ marginBottom: 14 }}>Taste profile</div>
          <div className="genre-bar-list">
            {topGenres.map(([genre, count]) => (
              <div key={genre} className="genre-bar-row">
                <span className="genre-bar-label">{genre}</span>
                <div className="genre-bar-track">
                  <div
                    className="genre-bar-fill"
                    style={{ width: `${Math.round((count / topGenres[0][1]) * 100)}%` }}
                  />
                </div>
                <span className="genre-bar-count">{count}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Rated anime ── */}
      <section className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">Rated anime</div>
            <div className="panel-sub">{filtered.length} of {ratings.length} shown</div>
          </div>
          <div className="ratings-controls">
            <input
              className="ratings-search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Filter by title…"
            />
            <select
              className="ratings-sort"
              value={sortKey}
              onChange={(e) => setSortKey(e.target.value)}
            >
              <option value="rating_desc">Rating ↓</option>
              <option value="rating_asc">Rating ↑</option>
              <option value="title_asc">Title A–Z</option>
              <option value="title_desc">Title Z–A</option>
            </select>
          </div>
        </div>

        {filtered.length === 0 ? (
          <div className="empty-state">No rated anime found.</div>
        ) : (
          <div className="ratings-list">
            {filtered.map((item) => (
              <Link to={`/anime/${item.anime_id}`} key={item.anime_id} className="rating-row">
                <div className="rating-score-badge">{item.rating}</div>
                <div className="rating-info">
                  <div className="rating-title">{item.title}</div>
                  <div className="rating-genres">
                    {(item.genres || []).slice(0, 4).map((g) => (
                      <span key={g} className="chip">{g}</span>
                    ))}
                  </div>
                </div>
                <div className="rating-stars">
                  <StarRating value={item.rating} />
                  <span className="rating-num">{item.rating}/10</span>
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
