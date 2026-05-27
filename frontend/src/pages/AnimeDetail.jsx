import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { fetchAnimeDetail, fetchMyRatings, rateAnime } from "../api.js";
import SimilarAnime from "../components/SimilarAnime.jsx";

const PRIMARY_KEYS = [
  "title",
  "synopsis",
  "genres",
  "producers",
  "studios",
  "type",
  "source",
  "episodes",
  "duration",
  "rating",
  "score",
  "popularity",
  "rank",
  "members",
  "favorites",
  "watching",
  "completed",
  "start_date",
  "end_date",
  "season",
  "year"
];

export default function AnimeDetail() {
  const { id }  = useParams();
  const token   = localStorage.getItem("auth_token") || "";

  const [data,        setData]        = useState(null);
  const [loading,     setLoading]     = useState(true);
  const [error,       setError]       = useState("");
  const [tab,         setTab]         = useState("details");

  // Rating widget state
  const [userRating,  setUserRating]  = useState(null);   // saved rating (1-10 | null)
  const [hovered,     setHovered]     = useState(null);   // which number is hovered
  const [rateLoading, setRateLoading] = useState(false);
  const [rateMsg,     setRateMsg]     = useState("");     // success / error feedback

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");

    const detailFetch = fetchAnimeDetail(id);
    const ratingFetch = token
      ? fetchMyRatings(token).then((rats) => {
          const match = rats.find((r) => String(r.anime_id) === String(id));
          return match ? match.rating : null;
        }).catch(() => null)
      : Promise.resolve(null);

    Promise.all([detailFetch, ratingFetch])
      .then(([detail, existingRating]) => {
        if (!active) return;
        setData(detail);
        setUserRating(existingRating);
      })
      .catch((err) => {
        if (!active) return;
        setError(err.message || "Failed to load anime details.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => { active = false; };
  }, [id, token]);

  async function handleRate(value) {
    if (rateLoading) return;
    setRateLoading(true);
    setRateMsg("");
    try {
      await rateAnime(id, value, token);
      setUserRating(value);
      setRateMsg(userRating ? "Rating updated!" : "Rating saved!");
      setTimeout(() => setRateMsg(""), 2500);
    } catch (err) {
      setRateMsg(err.message || "Failed to submit rating.");
    } finally {
      setRateLoading(false);
    }
  }

  const detailRows = useMemo(() => {
    if (!data) return [];
    return Object.entries(data).filter(([key]) => key !== "anime_id");
  }, [data]);

  const primaryRows = useMemo(() => {
    if (!data) return [];
    return PRIMARY_KEYS.filter((key) => key in data).map((key) => [key, data[key]]);
  }, [data]);

  if (loading) {
    return <main className="page"><div className="panel">Loading details...</div></main>;
  }

  if (error) {
    return <main className="page"><div className="panel error">{error}</div></main>;
  }

  if (!data) {
    return <main className="page"><div className="panel">Anime not found.</div></main>;
  }

  return (
    <main className="page">
      <section className="detail-hero">
        <div>
          <div className="eyebrow">Anime profile</div>
          <h1>{data.title || "Untitled"}</h1>
          <p>{data.synopsis || "No synopsis available."}</p>
          <div className="detail-chips">
            {(data.genres || []).map((g) => (
              <span key={g} className="chip">{g}</span>
            ))}
          </div>

          {/* ── Rating widget (logged-in users only) ── */}
          {token && (
            <div className="rate-section">
              <div className="rate-label">
                {userRating
                  ? <>Your rating: <span className="rate-current">{userRating}/10</span></>
                  : "Rate this anime"}
              </div>
              <div
                className="rate-bar"
                onMouseLeave={() => setHovered(null)}
              >
                {Array.from({ length: 10 }, (_, i) => i + 1).map((n) => {
                  const active  = n <= (hovered ?? userRating ?? 0);
                  return (
                    <button
                      key={n}
                      className={`rate-btn${active ? " rate-btn--on" : ""}`}
                      onMouseEnter={() => setHovered(n)}
                      onClick={() => handleRate(n)}
                      disabled={rateLoading}
                      title={`Rate ${n}/10`}
                    >
                      {n}
                    </button>
                  );
                })}
              </div>
              {rateMsg && (
                <div className={`rate-msg${rateMsg.includes("!") ? "" : " rate-msg--err"}`}>
                  {rateMsg}
                </div>
              )}
            </div>
          )}
        </div>
        <div className="detail-stats">
          <div className="stat-card">
            <div className="stat-label">Favorites</div>
            <div className="stat-value">{data.favorites ?? "-"}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Watching</div>
            <div className="stat-value">{data.watching ?? "-"}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Completed</div>
            <div className="stat-value">{data.completed ?? "-"}</div>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="tab-row">
          <button
            className={tab === "details" ? "tab-btn active" : "tab-btn"}
            onClick={() => setTab("details")}
          >
            Details
          </button>
          <button
            className={tab === "similar" ? "tab-btn active" : "tab-btn"}
            onClick={() => setTab("similar")}
          >
            Similar anime
          </button>
        </div>

        {tab === "similar" ? (
          <SimilarAnime animeId={id} />
        ) : (
          <div className="detail-grid">
            {primaryRows.map(([key, value]) => (
              <div className="detail-row" key={key}>
                <div className="detail-key">{key.replace(/_/g, " ")}</div>
                <div className="detail-value">
                  {Array.isArray(value) ? value.join(", ") : String(value ?? "-")}
                </div>
              </div>
            ))}
            <div className="detail-divider">All fields</div>
            {detailRows.map(([key, value]) => (
              <div className="detail-row" key={key}>
                <div className="detail-key">{key.replace(/_/g, " ")}</div>
                <div className="detail-value">
                  {Array.isArray(value) ? value.join(", ") : String(value ?? "-")}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
