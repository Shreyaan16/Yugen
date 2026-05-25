import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { fetchAnimeDetail } from "../api.js";
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
  const { id } = useParams();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("details");

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");

    fetchAnimeDetail(id)
      .then((res) => {
        if (!active) return;
        setData(res);
      })
      .catch((err) => {
        if (!active) return;
        setError(err.message || "Failed to load anime details.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [id]);

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
              <span key={g} className="chip">
                {g}
              </span>
            ))}
          </div>
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
