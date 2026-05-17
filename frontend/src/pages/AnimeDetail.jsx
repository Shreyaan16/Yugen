import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api.js";
import { isLoggedIn } from "../auth.js";

export default function AnimeDetail() {
  const { id } = useParams();
  const [detail, setDetail] = useState(null);
  const [similar, setSimilar] = useState([]);
  const [err, setErr] = useState(null);

  const [rating, setRating] = useState(8);
  const [rateMsg, setRateMsg] = useState(null);
  const [rateErr, setRateErr] = useState(null);
  const [rateBusy, setRateBusy] = useState(false);

  const loggedIn = isLoggedIn();

  useEffect(() => {
    setDetail(null);
    setSimilar([]);
    setErr(null);
    setRateMsg(null);
    setRateErr(null);

    api
      .animeDetail(id)
      .then(setDetail)
      .catch((e) => setErr(e.message));

    api
      .similarAnime(id, 10)
      .then((r) => setSimilar(r.items || []))
      .catch(() => {});
  }, [id]);

  async function submitRating() {
    setRateMsg(null);
    setRateErr(null);
    setRateBusy(true);
    try {
      const r = await api.rate(Number(id), Number(rating));
      setRateMsg(r.message);
    } catch (e) {
      setRateErr(e.message);
    } finally {
      setRateBusy(false);
    }
  }

  if (err) return <div className="alert alert-danger">{err}</div>;
  if (!detail) return <div className="text-body-secondary">Loading...</div>;

  return (
    <div>
      <p className="mb-2">
        <Link to="/anime" className="text-body-secondary small">
          ← Back to browse
        </Link>
      </p>
      <h1 className="h2 mb-1">{detail.title}</h1>
      <p className="text-body-secondary">
        {detail.era || "—"} &middot; {detail.rating || "—"} &middot;{" "}
        {detail.source || "—"}
      </p>

      <div className="d-flex flex-wrap gap-1 mb-3">
        {(detail.genres || []).map((g) => (
          <span className="badge text-bg-secondary" key={g}>
            {g}
          </span>
        ))}
      </div>

      <div className="row mb-4">
        <div className="col-md-8">
          <h6>Synopsis</h6>
          <p className="text-body-secondary">{detail.synopsis || "No synopsis."}</p>
        </div>
        <div className="col-md-4">
          <div className="card">
            <div className="card-body">
              <h6 className="card-title">Stats</h6>
              <p className="card-text small mb-1">
                Mean rating:{" "}
                {detail.mean_rating ? detail.mean_rating.toFixed(2) : "—"}
              </p>
              <p className="card-text small mb-1">
                Ratings count:{" "}
                {detail.num_ratings ? detail.num_ratings.toLocaleString() : "—"}
              </p>
              <p className="card-text small mb-1">
                Popularity:{" "}
                {detail.popularity_score
                  ? detail.popularity_score.toFixed(2)
                  : "—"}
              </p>
              <p className="card-text small mb-0">
                Episodes: {detail.ep_bin || "—"} &middot; Duration:{" "}
                {detail.dur_bin || "—"}
              </p>
            </div>
          </div>
        </div>
      </div>

      {(detail.studios?.length || 0) > 0 && (
        <p className="small">
          <span className="text-body-secondary">Studios: </span>
          {detail.studios.join(", ")}
        </p>
      )}
      {(detail.producers?.length || 0) > 0 && (
        <p className="small">
          <span className="text-body-secondary">Producers: </span>
          {detail.producers.join(", ")}
        </p>
      )}

      <hr />

      <section className="mb-4">
        <h5>Rate this anime</h5>
        {loggedIn ? (
          <div className="d-flex gap-2 align-items-center flex-wrap">
            <input
              type="range"
              min="1"
              max="10"
              value={rating}
              onChange={(e) => setRating(e.target.value)}
              className="form-range"
              style={{ maxWidth: 240 }}
            />
            <span className="badge text-bg-primary">{rating}/10</span>
            <button
              className="btn btn-primary btn-sm"
              disabled={rateBusy}
              onClick={submitRating}
            >
              {rateBusy ? "Saving..." : "Submit rating"}
            </button>
            {rateMsg && <span className="text-success small">{rateMsg}</span>}
            {rateErr && <span className="text-danger small">{rateErr}</span>}
          </div>
        ) : (
          <p className="text-body-secondary small">
            <Link to="/login">Log in</Link> to rate this anime.
          </p>
        )}
      </section>

      <section>
        <h5>Similar anime</h5>
        {similar.length === 0 ? (
          <p className="text-body-secondary small">No similar anime available.</p>
        ) : (
          <div className="row g-3">
            {similar.map((s) => (
              <div className="col-sm-6 col-md-4 col-lg-3" key={s.id}>
                <Link
                  to={`/anime/${s.id}`}
                  className="text-decoration-none text-reset"
                >
                  <div className="card h-100">
                    <div className="card-body">
                      <h6 className="card-title mb-1">{s.title}</h6>
                      <p className="card-text small text-body-secondary mb-1">
                        sim: {s.similarity?.toFixed?.(3) ?? "—"}
                      </p>
                      <p className="card-text small mb-0">
                        ⭐{" "}
                        {s.mean_rating ? s.mean_rating.toFixed(2) : "—"}
                      </p>
                    </div>
                  </div>
                </Link>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
