import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api.js";

export default function ForYou() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    api
      .recommendForMe()
      .then(setData)
      .catch((e) => {
        if (e.message?.toLowerCase().includes("credentials")) {
          navigate("/login");
          return;
        }
        setErr(e.message);
      });
  }, [navigate]);

  if (err) return <div className="alert alert-danger">{err}</div>;
  if (!data) return <div className="text-body-secondary">Loading...</div>;

  return (
    <div>
      <div className="d-flex justify-content-between align-items-end mb-3">
        <h1 className="h3 mb-0">For you</h1>
        <span className="badge text-bg-info">strategy: {data.strategy}</span>
      </div>
      {data.strategy === "popularity" && (
        <p className="text-body-secondary small">
          You don't have a trained user vector yet — showing popular anime. Rate a
          few titles and check back after the next model retrain for personalised
          picks.
        </p>
      )}

      <div className="row g-3">
        {data.items.map((a) => (
          <div className="col-sm-6 col-md-4 col-lg-3" key={a.id}>
            <Link
              to={`/anime/${a.id}`}
              className="text-decoration-none text-reset"
            >
              <div className="card h-100">
                <div className="card-body">
                  <h6 className="card-title mb-1">{a.title}</h6>
                  <p className="card-text small text-body-secondary mb-1">
                    {a.era || "—"} &middot; {a.rating || "—"}
                  </p>
                  {a.hybrid_score !== undefined && (
                    <p className="card-text small mb-1">
                      Hybrid score: {a.hybrid_score.toFixed(3)}
                    </p>
                  )}
                  <p className="card-text small mb-0">
                    ⭐ {a.mean_rating ? a.mean_rating.toFixed(2) : "—"}
                    {a.num_ratings
                      ? ` (${a.num_ratings.toLocaleString()})`
                      : ""}
                  </p>
                </div>
              </div>
            </Link>
          </div>
        ))}
      </div>
    </div>
  );
}
