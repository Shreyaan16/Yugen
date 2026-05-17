import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api.js";

const PAGE_SIZE = 24;

export default function AllAnime() {
  const [params, setParams] = useSearchParams();
  const page = Math.max(0, parseInt(params.get("page") || "0", 10));
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    setData(null);
    setErr(null);
    api
      .allAnime(PAGE_SIZE, page * PAGE_SIZE)
      .then(setData)
      .catch((e) => setErr(e.message));
  }, [page]);

  function goto(p) {
    setParams({ page: String(p) });
  }

  if (err) return <div className="alert alert-danger">{err}</div>;
  if (!data) return <div className="text-body-secondary">Loading...</div>;

  const totalPages = Math.ceil(data.total / data.limit);

  return (
    <div>
      <div className="d-flex justify-content-between align-items-end mb-3">
        <h1 className="h3 mb-0">Browse anime</h1>
        <span className="text-body-secondary small">
          {data.total.toLocaleString()} titles &middot; page {page + 1} / {totalPages}
        </span>
      </div>

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
                  <p className="card-text small text-body-secondary mb-2">
                    {a.era || "—"} &middot; {a.rating || "—"}
                  </p>
                  <div className="d-flex flex-wrap gap-1 mb-2">
                    {(a.genres || []).slice(0, 3).map((g) => (
                      <span className="badge text-bg-secondary" key={g}>
                        {g}
                      </span>
                    ))}
                  </div>
                  <div className="small text-body-secondary">
                    {a.mean_rating ? `⭐ ${a.mean_rating.toFixed(2)}` : "—"}
                    {a.num_ratings ? ` (${a.num_ratings.toLocaleString()})` : ""}
                  </div>
                </div>
              </div>
            </Link>
          </div>
        ))}
      </div>

      <nav className="mt-4 d-flex justify-content-center gap-2">
        <button
          className="btn btn-outline-secondary btn-sm"
          disabled={page === 0}
          onClick={() => goto(page - 1)}
        >
          ← Prev
        </button>
        <button
          className="btn btn-outline-secondary btn-sm"
          disabled={page + 1 >= totalPages}
          onClick={() => goto(page + 1)}
        >
          Next →
        </button>
      </nav>
    </div>
  );
}
