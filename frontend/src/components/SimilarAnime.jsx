import { useEffect, useState } from "react";
import { fetchSimilarAnime } from "../api.js";
import AnimeCard from "./AnimeCard.jsx";
import Pagination from "./Pagination.jsx";

const PAGE_SIZE = 8;

export default function SimilarAnime({ animeId }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [items, setItems] = useState([]);
  const [page, setPage] = useState(1);
  const [token, setToken] = useState(
    () => localStorage.getItem("auth_token") || ""
  );

  useEffect(() => {
    function syncAuth() {
      setToken(localStorage.getItem("auth_token") || "");
    }

    window.addEventListener("auth-changed", syncAuth);
    return () => window.removeEventListener("auth-changed", syncAuth);
  }, []);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    setPage(1);

    fetchSimilarAnime(animeId, token)
      .then((data) => {
        if (!active) return;
        setItems(data || []);
      })
      .catch((err) => {
        if (!active) return;
        setError(err.message || "Failed to load similar anime.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [animeId, token]);

  const totalPages = Math.max(1, Math.ceil(items.length / PAGE_SIZE));
  const pageItems = items.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  if (loading) {
    return <div className="panel">Loading similar anime...</div>;
  }

  if (error) {
    return <div className="panel error">{error}</div>;
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <div>
          <div className="panel-title">Similar picks</div>
          <div className="panel-sub">
            {token ? "Hybrid recommendations" : "Content-based similarity"}
          </div>
        </div>
        <div className="panel-count">{items.length} results</div>
      </div>
      <div className="card-grid">
        {pageItems.map((anime) => (
          <AnimeCard key={anime.anime_id} anime={anime} />
        ))}
      </div>
      <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />
    </div>
  );
}
