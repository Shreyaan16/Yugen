import { useEffect, useRef, useState } from "react";
import { fetchAnimeList } from "../api.js";
import AnimeCard from "../components/AnimeCard.jsx";
import logoUrl from "../../logo/YuGen-Logo.png";

const PAGE_SIZE = 24;

export default function Home() {
  const [page, setPage]     = useState(1);
  const [search, setSearch] = useState("");
  const [genre, setGenre]   = useState("");
  const [anime, setAnime]   = useState([]);
  const [total, setTotal]   = useState(0);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [error, setError]   = useState("");
  const loaderRef = useRef(null);

  // Reset list when filters change
  useEffect(() => {
    setAnime([]);
    setPage(1);
    setHasMore(true);
  }, [search, genre]);

  // Fetch on page / filter change
  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");

    fetchAnimeList({ page, limit: PAGE_SIZE, search, genre })
      .then((res) => {
        if (!active) return;
        setTotal(res.total);
        setAnime((prev) => (page === 1 ? res.anime : [...prev, ...res.anime]));
        setHasMore(page * PAGE_SIZE < res.total);
      })
      .catch((err) => {
        if (!active) return;
        setError(err.message || "Failed to load anime.");
      })
      .finally(() => { if (active) setLoading(false); });

    return () => { active = false; };
  }, [page, search, genre]);

  // IntersectionObserver — load next page when sentinel enters viewport
  useEffect(() => {
    const el = loaderRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting && hasMore && !loading) setPage((p) => p + 1); },
      { rootMargin: "200px" }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [hasMore, loading]);

  function scrollToCatalog() {
    document.getElementById("catalog")?.scrollIntoView({ behavior: "smooth" });
  }

  return (
    <main>
      {/* ── Full-screen landing hero ─────────────────────────────── */}
      <section className="landing-hero">
        <div className="landing-glow g1" />
        <div className="landing-glow g2" />
        <div className="landing-glow g3" />

        <div className="landing-content">
          <div className="landing-logo-wrap">
            <img src={logoUrl} alt="YuGen" className="landing-logo" />
            <div className="landing-logo-ring" />
          </div>
          <h1 className="landing-title">YuGen</h1>
          <p className="landing-tagline">Your anime discovery engine</p>
          <p className="landing-desc">
            Browse{total ? ` ${total.toLocaleString()}+` : " thousands of"} titles
            and get intelligent recommendations tailored to your taste.
          </p>
          <button className="landing-cta" onClick={scrollToCatalog}>
            Explore the catalog
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>
        </div>

        <button className="scroll-indicator" onClick={scrollToCatalog} aria-label="Scroll down">
          <span className="scroll-mouse"><span className="scroll-wheel" /></span>
        </button>
      </section>

      {/* ── Catalog section ──────────────────────────────────────── */}
      <section className="catalog-section" id="catalog">
        {/* Search / filter bar */}
        <div className="catalog-bar">
          <div className="catalog-search-wrap">
            <div className="search-field">
              <svg className="search-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              <input
                className="search-input"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search titles…"
              />
              {search && <button className="search-clear" onClick={() => setSearch("")}>×</button>}
            </div>

            <div className="search-field">
              <svg className="search-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
              </svg>
              <input
                className="search-input"
                value={genre}
                onChange={(e) => setGenre(e.target.value)}
                placeholder="Filter by genre…"
              />
              {genre && <button className="search-clear" onClick={() => setGenre("")}>×</button>}
            </div>
          </div>

          {total > 0 && (
            <div className="catalog-count">
              <span className="catalog-count-num">{total.toLocaleString()}</span>
              <span className="catalog-count-label">titles</span>
            </div>
          )}
        </div>

        {/* Cards */}
        {error ? (
          <div className="panel error">{error}</div>
        ) : (
          <>
            <div className="card-grid">
              {anime.map((a) => (
                <AnimeCard key={a.anime_id} anime={a} />
              ))}
            </div>

            {/* Infinite scroll sentinel */}
            <div ref={loaderRef} className="scroll-sentinel">
              {loading && (
                <div className="scroll-loader">
                  <span /><span /><span />
                </div>
              )}
              {!hasMore && anime.length > 0 && (
                <p className="scroll-end">✦ You've seen all {total.toLocaleString()} titles ✦</p>
              )}
            </div>
          </>
        )}
      </section>
    </main>
  );
}
