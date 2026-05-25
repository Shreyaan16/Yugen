/**
 * Smart pagination — always shows first/last, current ±2, and ellipsis gaps.
 * Never renders more than ~9 buttons regardless of total pages.
 */
export default function Pagination({ page, totalPages, onPageChange }) {
  if (totalPages <= 1) return null;

  function buildPages() {
    const pages = [];
    const add = (n) => pages.push(n);
    const dot = () => pages.push("…");

    add(1);
    if (page > 4) dot();

    const start = Math.max(2, page - 2);
    const end   = Math.min(totalPages - 1, page + 2);

    for (let p = start; p <= end; p++) add(p);

    if (page < totalPages - 3) dot();
    if (totalPages > 1) add(totalPages);

    // deduplicate (1 might appear twice if range starts at 2)
    return [...new Set(pages)];
  }

  const pages = buildPages();

  return (
    <div className="pagination">
      <button
        className="pag-arrow"
        onClick={() => onPageChange(1)}
        disabled={page === 1}
        title="First page"
      >
        «
      </button>
      <button
        className="pag-arrow"
        onClick={() => onPageChange(page - 1)}
        disabled={page === 1}
        title="Previous page"
      >
        ‹
      </button>

      <div className="page-list">
        {pages.map((p, i) =>
          p === "…" ? (
            <span key={`dot-${i}`} className="pag-dot">…</span>
          ) : (
            <button
              key={p}
              className={p === page ? "page-btn active" : "page-btn"}
              onClick={() => onPageChange(p)}
            >
              {p}
            </button>
          )
        )}
      </div>

      <button
        className="pag-arrow"
        onClick={() => onPageChange(page + 1)}
        disabled={page === totalPages}
        title="Next page"
      >
        ›
      </button>
      <button
        className="pag-arrow"
        onClick={() => onPageChange(totalPages)}
        disabled={page === totalPages}
        title="Last page"
      >
        »
      </button>
    </div>
  );
}
