import { Link } from "react-router-dom";

export default function AnimeCard({ anime }) {
  return (
    <Link to={`/anime/${anime.anime_id}`} className="card">
      <div className="card-title">{anime.title}</div>
      <div className="card-genres">
        {(anime.genres || []).slice(0, 4).map((genre) => (
          <span key={genre} className="chip">
            {genre}
          </span>
        ))}
      </div>
    </Link>
  );
}
