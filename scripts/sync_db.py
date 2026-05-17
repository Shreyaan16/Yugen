"""Sync Postgres with the full anime catalogue (train+val+test) and per-anime rating aggregates.

Fixes a previous ingestion bug where only train-split anime/genres/studios/producers
were pushed. Re-runs are safe: missing rows are inserted, existing rows are left alone.
Aggregates from rating_complete.csv are stored in the `ratings` table (one row per anime).
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
PREP_DIR = "artifacts/data_preprocessing"
RAW_RATINGS = "artifacts/data_ingestion/rating_complete.csv"
POPULARITY_PRIOR_VOTES = 50  # Bayesian smoothing: anime need this many ratings to lean on their own mean


def load_full_anime_df() -> pd.DataFrame:
    frames = [
        pd.read_csv(os.path.join(PREP_DIR, split, "data.csv"))
        for split in ("train_data", "val_data", "test_data")
    ]
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["MAL_ID"]).reset_index(drop=True)


def explode_multi(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df[["MAL_ID", col]].copy()
    out[col] = out[col].fillna("Unknown").astype(str)
    out = out.assign(name=out[col].str.split(",")).explode("name")
    out["name"] = out["name"].str.strip()
    out = out[out["name"] != ""]
    return out.drop(columns=[col]).rename(columns={"MAL_ID": "anime_id"}).drop_duplicates().reset_index(drop=True)


def upsert_lookup(conn, table: str, name_col: str, names: set[str]) -> dict[str, int]:
    existing = {
        row[1]: row[0]
        for row in conn.execute(text(f"SELECT id, {name_col} FROM {table}")).fetchall()
    }
    missing = sorted(n for n in names if n not in existing)
    if missing:
        conn.execute(
            text(f"INSERT INTO {table} ({name_col}) VALUES (:n)"),
            [{"n": n} for n in missing],
        )
        existing = {
            row[1]: row[0]
            for row in conn.execute(text(f"SELECT id, {name_col} FROM {table}")).fetchall()
        }
    return existing


def sync_anime_catalogue(engine) -> None:
    df = load_full_anime_df()
    print(f"[catalogue] full set size: {len(df)}", flush=True)

    with engine.begin() as conn:
        existing_anime_ids = {
            row[0] for row in conn.execute(text("SELECT id FROM anime")).fetchall()
        }
        new_df = df[~df["MAL_ID"].isin(existing_anime_ids)]
        print(
            f"[catalogue] anime already in DB: {len(existing_anime_ids)}, new: {len(new_df)}",
            flush=True,
        )

        all_genres: set[str] = set()
        all_studios: set[str] = set()
        all_producers: set[str] = set()
        all_sources: set[str] = set()
        for col, bucket in [
            ("Genres", all_genres),
            ("Studios", all_studios),
            ("Producers", all_producers),
        ]:
            for v in df[col].dropna().astype(str):
                for piece in v.split(","):
                    p = piece.strip()
                    if p:
                        bucket.add(p)
        for v in df["Source"].dropna().astype(str):
            v = v.strip()
            if v:
                all_sources.add(v)
        for bucket in (all_genres, all_studios, all_producers, all_sources):
            bucket.add("Unknown")

        genres_map = upsert_lookup(conn, "genres", "genre_name", all_genres)
        studios_map = upsert_lookup(conn, "studios", "studio_name", all_studios)
        producers_map = upsert_lookup(conn, "producers", "producer_name", all_producers)
        sources_map = upsert_lookup(conn, "sources", "source_name", all_sources)
        print(
            f"[lookups] genres={len(genres_map)} studios={len(studios_map)} "
            f"producers={len(producers_map)} sources={len(sources_map)}",
            flush=True,
        )

        if not new_df.empty:
            rows = []
            for _, r in new_df.iterrows():
                src = str(r["Source"]).strip() if pd.notna(r["Source"]) else "Unknown"
                rows.append(
                    {
                        "id": int(r["MAL_ID"]),
                        "title": (str(r["Title"])[:300]) if pd.notna(r["Title"]) else None,
                        "rating": (str(r["Rating"])[:50]) if pd.notna(r["Rating"]) else None,
                        "synopsis": str(r["synopsis"]) if pd.notna(r["synopsis"]) else None,
                        "ep_bin": str(r["Ep_bin"]) if pd.notna(r["Ep_bin"]) else None,
                        "dur_bin": str(r["Dur_bin"]) if pd.notna(r["Dur_bin"]) else None,
                        "era": str(r["Era"]) if pd.notna(r["Era"]) else None,
                        "source_id": sources_map.get(src, sources_map["Unknown"]),
                    }
                )
            conn.execute(
                text(
                    """
                    INSERT INTO anime (id, title, rating, synopsis, ep_bin, dur_bin, era, source_id)
                    VALUES (:id, :title, :rating, :synopsis, :ep_bin, :dur_bin, :era, :source_id)
                    ON CONFLICT (id) DO NOTHING
                    """
                ),
                rows,
            )
            print(f"[anime] inserted {len(rows)} rows", flush=True)

        catalogue_ids = {int(x) for x in df["MAL_ID"].tolist()}

        def sync_join(join_table: str, value_col: str, lookup_map: dict[str, int], src_col: str) -> None:
            pairs = explode_multi(df, src_col)
            pairs = pairs[pairs["anime_id"].isin(catalogue_ids)]
            pairs["value_id"] = pairs["name"].map(lookup_map)
            pairs = pairs.dropna(subset=["value_id"])
            pairs["value_id"] = pairs["value_id"].astype(int)

            existing = {
                (r[0], r[1])
                for r in conn.execute(
                    text(f"SELECT anime_id, {value_col} FROM {join_table}")
                ).fetchall()
            }
            new = [
                {"a": int(a), "v": int(v)}
                for a, v in zip(pairs["anime_id"], pairs["value_id"])
                if (int(a), int(v)) not in existing
            ]
            if new:
                conn.execute(
                    text(
                        f"INSERT INTO {join_table} (anime_id, {value_col}) "
                        f"VALUES (:a, :v) ON CONFLICT DO NOTHING"
                    ),
                    new,
                )
            print(
                f"[{join_table}] inserted {len(new)} new pairs (total source: {len(pairs)})",
                flush=True,
            )

        sync_join("anime_genres", "genre_id", genres_map, "Genres")
        sync_join("anime_studios", "studio_id", studios_map, "Studios")
        sync_join("anime_producers", "producer_id", producers_map, "Producers")


def sync_rating_aggregates(engine) -> None:
    print(f"[ratings] reading {RAW_RATINGS} ...", flush=True)
    ratings = pd.read_csv(
        RAW_RATINGS,
        dtype={"user_id": "int32", "anime_id": "int32", "rating": "int16"},
    )
    print(f"[ratings] raw rows: {len(ratings):,}", flush=True)

    agg = (
        ratings.groupby("anime_id")["rating"]
        .agg(num_ratings="count", mean_rating="mean")
        .reset_index()
    )
    global_mean = agg["mean_rating"].mean()
    m = POPULARITY_PRIOR_VOTES
    agg["popularity_score"] = (
        (agg["num_ratings"] / (agg["num_ratings"] + m)) * agg["mean_rating"]
        + (m / (agg["num_ratings"] + m)) * global_mean
    )
    print(
        f"[ratings] aggregated to {len(agg):,} anime (global mean={global_mean:.3f}, prior m={m})",
        flush=True,
    )

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS ratings (
                    anime_id INTEGER PRIMARY KEY REFERENCES anime(id) ON DELETE CASCADE,
                    num_ratings INTEGER NOT NULL,
                    mean_rating DOUBLE PRECISION NOT NULL,
                    popularity_score DOUBLE PRECISION NOT NULL
                )
                """
            )
        )
        anime_ids = {r[0] for r in conn.execute(text("SELECT id FROM anime")).fetchall()}
        before = len(agg)
        agg = agg[agg["anime_id"].isin(anime_ids)]
        print(
            f"[ratings] kept {len(agg):,} rows whose anime_id is in DB (dropped {before - len(agg):,})",
            flush=True,
        )

        conn.execute(text("TRUNCATE TABLE ratings"))
        rows = [
            {
                "anime_id": int(r.anime_id),
                "num_ratings": int(r.num_ratings),
                "mean_rating": float(r.mean_rating),
                "popularity_score": float(r.popularity_score),
            }
            for r in agg.itertuples(index=False)
        ]
        conn.execute(
            text(
                """
                INSERT INTO ratings (anime_id, num_ratings, mean_rating, popularity_score)
                VALUES (:anime_id, :num_ratings, :mean_rating, :popularity_score)
                """
            ),
            rows,
        )
        print(f"[ratings] inserted {len(rows):,} aggregate rows", flush=True)


def main() -> None:
    engine = create_engine(DATABASE_URL)
    sync_anime_catalogue(engine)
    sync_rating_aggregates(engine)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
