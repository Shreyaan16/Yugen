import os
import pandas as pd
from recommender.src.constants import *
from recommender.src.entity.config_entity import DataPreprocessingConfig
from recommender.src.entity.artifact_entity import DataPreprocessingArtifact
from recommender.src.utils import *


class DataPreprocessing:
    def __init__(self, data_preprocessing_config: DataPreprocessingConfig):
        self.config = data_preprocessing_config
        os.makedirs(self.config.data_preprocessing_dir, exist_ok=True)

        self.bins_cfg = read_yaml(FEATURE_BINS_CONFIG)

    def _load_tables(self) -> dict:
        raw_dir = self.config.raw_data_dir
        names = [
            "anime",
            "anime_genres",
            "anime_producers",
            "anime_studios",
            "genres",
            "producers",
            "studios",
            "sources",
            "ratings",
            "users",
            "user_anime_ratings",
        ]
        tables = {}
        for name in names:
            path = os.path.join(raw_dir, f"{name}.csv")
            tables[name] = read_df(path)
        return tables

    def _build_anime_df(self, t: dict) -> pd.DataFrame:
        anime_drop = get_cols_to_drop("preprocessing", "anime_df")
        df = drop_unnecessary_columns(t["anime"], anime_drop)

        genres_agg = (
            t["anime_genres"]
            .merge(t["genres"].rename(columns={"id": "genre_id"}), on="genre_id")
            .groupby("anime_id")["genre_name"]
            .apply(list)
            .reset_index(name="genres")
        )
        producers_agg = (
            t["anime_producers"]
            .merge(t["producers"].rename(columns={"id": "producer_id"}), on="producer_id")
            .groupby("anime_id")["producer_name"]
            .apply(list)
            .reset_index(name="producers")
        )
        studios_agg = (
            t["anime_studios"]
            .merge(t["studios"].rename(columns={"id": "studio_id"}), on="studio_id")
            .groupby("anime_id")["studio_name"]
            .apply(list)
            .reset_index(name="studios")
        )

        final_df = (
            df.rename(columns={"id": "anime_id"})
            .merge(
                t["sources"].rename(columns={"id": "source_id", "source_name": "source"}),
                on="source_id",
                how="left",
            )
            .merge(genres_agg, on="anime_id", how="left")
            .merge(producers_agg, on="anime_id", how="left")
            .merge(studios_agg, on="anime_id", how="left")
            .drop(columns=["source_id"])
        )

        for col in ["genres", "producers", "studios"]:
            final_df[col] = final_df[col].apply(
                lambda lst: [x for x in lst if x != "Unknown"] if isinstance(lst, list) else []
            )
        final_df["source"] = final_df["source"].replace("Unknown", "")

        era_cfg = self.bins_cfg["era"]
        era_mapping = dict(zip(era_cfg["labels"], era_cfg["text_labels"]))
        final_df["era"] = final_df["era"].map(era_mapping).fillna(final_df["era"])

        return final_df

    def _build_users_df(self, t: dict) -> pd.DataFrame:
        users_drop = get_cols_to_drop("preprocessing", "users_df")
        return drop_unnecessary_columns(t["users"], users_drop)

    def run(self) -> DataPreprocessingArtifact:
        tables = self._load_tables()

        anime_df = self._build_anime_df(tables)
        users_df = self._build_users_df(tables)
        ratings_df = tables["user_anime_ratings"]

        out_dir = self.config.data_preprocessing_dir
        save_df(anime_df, os.path.join(out_dir, "anime.csv"))
        save_df(users_df, os.path.join(out_dir, "users.csv"))
        save_df(ratings_df, os.path.join(out_dir, "user_anime_ratings.csv"))

        return DataPreprocessingArtifact(
            data_preprocessing_dir=self.config.data_preprocessing_dir,
            raw_data_dir=self.config.raw_data_dir
        )
