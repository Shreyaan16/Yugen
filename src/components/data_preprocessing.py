import os
import re
import numpy as np
import pandas as pd
from src.constants import *
from src.entity.config_entity import DataPreprocessingConfig
from src.entity.artifact_entity import DataPreprocessingArtifact
from src.utils import read_yaml, save_df, get_cols_to_drop

class DataPreprocessing:
    def __init__(self, data_preprocessing_config: DataPreprocessingConfig):
        self.config = data_preprocessing_config
        os.makedirs(self.config.data_preprocessing_dir, exist_ok=True)
        os.makedirs(self.config.train_data_dir, exist_ok=True)
        os.makedirs(self.config.test_data_dir, exist_ok=True)
        os.makedirs(self.config.val_data_dir, exist_ok=True)

        self.bins_cfg = read_yaml(FEATURE_BINS_CONFIG)

    @staticmethod
    def replace_unknown_with_nan(df: pd.DataFrame) -> pd.DataFrame:
        return df.replace(r"(?i)^unknown$", np.nan, regex=True)

    @staticmethod
    def extract_duration_minutes(duration):
        if pd.isna(duration):
            return None
        duration = str(duration).lower()
        hours = minutes = seconds = 0
        if m := re.search(r"(\d+)\s*hr", duration):
            hours = int(m.group(1))
        if m := re.search(r"(\d+)\s*min", duration):
            minutes = int(m.group(1))
        if m := re.search(r"(\d+)\s*sec", duration):
            seconds = int(m.group(1))
        return round(hours * 60 + minutes + seconds / 60, 2)

    def _clean_anime_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.replace_unknown_with_nan(df)
        df = df.drop(columns=[c for c in get_cols_to_drop("preprocessing", "anime_df") if c in df.columns])
        df["Episodes"] = pd.to_numeric(df["Episodes"], errors="coerce").astype("Int64")
        df["Duration"] = df["Duration"].apply(self.extract_duration_minutes)
        return df

    def _clean_synopsis_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.replace_unknown_with_nan(df)
        df = df.drop(columns=[c for c in get_cols_to_drop("preprocessing", "anime_with_synopsis_df") if c in df.columns])
        return df

    @staticmethod
    def _merge(anime_df: pd.DataFrame, synopsis_df: pd.DataFrame) -> pd.DataFrame:
        df = pd.merge(anime_df, synopsis_df, on="MAL_ID", how="left")
        df["Title"] = df["Name_x"].combine_first(df["Name_y"]).combine_first(df["English name"])

        def union_genres(row):
            g1, g2 = row["Genres_x"], row["Genres_y"]
            s1 = set(map(str.strip, g1.split(","))) if pd.notna(g1) else set()
            s2 = set(map(str.strip, g2.split(","))) if pd.notna(g2) else set()
            merged = sorted(s1 | s2)
            return ", ".join(merged) if merged else np.nan

        df["Genres"] = df.apply(union_genres, axis=1)
        return df.drop(columns=["Name_x", "Name_y", "Genres_x", "Genres_y", "English name"])

    def _bin(self, series: pd.Series, key: str):
        cfg = self.bins_cfg[key]
        bins = [float(b) for b in cfg["bins"]]
        return pd.cut(series, bins=bins, labels=cfg["labels"])

    def _impute_and_bin(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Source"] = df["Source"].fillna("Original")

        df["Episodes"] = df.groupby("Source")["Episodes"].transform(lambda x: x.fillna(x.median()))
        df["Episodes"] = df["Episodes"].fillna(df["Episodes"].median())
        df["Ep_bin"] = self._bin(df["Episodes"], "ep_bin")
        df = df.drop(columns=["Episodes"])

        df = df.dropna(subset=["synopsis"])

        df["Premiered"] = df["Premiered"].fillna("Unknown")
        df["Producers"] = df["Producers"].fillna("Unknown")
        df["Studios"] = df["Studios"].fillna("Unknown")
        df["Rating"] = df["Rating"].fillna(df["Rating"].mode()[0])
        df["Genres"] = df["Genres"].fillna("Unknown")

        df["Duration"] = df.groupby("Ep_bin", observed=False)["Duration"].transform(lambda x: x.fillna(x.median()))
        df["Duration"] = df["Duration"].fillna(df["Duration"].median())
        df["Dur_bin"] = self._bin(df["Duration"], "dur_bin")
        df = df.drop(columns=["Duration"])

        start_date = pd.to_datetime(df["Aired"].str.extract(r"^([A-Za-z]+ \d+, \d{4})")[0], errors="coerce")
        year_from_premiered = df["Premiered"].str.extract(r"(\d{4})").astype(float)[0]
        start_year = start_date.dt.year.fillna(year_from_premiered)
        start_year = start_year.fillna(start_year.median())
        df["Era"] = self._bin(start_year, "era")

        return df.drop(columns=["Aired", "Premiered"])

    def _split(self, df: pd.DataFrame):
        df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        n = len(df)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        train = df.iloc[:n_train]
        val = df.iloc[n_train : n_train + n_val]
        test = df.iloc[n_train + n_val :]
        return train, val, test

    def _preprocess_anime(self) -> None:
        anime_df = pd.read_csv(os.path.join(self.config.raw_data_dir, ANIME_FILE_NAME))
        synopsis_df = pd.read_csv(os.path.join(self.config.raw_data_dir, ANIME_SYNOPSIS_FILE_NAME))

        anime_df = self._clean_anime_df(anime_df)
        synopsis_df = self._clean_synopsis_df(synopsis_df)

        df = self._merge(anime_df, synopsis_df)
        df = self._impute_and_bin(df)

        train, val, test = self._split(df)
        save_df(train, os.path.join(self.config.train_data_dir, SPLIT_FILE_NAME))
        save_df(val, os.path.join(self.config.val_data_dir, SPLIT_FILE_NAME))
        save_df(test, os.path.join(self.config.test_data_dir, SPLIT_FILE_NAME))

    def _preprocess_ratings(self) -> None:
        ratings_df = pd.read_csv(os.path.join(self.config.raw_data_dir, RATINGS_FILE_NAME))
        ratings_df = ratings_df.dropna(subset=["anime_id", "rating"])
        ratings_df["anime_id"] = ratings_df["anime_id"].astype(int)

        train, val, test = self._split(ratings_df)
        save_df(train, os.path.join(self.config.train_data_dir, RATINGS_SPLIT_FILE_NAME))
        save_df(val, os.path.join(self.config.val_data_dir, RATINGS_SPLIT_FILE_NAME))
        save_df(test, os.path.join(self.config.test_data_dir, RATINGS_SPLIT_FILE_NAME))

    def preprocess(self) -> None:
        self._preprocess_anime()
        self._preprocess_ratings()

    def run(self) -> DataPreprocessingArtifact:
        self.preprocess()
        return DataPreprocessingArtifact(
            data_preprocessing_dir=self.config.data_preprocessing_dir,
            raw_data_dir=self.config.raw_data_dir,
            train_data_dir=self.config.train_data_dir,
            test_data_dir=self.config.test_data_dir,
            val_data_dir=self.config.val_data_dir,
        )
