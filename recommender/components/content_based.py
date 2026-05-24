import json
import os
import pickle
import numpy as np
import pandas as pd
import faiss
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from recommender.constants import *
from recommender.entity.config_entity import ContentBasedRecommenderConfig
from recommender.entity.artifact_entity import ContentBasedRecommenderArtifact
from recommender.utils import read_yaml

_params = read_yaml(PARAMS_PATH)
_cb = _params["content_based"]

def _join_list(x):
    return " ".join(x) if isinstance(x, list) else ""

class ContentBasedRecommender:
    def __init__(self, config: ContentBasedRecommenderConfig | None = None):
        self.config = config or ContentBasedRecommenderConfig()
        os.makedirs(self.config.content_based_dir, exist_ok=True)
        self.anime_df: pd.DataFrame = pd.read_csv(os.path.join(self.config.preprocessed_dir, ANIME_FILE_NAME)) 
        self.tfidf_matrix = None
        self.vectors: np.ndarray | None = None
        self.index: faiss.Index | None = None
        self.anime_id_to_idx: dict[int, int] = {}
        self.idx_to_anime_id: dict[int, int] = {}
        self.max_features: int = _cb["max_features"]
        self.ngram_range: tuple = tuple(_cb["ngram_range"])
        self.min_df: int = _cb["min_df"]

    @staticmethod
    def _build_soup(df: pd.DataFrame) -> pd.Series:
        list_cols = LIST_COLUMNS
        parts: list[pd.Series] = []
        for col in COLUMN_NAMES:
            if col not in df.columns:
                continue
            if col in list_cols:
                parts.append(df[col].apply(_join_list))
            else:
                parts.append(df[col].fillna("").astype(str))
        if not parts:
            return pd.Series([""] * len(df), index=df.index)
        soup = parts[0]
        for part in parts[1:]:
            soup = soup + " " + part
        return soup.str.lower()

    def _load_if_needed(self):
        if self.index is not None:
            return
        out_dir = self.config.content_based_dir
        self.index = faiss.read_index(os.path.join(out_dir, CONTENT_INDEX_FILE))
        with open(os.path.join(out_dir, ANIME_ID_TO_IDX_FILE), "r", encoding="utf-8") as f:
            self.anime_id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
        self.idx_to_anime_id = {v: k for k, v in self.anime_id_to_idx.items()}
        self.anime_df = pd.read_csv(os.path.join(self.config.preprocessed_dir, ANIME_FILE_NAME))
        tfidf_matrix_path = os.path.join(out_dir, TFIDF_MATRIX_FILE)
        if os.path.exists(tfidf_matrix_path):
            self.tfidf_matrix = scipy.sparse.load_npz(tfidf_matrix_path)
        # Reconstruct all vectors into a dense numpy array in one C call so the
        # hybrid recommender can do batch dot-products instead of a Python loop
        # of individual index.reconstruct() calls.
        n, d = self.index.ntotal, self.index.d
        self.vectors = np.empty((n, d), dtype=np.float32)
        self.index.reconstruct_n(0, n, self.vectors)

    def fit(self, anime_df: pd.DataFrame) -> "ContentBasedRecommender":
        df = anime_df.reset_index(drop=True).copy()
        soup = self._build_soup(df)
        vectorizer = TfidfVectorizer(stop_words="english", max_features=self.max_features, ngram_range=self.ngram_range, min_df=self.min_df)
        tfidf_matrix = vectorizer.fit_transform(soup)
        vectors = normalize(tfidf_matrix, norm="l2", axis=1).astype(np.float32).toarray()
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.anime_df = df
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.vectors = vectors
        self.index = index
        self.anime_id_to_idx = {int(a): int(i) for i, a in enumerate(df["anime_id"].tolist())}
        self.idx_to_anime_id = {v: k for k, v in self.anime_id_to_idx.items()}
        return self

    def save(self) -> ContentBasedRecommenderArtifact:
        out_dir = self.config.content_based_dir
        index_path = os.path.join(out_dir, CONTENT_INDEX_FILE)
        anime_id_to_idx_path = os.path.join(out_dir, ANIME_ID_TO_IDX_FILE)
        tfidf_path = os.path.join(out_dir, TFIDF_VECTORIZER_FILE)
        tfidf_matrix_path = os.path.join(out_dir, TFIDF_MATRIX_FILE)
        faiss.write_index(self.index, index_path)
        with open(anime_id_to_idx_path, "w", encoding="utf-8") as f:
            json.dump(self.anime_id_to_idx, f)
        with open(tfidf_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        scipy.sparse.save_npz(tfidf_matrix_path, self.tfidf_matrix)
        return ContentBasedRecommenderArtifact(
            content_based_dir=out_dir,
            index_path=index_path,
            anime_id_to_idx_path=anime_id_to_idx_path,
            tfidf_vectorizer_path=tfidf_path,
            tfidf_matrix_path=tfidf_matrix_path,
        )

    def recommend(self, anime_id: int, k: int = 10) -> pd.DataFrame:
        self._load_if_needed()
        idx = self.anime_id_to_idx.get(int(anime_id))
        if idx is None:
            raise ValueError(f"anime_id {anime_id} not found")
        query = (self.vectors[idx : idx + 1] if self.vectors is not None else self.index.reconstruct(idx).reshape(1, -1))
        scores, neighbors = self.index.search(query, k + 1)
        rows = []
        for score, nbr in zip(scores[0], neighbors[0]):
            if nbr == idx or nbr == -1:
                continue
            rows.append({"anime_id": self.idx_to_anime_id[int(nbr)], "similarity": float(score)})
            if len(rows) == k:
                break
        return pd.DataFrame(rows).merge(self.anime_df[["anime_id", "title", "genres", "era"]], on="anime_id", how="left")

    def run(self, anime_df: pd.DataFrame) -> ContentBasedRecommenderArtifact:
        self.fit(anime_df)
        return self.save()