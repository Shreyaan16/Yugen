import json
import os
import numpy as np
import pandas as pd
import faiss
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.constants import (
    CONTENT_INDEX_FILE,
    CONTENT_MAL_TO_FAISS_FILE,
    CONTENT_FAISS_TO_MAL_FILE,
    USER_INDEX_FILE,
    USER_ID_TO_FAISS_FILE,
    USER_ID_TO_VECTOR_FILE,
)
from src.entity.config_entity import ContentBasedRecommenderConfig, UserBasedRecommenderConfig
from src.entity.artifact_entity import ContentBasedRecommenderArtifact, UserBasedRecommenderArtifact


class ContentBasedRecommender:
    def __init__(self, config: ContentBasedRecommenderConfig | None = None):
        self.config = config or ContentBasedRecommenderConfig()
        self.max_features = self.config.max_features
        self.n_components = self.config.n_components
        self.hnsw_m = self.config.hnsw_m
        self.ef_construction = self.config.ef_construction
        self.random_state = self.config.random_state

        self.df: pd.DataFrame | None = None
        self.embeddings: np.ndarray | None = None
        self.index: faiss.Index | None = None
        self.mal_id_to_faiss_id: dict[int, int] = {}
        self.faiss_id_to_mal_id: dict[int, int] = {}
        self.title_to_idx: pd.Series | None = None

    @staticmethod
    def _build_soup(row: pd.Series) -> str:
        genres = str(row["Genres"]).replace(",", " ").replace("-", "")
        producers = str(row["Producers"]).replace(",", " ")
        studios = str(row["Studios"]).replace(",", " ")
        source = str(row["Source"]).replace(" ", "_")
        rating = str(row["Rating"]).replace(" ", "_").replace("-", "")
        ep_bin = str(row["Ep_bin"])
        dur_bin = str(row["Dur_bin"])
        era = str(row["Era"])
        synopsis = str(row["synopsis"])
        title = str(row["Title"]).replace(" ", "_").replace("-", "")
        return f"{genres} {genres} {studios} {producers} {source} {rating} {ep_bin} {dur_bin} {era} {synopsis} {title}"

    def fit(self, df: pd.DataFrame) -> "ContentBasedRecommender":
        df = df.reset_index(drop=True).copy()
        df["soup"] = df.apply(self._build_soup, axis=1)

        tfidf = TfidfVectorizer(stop_words="english", max_features=self.max_features)
        tfidf_matrix = tfidf.fit_transform(df["soup"])

        svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        embeddings = svd.fit_transform(tfidf_matrix).astype("float32")
        embeddings = normalize(embeddings)

        index = faiss.IndexHNSWFlat(embeddings.shape[1], self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.add(embeddings)

        self.df = df
        self.embeddings = embeddings
        self.index = index
        self.mal_id_to_faiss_id = {int(m): int(i) for i, m in enumerate(df["MAL_ID"]) if pd.notna(m)}
        self.faiss_id_to_mal_id = {v: k for k, v in self.mal_id_to_faiss_id.items()}
        self.title_to_idx = pd.Series(df.index, index=df["Title"].str.lower())
        return self

    def save(self) -> ContentBasedRecommenderArtifact:
        assert self.index is not None, "Call fit() first"
        out_dir = self.config.content_based_dir
        os.makedirs(out_dir, exist_ok=True)

        index_path = os.path.join(out_dir, CONTENT_INDEX_FILE)
        mal_to_faiss_path = os.path.join(out_dir, CONTENT_MAL_TO_FAISS_FILE)
        faiss_to_mal_path = os.path.join(out_dir, CONTENT_FAISS_TO_MAL_FILE)

        faiss.write_index(self.index, index_path)
        with open(mal_to_faiss_path, "w", encoding="utf-8") as f:
            json.dump(self.mal_id_to_faiss_id, f)
        with open(faiss_to_mal_path, "w", encoding="utf-8") as f:
            json.dump(self.faiss_id_to_mal_id, f)

        return ContentBasedRecommenderArtifact(
            content_based_dir=out_dir,
            index_path=index_path,
            mal_to_faiss_path=mal_to_faiss_path,
            faiss_to_mal_path=faiss_to_mal_path,
        )

    def load(self, df: pd.DataFrame) -> "ContentBasedRecommender":
        out_dir = self.config.content_based_dir
        self.df = df.reset_index(drop=True).copy()
        self.index = faiss.read_index(os.path.join(out_dir, CONTENT_INDEX_FILE))
        with open(os.path.join(out_dir, CONTENT_MAL_TO_FAISS_FILE), "r", encoding="utf-8") as f:
            self.mal_id_to_faiss_id = {int(k): int(v) for k, v in json.load(f).items()}
        self.faiss_id_to_mal_id = {v: k for k, v in self.mal_id_to_faiss_id.items()}
        self.embeddings = np.vstack(
            [self.index.reconstruct(i) for i in range(self.index.ntotal)]
        ).astype("float32")
        self.title_to_idx = pd.Series(self.df.index, index=self.df["Title"].str.lower())
        return self

    def run(self, df: pd.DataFrame) -> ContentBasedRecommenderArtifact:
        self.fit(df)
        return self.save()

    def recommend(self, title: str, n: int = 10) -> pd.DataFrame | None:
        assert self.index is not None and self.df is not None, "Call fit() or load() first"
        key = title.lower()
        if self.title_to_idx is None or key not in self.title_to_idx:
            print(f"'{title}' not found.")
            return None

        idx = int(self.title_to_idx[key])
        query_vec = self.embeddings[idx : idx + 1]
        distances, indices = self.index.search(query_vec, n + 1)

        top_indices = [i for i in indices[0] if i != idx][:n]
        sim_scores = [d for i, d in zip(indices[0], distances[0]) if i != idx][:n]

        results = self.df.iloc[top_indices][["Title", "Genres", "Studios", "Era", "Ep_bin"]].copy()
        results["similarity"] = np.round(sim_scores, 3)
        return results.reset_index(drop=True)


class UserBasedRecommender:
    def __init__(self, config: UserBasedRecommenderConfig | None = None):
        self.config = config or UserBasedRecommenderConfig()
        self.hnsw_m = self.config.hnsw_m
        self.ef_construction = self.config.ef_construction

        self.user_ids: np.ndarray | None = None
        self.user_matrix: np.ndarray | None = None
        self.index: faiss.Index | None = None
        self.user_id_to_faiss_id: dict[int, int] = {}

    def fit(
        self,
        ratings_df: pd.DataFrame,
        content_artifact: ContentBasedRecommenderArtifact,
    ) -> "UserBasedRecommender":
        content_index = faiss.read_index(content_artifact.index_path)
        with open(content_artifact.mal_to_faiss_path, "r", encoding="utf-8") as f:
            mal_id_to_faiss_id = {int(k): int(v) for k, v in json.load(f).items()}

        ratings = ratings_df.dropna(subset=["anime_id", "rating"]).copy()
        ratings["anime_id"] = ratings["anime_id"].astype(int)
        ratings = ratings[ratings["anime_id"].isin(mal_id_to_faiss_id)]
        ratings = ratings[ratings["rating"] > 0]

        unique_anime_ids = ratings["anime_id"].unique()
        faiss_ids = np.array(
            [mal_id_to_faiss_id[a] for a in unique_anime_ids], dtype=np.int64
        )
        anime_vectors = np.vstack(
            [content_index.reconstruct(int(i)) for i in faiss_ids]
        ).astype("float32")
        anime_id_to_vec = dict(zip(unique_anime_ids, anime_vectors))

        user_ids_list: list[int] = []
        user_vectors_list: list[np.ndarray] = []
        for user_id, group in ratings.groupby("user_id", sort=False):
            vecs = np.stack([anime_id_to_vec[a] for a in group["anime_id"].to_numpy()])
            r = group["rating"].to_numpy(dtype="float32")
            user_ids_list.append(int(user_id))  # type: ignore[arg-type]
            user_vectors_list.append((vecs * r[:, None]).sum(axis=0) / r.sum())

        user_ids = np.array(user_ids_list)
        user_matrix = np.vstack(user_vectors_list).astype("float32")
        user_matrix = normalize(user_matrix)

        index = faiss.IndexHNSWFlat(user_matrix.shape[1], self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.add(user_matrix)

        self.user_ids = user_ids
        self.user_matrix = user_matrix
        self.index = index
        self.user_id_to_faiss_id = {int(u): int(i) for i, u in enumerate(user_ids)}
        return self

    def save(self) -> UserBasedRecommenderArtifact:
        assert self.index is not None and self.user_matrix is not None and self.user_ids is not None
        out_dir = self.config.user_based_dir
        os.makedirs(out_dir, exist_ok=True)

        index_path = os.path.join(out_dir, USER_INDEX_FILE)
        user_id_to_faiss_path = os.path.join(out_dir, USER_ID_TO_FAISS_FILE)
        user_id_to_vector_path = os.path.join(out_dir, USER_ID_TO_VECTOR_FILE)

        faiss.write_index(self.index, index_path)
        with open(user_id_to_faiss_path, "w", encoding="utf-8") as f:
            json.dump(self.user_id_to_faiss_id, f)

        user_id_to_vector = {
            int(u): self.user_matrix[i].tolist() for i, u in enumerate(self.user_ids)
        }
        with open(user_id_to_vector_path, "w", encoding="utf-8") as f:
            json.dump(user_id_to_vector, f)

        return UserBasedRecommenderArtifact(
            user_based_dir=out_dir,
            index_path=index_path,
            user_id_to_faiss_path=user_id_to_faiss_path,
            user_id_to_vector_path=user_id_to_vector_path,
        )

    def run(
        self,
        ratings_df: pd.DataFrame,
        content_artifact: ContentBasedRecommenderArtifact,
    ) -> UserBasedRecommenderArtifact:
        self.fit(ratings_df, content_artifact)
        return self.save()

    def recommend_similar_users(self, user_id: int, n: int = 5) -> pd.DataFrame | None:
        assert self.index is not None and self.user_matrix is not None and self.user_ids is not None
        user_id = int(user_id)
        if user_id not in self.user_id_to_faiss_id:
            print(f"User {user_id} not found.")
            return None

        query_idx = self.user_id_to_faiss_id[user_id]
        query_vec = self.user_matrix[query_idx : query_idx + 1]
        distances, indices = self.index.search(query_vec, n + 1)

        top_indices = [i for i in indices[0] if i != query_idx][:n]
        sim_scores = [d for i, d in zip(indices[0], distances[0]) if i != query_idx][:n]

        return pd.DataFrame(
            {
                "user_id": [int(self.user_ids[i]) for i in top_indices],
                "similarity": np.round(sim_scores, 3),
            }
        ).reset_index(drop=True)
