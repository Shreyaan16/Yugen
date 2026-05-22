import json
import os
import numpy as np
import pandas as pd
import faiss
import scipy.sparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from recommender.constants import *
from recommender.entity.config_entity import UserBasedRecommenderConfig
from recommender.entity.artifact_entity import UserBasedRecommenderArtifact
from recommender.utils import read_yaml

_params = read_yaml(PARAMS_PATH)
_ub = _params["user_based"]


class UserBasedRecommender:
    def __init__(self, config: UserBasedRecommenderConfig | None = None):
        self.config = config or UserBasedRecommenderConfig()
        os.makedirs(self.config.user_based_dir, exist_ok=True)
        self.ratings_df: pd.DataFrame = pd.read_csv(os.path.join(self.config.preprocessed_dir, USER_ANIME_RATINGS_FILE_NAME))
        self.users_df: pd.DataFrame = pd.read_csv(os.path.join(self.config.preprocessed_dir, USERS_FILE_NAME))

        self.index: faiss.Index | None = None
        self.user_id_to_idx: dict[int, int] = {}
        self.idx_to_user_id: dict[int, int] = {}
        self.user_vectors_sparse = None
        self.avg_idx: int | None = None
        self.min_ratings: int = _ub["min_ratings"]
        self.chunk_size: int = _ub["chunk_size"]

        with open(self.config.anime_id_to_idx_path, "r", encoding="utf-8") as f:
            self.anime_idx_map: dict[int, int] = {int(k): int(v) for k, v in json.load(f).items()}

        self.tfidf_matrix = None  # built lazily in fit()

    def _load_if_needed(self):
        """Load index and mapping from disk if not already in memory."""
        if self.index is not None:
            return
        index_path = os.path.join(self.config.user_based_dir, USER_INDEX_FILE)
        user_id_to_idx_path = os.path.join(self.config.user_based_dir, USER_ID_TO_IDX_FILE)

        self.index = faiss.read_index(index_path)
        with open(user_id_to_idx_path, "r", encoding="utf-8") as f:
            self.user_id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
        self.avg_idx = self.index.ntotal - 1
        self.idx_to_user_id = {i: uid for uid, i in self.user_id_to_idx.items() if i != self.avg_idx}

    def fit(self, ratings_df: pd.DataFrame, users_df: pd.DataFrame) -> "UserBasedRecommender":
        # Load the saved sparse tfidf_matrix directly
        tfidf_matrix = scipy.sparse.load_npz(self.config.tfidf_matrix_path)
        self.tfidf_matrix = tfidf_matrix
        anime_idx_map = self.anime_idx_map
        ratings = ratings_df[ratings_df["anime_id"].isin(anime_idx_map)].copy()
        ratings["anime_idx"] = ratings["anime_id"].map(anime_idx_map)
        counts = ratings.groupby("user_id").size()
        active = counts[counts >= self.min_ratings].index
        ratings = ratings[ratings["user_id"].isin(active)]
        user_means = ratings.groupby("user_id")["rating"].transform("mean")
        ratings["centered"] = ratings["rating"] - user_means
        user_ids = ratings["user_id"].unique()
        user_idx_map = {int(uid): i for i, uid in enumerate(user_ids)}
        ratings["user_idx"] = ratings["user_id"].map(user_idx_map)
        n_users = len(user_ids)
        n_anime = tfidf_matrix.shape[0]
        R = csr_matrix(
            (
                ratings["centered"].astype(np.float32),
                (ratings["user_idx"].values, ratings["anime_idx"].values),
            ),
            shape=(n_users, n_anime),
        )
        user_vectors_sparse = R @ tfidf_matrix
        user_vectors_sparse = normalize(user_vectors_sparse, norm="l2", axis=1)
        dim = user_vectors_sparse.shape[1]
        index = faiss.IndexFlatIP(dim)
        chunk = self.chunk_size
        for start in range(0, n_users, chunk):
            end = min(start + chunk, n_users)
            block = user_vectors_sparse[start:end].toarray().astype(np.float32)
            index.add(block)
        avg_vec = np.asarray(user_vectors_sparse.mean(axis=0)).astype(np.float32)
        avg_vec = avg_vec / (np.linalg.norm(avg_vec) + 1e-12)
        avg_idx = index.ntotal
        index.add(avg_vec.reshape(1, -1))
        user_id_to_idx = {int(uid): int(i) for uid, i in user_idx_map.items()}
        all_user_ids = set(users_df["user_id"].astype(int).tolist())
        cold = all_user_ids - set(user_id_to_idx.keys())
        for uid in cold:
            user_id_to_idx[int(uid)] = int(avg_idx)
        self.index = index
        self.user_id_to_idx = user_id_to_idx
        self.idx_to_user_id = {i: uid for uid, i in user_idx_map.items()}
        self.user_vectors_sparse = user_vectors_sparse
        self.avg_idx = avg_idx
        return self

    def save(self) -> UserBasedRecommenderArtifact:
        index_path = os.path.join(self.config.user_based_dir, USER_INDEX_FILE)
        user_id_to_idx_path = os.path.join(self.config.user_based_dir, USER_ID_TO_IDX_FILE)

        faiss.write_index(self.index, index_path)
        with open(user_id_to_idx_path, "w", encoding="utf-8") as f:
            json.dump(self.user_id_to_idx, f)

        return UserBasedRecommenderArtifact(
            user_based_dir=self.config.user_based_dir,
            index_path=index_path,
            user_id_to_idx_path=user_id_to_idx_path,
        )

    def get_user_vec(self, u_idx: int) -> np.ndarray:
        self._load_if_needed()
        if self.user_vectors_sparse is not None:
            return self.user_vectors_sparse[u_idx].toarray().astype(np.float32)
        return self.index.reconstruct(int(u_idx)).reshape(1, -1).astype(np.float32)

    def similar_users(self, user_id: int, k: int = 10) -> pd.DataFrame:
        self._load_if_needed()
        u_idx = self.user_id_to_idx.get(int(user_id))
        if u_idx is None:
            raise ValueError(f"user_id {user_id} not found")
        if u_idx == self.avg_idx:
            raise ValueError(f"user_id {user_id} is cold; no neighbors available")
        query = self.get_user_vec(u_idx)
        sims, nbrs = self.index.search(query, k + 5)
        rows = []
        for s, ni in zip(sims[0], nbrs[0]):
            if ni == -1 or ni == u_idx or ni == self.avg_idx:
                continue
            uid = self.idx_to_user_id.get(int(ni))
            if uid is None:
                continue
            rows.append({"user_id": int(uid), "similarity": float(s)})
            if len(rows) == k:
                break
        return pd.DataFrame(rows)

    def run(self, ratings_df: pd.DataFrame, users_df: pd.DataFrame) -> UserBasedRecommenderArtifact:
        self.fit(ratings_df, users_df)
        return self.save()