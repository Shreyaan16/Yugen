import json
import os
import pickle
import numpy as np
import pandas as pd
import faiss
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from recommender.src.constants import (CONTENT_INDEX_FILE, ANIME_ID_TO_IDX_FILE, TFIDF_VECTORIZER_FILE, USER_INDEX_FILE, USER_ID_TO_IDX_FILE)
from recommender.src.entity.config_entity import (ContentBasedRecommenderConfig, UserBasedRecommenderConfig, HybridRecommenderConfig)
from recommender.src.entity.artifact_entity import (ContentBasedRecommenderArtifact, UserBasedRecommenderArtifact, HybridRecommenderArtifact)


def _join_list(x):
    return " ".join(x) if isinstance(x, list) else ""

class ContentBasedRecommender:
    def __init__(self, config: ContentBasedRecommenderConfig | None = None):
        self.config = config or ContentBasedRecommenderConfig()
        os.makedirs(self.config.content_based_dir, exist_ok=True)

        self.anime_df: pd.DataFrame | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.vectors: np.ndarray | None = None
        self.index: faiss.Index | None = None
        self.anime_id_to_idx: dict[int, int] = {}
        self.idx_to_anime_id: dict[int, int] = {}

    @staticmethod
    def _build_soup(df: pd.DataFrame) -> pd.Series:
        return (
            df["title"].fillna("") + " "
            + df["synopsis"].fillna("") + " "
            + df["rating"].fillna("") + " "
            + df["ep_bin"].fillna("") + " "
            + df["dur_bin"].fillna("") + " "
            + df["era"].fillna("") + " "
            + df["source"].fillna("") + " "
            + df["genres"].apply(_join_list) + " "
            + df["producers"].apply(_join_list) + " "
            + df["studios"].apply(_join_list)
        ).str.lower()

    def fit(self, anime_df: pd.DataFrame) -> "ContentBasedRecommender":
        df = anime_df.reset_index(drop=True).copy()
        soup = self._build_soup(df)

        vectorizer = TfidfVectorizer(stop_words="english", max_features=self.config.max_features, ngram_range=self.config.ngram_range, min_df=self.config.min_df)
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

        faiss.write_index(self.index, index_path)
        with open(anime_id_to_idx_path, "w", encoding="utf-8") as f:
            json.dump(self.anime_id_to_idx, f)
        with open(tfidf_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        return ContentBasedRecommenderArtifact(
            content_based_dir=out_dir,
            index_path=index_path,
            anime_id_to_idx_path=anime_id_to_idx_path,
            tfidf_vectorizer_path=tfidf_path,
        )

    def run(self, anime_df: pd.DataFrame) -> ContentBasedRecommenderArtifact:
        self.fit(anime_df)
        return self.save()

    def load(self, anime_df: pd.DataFrame) -> "ContentBasedRecommender":
        out_dir = self.config.content_based_dir
        self.anime_df = anime_df.reset_index(drop=True).copy()

        with open(os.path.join(out_dir, TFIDF_VECTORIZER_FILE), "rb") as f:
            self.vectorizer = pickle.load(f)
        soup = self._build_soup(self.anime_df)
        self.tfidf_matrix = self.vectorizer.transform(soup)
        self.vectors = normalize(self.tfidf_matrix, norm="l2", axis=1).astype(np.float32).toarray()

        self.index = faiss.read_index(os.path.join(out_dir, CONTENT_INDEX_FILE))
        with open(os.path.join(out_dir, ANIME_ID_TO_IDX_FILE), "r", encoding="utf-8") as f:
            self.anime_id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
        self.idx_to_anime_id = {v: k for k, v in self.anime_id_to_idx.items()}
        return self

    def recommend(self, anime_id: int, k: int = 10) -> pd.DataFrame:
        idx = self.anime_id_to_idx.get(int(anime_id))
        if idx is None:
            raise ValueError(f"anime_id {anime_id} not found")
        query = self.vectors[idx : idx + 1]
        scores, neighbors = self.index.search(query, k + 1)
        rows = []
        for score, nbr in zip(scores[0], neighbors[0]):
            if nbr == idx or nbr == -1:
                continue
            rows.append({"anime_id": self.idx_to_anime_id[int(nbr)], "similarity": float(score)})
            if len(rows) == k:
                break
        return pd.DataFrame(rows).merge(self.anime_df[["anime_id", "title", "genres", "era"]], on="anime_id", how="left")


class UserBasedRecommender:
    def __init__(self, config: UserBasedRecommenderConfig | None = None):
        self.config = config or UserBasedRecommenderConfig()
        os.makedirs(self.config.user_based_dir, exist_ok=True)

        self.index: faiss.Index | None = None
        self.user_id_to_idx: dict[int, int] = {}
        self.idx_to_user_id: dict[int, int] = {}
        self.user_vectors_sparse = None
        self.avg_idx: int | None = None

    def fit(self, ratings_df: pd.DataFrame, users_df: pd.DataFrame, content: ContentBasedRecommender) -> "UserBasedRecommender":
        anime_idx_map = content.anime_id_to_idx
        tfidf_matrix = content.tfidf_matrix

        ratings = ratings_df[ratings_df["anime_id"].isin(anime_idx_map)].copy()
        ratings["anime_idx"] = ratings["anime_id"].map(anime_idx_map)

        counts = ratings.groupby("user_id").size()
        active = counts[counts >= self.config.min_ratings].index
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
        chunk = self.config.chunk_size
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
        out_dir = self.config.user_based_dir
        index_path = os.path.join(out_dir, USER_INDEX_FILE)
        user_id_to_idx_path = os.path.join(out_dir, USER_ID_TO_IDX_FILE)

        faiss.write_index(self.index, index_path)
        with open(user_id_to_idx_path, "w", encoding="utf-8") as f:
            json.dump(self.user_id_to_idx, f)

        return UserBasedRecommenderArtifact(user_based_dir=out_dir, index_path=index_path, user_id_to_idx_path=user_id_to_idx_path)

    def run(self, ratings_df: pd.DataFrame, users_df: pd.DataFrame, content: ContentBasedRecommender) -> UserBasedRecommenderArtifact:
        self.fit(ratings_df, users_df, content)
        return self.save()

    def load(self) -> "UserBasedRecommender":
        """Load index + user_id_to_idx from disk. user_vectors_sparse is not
        restored — lookups go through get_user_vec which uses index.reconstruct."""
        out_dir = self.config.user_based_dir
        self.index = faiss.read_index(os.path.join(out_dir, USER_INDEX_FILE))
        with open(os.path.join(out_dir, USER_ID_TO_IDX_FILE), "r", encoding="utf-8") as f:
            self.user_id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}

        # Last vector in the index is the avg vector for cold users.
        self.avg_idx = self.index.ntotal - 1
        # idx_to_user_id excludes cold users (which all share avg_idx).
        self.idx_to_user_id = {
            i: uid for uid, i in self.user_id_to_idx.items() if i != self.avg_idx
        }
        return self

    def get_user_vec(self, u_idx: int) -> np.ndarray:
        """Return a (1, dim) float32 vector for index position u_idx.
        Uses the cached sparse matrix when available (post-training), else
        falls back to FAISS reconstruct (post-load)."""
        if self.user_vectors_sparse is not None:
            return self.user_vectors_sparse[u_idx].toarray().astype(np.float32)
        return self.index.reconstruct(int(u_idx)).reshape(1, -1).astype(np.float32)

    def similar_users(self, user_id: int, k: int = 10) -> pd.DataFrame:
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


class HybridRecommender:
    def __init__(self, config: HybridRecommenderConfig | None = None):
        self.config = config or HybridRecommenderConfig()
        os.makedirs(self.config.hybrid_dir, exist_ok=True)

        self.content: ContentBasedRecommender | None = None
        self.user_based: UserBasedRecommender | None = None
        self.anime_df: pd.DataFrame | None = None
        self.user_anime_ratings_df: pd.DataFrame | None = None
        self.ratings_df: pd.DataFrame | None = None

    def fit(self, content: ContentBasedRecommender, user_based: UserBasedRecommender, anime_df: pd.DataFrame, user_anime_ratings_df: pd.DataFrame, 
            ratings_df: pd.DataFrame) -> "HybridRecommender":
        self.content = content
        self.user_based = user_based
        self.anime_df = anime_df
        self.user_anime_ratings_df = user_anime_ratings_df
        self.ratings_df = ratings_df
        return self

    def save(self) -> HybridRecommenderArtifact:
        return HybridRecommenderArtifact(hybrid_dir=self.config.hybrid_dir)

    def run(self, content: ContentBasedRecommender, user_based: UserBasedRecommender, anime_df: pd.DataFrame, user_anime_ratings_df: pd.DataFrame,
        ratings_df: pd.DataFrame,) -> HybridRecommenderArtifact:
        self.fit(content, user_based, anime_df, user_anime_ratings_df, ratings_df)
        return self.save()

    def _reliable_anime_ids(self) -> set | None:
        cfg = self.config
        if cfg.min_num_ratings > 0 and "num_ratings" in self.ratings_df.columns:
            return set(
                self.ratings_df.loc[self.ratings_df["num_ratings"] >= cfg.min_num_ratings, "anime_id"]
            )
        return None

    def recommend(self, user_id: int) -> pd.DataFrame:
        """
        Two-stage hybrid:
          1. CF candidate generation — top-k similar users supply a pool of anime
             they rated >= neighbor_rating_threshold.
          2. Content ranking — pool is ranked by cosine(user_taste_vec, anime_vec).
        Cold users fall back to pure content search against the avg taste vector.
        """
        ub = self.user_based
        cfg = self.config

        u_idx = ub.user_id_to_idx.get(int(user_id))
        if u_idx is None:
            raise ValueError(f"user_id {user_id} not found")

        is_cold = u_idx == ub.avg_idx
        user_vec = ub.get_user_vec(ub.avg_idx if is_cold else u_idx)

        uar = self.user_anime_ratings_df
        seen = set(uar.loc[uar["user_id"] == user_id, "anime_id"].tolist())
        reliable = self._reliable_anime_ids()

        pool_aids, scores = self._score_pool(
            user_vec=user_vec,
            self_idx=u_idx,
            seen=seen,
            reliable=reliable,
            is_cold=is_cold,
        )

        if not pool_aids:
            return pd.DataFrame(columns=["anime_id", "title", "genres", "era", "score"])

        order = np.argsort(-scores)[: cfg.top_n]
        recs = pd.DataFrame({
            "anime_id": [pool_aids[i] for i in order],
            "score": [float(scores[i]) for i in order],
        })
        return recs.merge(
            self.anime_df[["anime_id", "title", "genres", "era"]],
            on="anime_id",
            how="left",
        )[["anime_id", "title", "genres", "era", "score"]]

    def _score_pool(self, user_vec: np.ndarray, self_idx: int | None, seen: set,
                    reliable: set | None, is_cold: bool) -> tuple[list[int], np.ndarray]:
        """Build candidate pool from CF neighbors, then score by content similarity."""
        ub = self.user_based
        cb = self.content
        cfg = self.config

        if is_cold:
            # No neighbors → fall back to top-K content matches over the whole catalogue.
            k_fetch = cfg.top_n + len(seen) + 50
            scores_raw, idxs = cb.index.search(user_vec, k_fetch)
            pool_aids, pool_scores = [], []
            for s, ai in zip(scores_raw[0], idxs[0]):
                if ai == -1:
                    continue
                aid = cb.idx_to_anime_id[int(ai)]
                if aid in seen:
                    continue
                if reliable is not None and aid not in reliable:
                    continue
                pool_aids.append(aid)
                pool_scores.append(float(s))
            return pool_aids, np.array(pool_scores, dtype=np.float32)

        sims, nbr_idxs = ub.index.search(user_vec, cfg.k_neighbors + 5)
        sims, nbr_idxs = sims[0], nbr_idxs[0]
        neighbor_ids: list[int] = []
        for s, ni in zip(sims, nbr_idxs):
            if ni == -1 or ni == self_idx or ni == ub.avg_idx:
                continue
            nid = ub.idx_to_user_id.get(int(ni))
            if nid is None:
                continue
            neighbor_ids.append(nid)
            if len(neighbor_ids) == cfg.k_neighbors:
                break
        if not neighbor_ids:
            return [], np.array([], dtype=np.float32)

        uar = self.user_anime_ratings_df
        pool_rows = uar[
            uar["user_id"].isin(neighbor_ids)
            & (uar["rating"] >= cfg.neighbor_rating_threshold)
            & ~uar["anime_id"].isin(seen)
        ]
        pool = pool_rows["anime_id"].unique().tolist()
        if reliable is not None:
            pool = [aid for aid in pool if aid in reliable]
        pool = [aid for aid in pool if aid in cb.anime_id_to_idx]
        if not pool:
            return [], np.array([], dtype=np.float32)

        pool_idxs = np.array([cb.anime_id_to_idx[aid] for aid in pool], dtype=np.int64)
        pool_vecs = cb.vectors[pool_idxs]
        scores = pool_vecs @ user_vec.ravel().astype(np.float32)
        return pool, scores.astype(np.float32)
