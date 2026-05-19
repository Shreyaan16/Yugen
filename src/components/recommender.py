import json
import os
import pickle
import numpy as np
import pandas as pd
import faiss
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from src.constants import (CONTENT_INDEX_FILE, ANIME_ID_TO_IDX_FILE, TFIDF_VECTORIZER_FILE, USER_INDEX_FILE, USER_ID_TO_IDX_FILE)
from src.entity.config_entity import (ContentBasedRecommenderConfig, UserBasedRecommenderConfig, CFRecommenderConfig, HybridRecommenderConfig)
from src.entity.artifact_entity import (ContentBasedRecommenderArtifact, UserBasedRecommenderArtifact,  CFRecommenderArtifact, HybridRecommenderArtifact)


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
        """Load index + user_id_to_idx from disk. user_vectors_sparse is NOT restored
        (would require retraining); methods that need it (similar_users) won't work
        after load — use index.reconstruct() instead."""
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

    def similar_users(self, user_id: int, k: int = 10) -> pd.DataFrame:
        u_idx = self.user_id_to_idx.get(int(user_id))
        if u_idx is None:
            raise ValueError(f"user_id {user_id} not found")
        if u_idx == self.avg_idx:
            raise ValueError(f"user_id {user_id} is cold; no neighbors available")

        query = self.user_vectors_sparse[u_idx].toarray().astype(np.float32)
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


class CFRecommender:
    def __init__(self, config: CFRecommenderConfig | None = None):
        self.config = config or CFRecommenderConfig()
        os.makedirs(self.config.cf_dir, exist_ok=True)

        self.user_based: UserBasedRecommender | None = None
        self.anime_df: pd.DataFrame | None = None
        self.user_anime_ratings_df: pd.DataFrame | None = None
        self.ratings_df: pd.DataFrame | None = None

    def fit(self, user_based: UserBasedRecommender, anime_df: pd.DataFrame, user_anime_ratings_df: pd.DataFrame, ratings_df: pd.DataFrame,) -> "CFRecommender":
        self.user_based = user_based
        self.anime_df = anime_df
        self.user_anime_ratings_df = user_anime_ratings_df
        self.ratings_df = ratings_df
        return self

    def save(self) -> CFRecommenderArtifact:
        return CFRecommenderArtifact(cf_dir=self.config.cf_dir)

    def run(self, user_based: UserBasedRecommender, anime_df: pd.DataFrame, user_anime_ratings_df: pd.DataFrame, ratings_df: pd.DataFrame) -> CFRecommenderArtifact:
        self.fit(user_based, anime_df, user_anime_ratings_df, ratings_df)
        return self.save()

    def recommend(self, user_id: int) -> pd.DataFrame:
        ub = self.user_based
        cfg = self.config
        u_idx = ub.user_id_to_idx.get(int(user_id))
        if u_idx is None:
            raise ValueError(f"user_id {user_id} not found")
        if u_idx == ub.avg_idx:
            raise ValueError(f"user_id {user_id} is cold; use a fallback recommender")

        query = ub.user_vectors_sparse[u_idx].toarray().astype(np.float32)
        sims, nbr_idxs = ub.index.search(query, cfg.k_neighbors + 5)
        sims, nbr_idxs = sims[0], nbr_idxs[0]

        neighbor_ids, neighbor_sims = [], []
        for s, ni in zip(sims, nbr_idxs):
            if ni == -1 or ni == u_idx or ni == ub.avg_idx:
                continue
            nid = ub.idx_to_user_id.get(int(ni))
            if nid is None:
                continue
            neighbor_ids.append(nid)
            neighbor_sims.append(float(s))
            if len(neighbor_ids) == cfg.k_neighbors:
                break

        sim_map = dict(zip(neighbor_ids, neighbor_sims))
        uar = self.user_anime_ratings_df
        seen = set(uar.loc[uar["user_id"] == user_id, "anime_id"].tolist())

        nbr = uar[uar["user_id"].isin(sim_map) & ~uar["anime_id"].isin(seen)].copy()
        nbr["sim"] = nbr["user_id"].map(sim_map)
        nbr["w_rating"] = nbr["rating"] * nbr["sim"]

        agg = nbr.groupby("anime_id").agg(num_neighbors=("user_id", "size"), weighted_sum=("w_rating", "sum"), sim_sum=("sim", "sum"))
        agg = agg[agg["num_neighbors"] >= cfg.min_support]
        agg["score"] = agg["weighted_sum"] / agg["sim_sum"]

        if cfg.min_num_ratings > 0 and "num_ratings" in self.ratings_df.columns:
            reliable = set(
                self.ratings_df.loc[self.ratings_df["num_ratings"] >= cfg.min_num_ratings, "anime_id"]
            )
            agg = agg[agg.index.isin(reliable)]

        recs = (
            agg.sort_values("score", ascending=False)
            .head(cfg.top_n)
            .reset_index()
            .merge(
                self.anime_df[["anime_id", "title", "genres", "era"]],
                on="anime_id",
                how="left",
            )
        )
        return recs[["anime_id", "title", "genres", "era", "score", "num_neighbors"]]


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

    @staticmethod
    def _minmax(s: pd.Series) -> pd.Series:
        s = s.astype("float32")
        lo, hi = s.min(skipna=True), s.max(skipna=True)
        if pd.isna(lo) or hi == lo:
            return s.fillna(0.0) * 0.0
        return ((s - lo) / (hi - lo)).fillna(0.0)

    def recommend(self, user_id: int) -> pd.DataFrame:
        ub = self.user_based
        cb = self.content
        cfg = self.config

        u_idx = ub.user_id_to_idx.get(int(user_id))
        if u_idx is None:
            raise ValueError(f"user_id {user_id} not found")

        is_cold = u_idx == ub.avg_idx
        user_vec = ub.user_vectors_sparse[u_idx].toarray().astype(np.float32) if not is_cold \
            else ub.index.reconstruct(ub.avg_idx).reshape(1, -1)

        cf_scores = pd.Series(dtype="float32", name="cf_score")
        seen: set = set()

        if not is_cold:
            sims, nbr_idxs = ub.index.search(user_vec, cfg.k_neighbors + 5)
            sims, nbr_idxs = sims[0], nbr_idxs[0]
            neighbor_ids, neighbor_sims = [], []
            for s, ni in zip(sims, nbr_idxs):
                if ni == -1 or ni == u_idx or ni == ub.avg_idx:
                    continue
                nid = ub.idx_to_user_id.get(int(ni))
                if nid is None:
                    continue
                neighbor_ids.append(nid)
                neighbor_sims.append(float(s))
                if len(neighbor_ids) == cfg.k_neighbors:
                    break
            sim_map = dict(zip(neighbor_ids, neighbor_sims))

            uar = self.user_anime_ratings_df
            seen = set(uar.loc[uar["user_id"] == user_id, "anime_id"].tolist())

            nbr = uar[uar["user_id"].isin(sim_map) & ~uar["anime_id"].isin(seen)].copy()
            nbr["sim"] = nbr["user_id"].map(sim_map)
            nbr["w_rating"] = nbr["rating"] * nbr["sim"]

            agg = nbr.groupby("anime_id").agg(
                num_neighbors=("user_id", "size"),
                weighted_sum=("w_rating", "sum"),
                sim_sum=("sim", "sum"),
            )
            agg = agg[agg["num_neighbors"] >= cfg.min_support]
            cf_scores = (agg["weighted_sum"] / agg["sim_sum"]).rename("cf_score")

        c_scores_raw, c_idxs = cb.index.search(user_vec, cfg.content_pool)
        c_scores_raw, c_idxs = c_scores_raw[0], c_idxs[0]
        content_rows = []
        for s, ai in zip(c_scores_raw, c_idxs):
            if ai == -1:
                continue
            aid = cb.idx_to_anime_id[int(ai)]
            if aid in seen:
                continue
            content_rows.append((aid, float(s)))
        content_scores = pd.Series(dict(content_rows), name="content_score", dtype="float32")

        candidates = pd.concat([cf_scores, content_scores], axis=1)

        if cfg.min_num_ratings > 0 and "num_ratings" in self.ratings_df.columns:
            reliable = set(
                self.ratings_df.loc[self.ratings_df["num_ratings"] >= cfg.min_num_ratings, "anime_id"]
            )
            candidates = candidates[candidates.index.isin(reliable)]

        candidates["cf_n"] = self._minmax(candidates["cf_score"])
        candidates["content_n"] = self._minmax(candidates["content_score"])

        effective_alpha = 0.0 if is_cold else cfg.alpha
        candidates["final_score"] = (
            effective_alpha * candidates["cf_n"]
            + (1 - effective_alpha) * candidates["content_n"]
        )

        out = (
            candidates.sort_values("final_score", ascending=False)
            .head(cfg.top_n)
            .reset_index()
            .rename(columns={"index": "anime_id"})
            .merge(
                self.anime_df[["anime_id", "title", "genres", "era"]],
                on="anime_id",
                how="left",
            )
        )
        return out[["anime_id", "title", "genres", "era", "final_score", "cf_score", "content_score"]]
