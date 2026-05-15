import json
import os
import numpy as np
import pandas as pd
import faiss
from scipy.sparse import load_npz, csr_matrix

from src.constants import (
    EVAL_HYBRID_METRICS_FILE,
    EVAL_CONTENT_METRICS_FILE,
    EVAL_USER_METRICS_FILE,
    EVAL_SUMMARY_FILE,
)
from src.entity.config_entity import EvaluationConfig
from src.entity.artifact_entity import (
    ContentBasedRecommenderArtifact,
    UserBasedRecommenderArtifact,
    HybridRecommenderArtifact,
    EvaluationArtifact,
)
from src.components.recommender import HybridRecommender


class Evaluation:
    def __init__(
        self,
        config: EvaluationConfig,
        content_artifact: ContentBasedRecommenderArtifact,
        user_artifact: UserBasedRecommenderArtifact,
        hybrid_artifact: HybridRecommenderArtifact,
        train_anime_df: pd.DataFrame,
    ):
        self.config = config
        self.content_artifact = content_artifact
        self.user_artifact = user_artifact
        self.hybrid_artifact = hybrid_artifact

        self.content_index = faiss.read_index(content_artifact.index_path)
        self.user_index = faiss.read_index(user_artifact.index_path)
        self.ratings_csr: csr_matrix = load_npz(hybrid_artifact.ratings_matrix_path)

        with open(content_artifact.mal_to_faiss_path, "r", encoding="utf-8") as f:
            self.mal_id_to_aidx = {int(k): int(v) for k, v in json.load(f).items()}
        self.aidx_to_mal_id = {v: k for k, v in self.mal_id_to_aidx.items()}

        with open(user_artifact.user_id_to_faiss_path, "r", encoding="utf-8") as f:
            self.user_id_to_uidx = {int(k): int(v) for k, v in json.load(f).items()}

        self.hybrid = HybridRecommender.__new__(HybridRecommender)
        self.hybrid.config = hybrid_artifact
        self.hybrid.similar_users_k = config.similar_users_k
        self.hybrid.top_k = config.top_k
        self.hybrid.ratings_csr = self.ratings_csr
        self.hybrid.user_index = self.user_index
        self.hybrid.mal_id_to_aidx = self.mal_id_to_aidx
        self.hybrid.aidx_to_mal_id = self.aidx_to_mal_id
        self.hybrid.user_id_to_uidx = self.user_id_to_uidx
        self.hybrid.anime_titles = {}

        self.genre_lookup: dict[int, set[str]] = {}
        for mal_id, genres in zip(train_anime_df["MAL_ID"], train_anime_df["Genres"]):
            if pd.isna(genres):
                continue
            self.genre_lookup[int(mal_id)] = {g.strip() for g in str(genres).split(",")}

    @staticmethod
    def _ndcg_at_k(rec_ids: list[int], relevant_set: set[int], k: int) -> float:
        dcg = sum(1.0 / np.log2(i + 2) for i, mal in enumerate(rec_ids[:k]) if mal in relevant_set)
        ideal_k = min(len(relevant_set), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))
        return float(dcg / idcg) if idcg > 0 else 0.0

    def evaluate_hybrid(self, ratings_df: pd.DataFrame) -> dict:
        threshold = self.config.relevant_threshold
        top_k = self.config.top_k

        relevant_df = ratings_df[ratings_df["rating"] >= threshold]

        users_evaluated = 0
        hit_users = 0
        recall_sum = 0.0
        precision_sum = 0.0
        ndcg_sum = 0.0

        for user_id, group in relevant_df.groupby("user_id"):
            uid = int(user_id)  # type: ignore[arg-type]
            if uid not in self.user_id_to_uidx:
                continue
            relevant_set = {int(a) for a in group["anime_id"].tolist()}
            try:
                recs = self.hybrid.recommend(uid, top_k=top_k)
            except Exception:
                continue
            rec_ids = [int(m) for m in recs["mal_id"].tolist()]
            n_hits = len(set(rec_ids) & relevant_set)

            users_evaluated += 1
            hit_users += 1 if n_hits > 0 else 0
            recall_sum += n_hits / len(relevant_set)
            precision_sum += n_hits / top_k
            ndcg_sum += self._ndcg_at_k(rec_ids, relevant_set, top_k)

        if users_evaluated == 0:
            return {"users_evaluated": 0}

        return {
            "users_evaluated": users_evaluated,
            "k": top_k,
            "relevant_threshold": threshold,
            "hit_rate@k": hit_users / users_evaluated,
            "recall@k": recall_sum / users_evaluated,
            "precision@k": precision_sum / users_evaluated,
            "ndcg@k": ndcg_sum / users_evaluated,
        }

    def evaluate_content(self) -> dict:
        k = self.config.content_neighbors_k
        sample_size = min(self.config.content_sample_size, self.content_index.ntotal)
        rng = np.random.default_rng(self.config.random_state)
        sample_aidx = rng.choice(self.content_index.ntotal, size=sample_size, replace=False)

        jaccards: list[float] = []
        evaluated = 0

        for aidx in sample_aidx:
            query_mal = self.aidx_to_mal_id.get(int(aidx))
            if query_mal is None or query_mal not in self.genre_lookup:
                continue
            query_genres = self.genre_lookup[query_mal]
            if not query_genres:
                continue

            vec = self.content_index.reconstruct(int(aidx)).reshape(1, -1)
            _, indices = self.content_index.search(vec, k + 1)
            neighbor_aidxs = [i for i in indices[0] if i != aidx][:k]

            per_query_jaccards = []
            for nidx in neighbor_aidxs:
                nmal = self.aidx_to_mal_id.get(int(nidx))
                if nmal is None or nmal not in self.genre_lookup:
                    continue
                ngenres = self.genre_lookup[nmal]
                union = query_genres | ngenres
                if not union:
                    continue
                per_query_jaccards.append(len(query_genres & ngenres) / len(union))

            if per_query_jaccards:
                jaccards.append(float(np.mean(per_query_jaccards)))
                evaluated += 1

        return {
            "queries_evaluated": evaluated,
            "k": k,
            "mean_genre_jaccard@k": float(np.mean(jaccards)) if jaccards else 0.0,
            "median_genre_jaccard@k": float(np.median(jaccards)) if jaccards else 0.0,
        }

    def _user_based_recommend(self, user_id: int) -> list[int]:
        threshold = self.config.relevant_threshold / 10.0
        k_users = self.config.similar_users_k
        top_k = self.config.top_k

        uidx = self.user_id_to_uidx[user_id]
        query_vec = self.user_index.reconstruct(uidx).reshape(1, -1)
        sims, peers = self.user_index.search(query_vec, k_users + 1)
        mask = peers[0] != uidx
        peer_idx = peers[0][mask][:k_users]
        peer_sim = sims[0][mask][:k_users].astype("float32")

        peer_rows = self.ratings_csr[peer_idx]
        high_mask = peer_rows >= threshold
        peer_rows_high = peer_rows.multiply(high_mask)

        scores = np.asarray(peer_sim @ peer_rows_high).ravel()

        seen = self.ratings_csr.getrow(uidx).indices
        scores[seen] = -np.inf

        if scores.max() == -np.inf:
            return []
        top = np.argpartition(-scores, top_k)[:top_k]
        top = top[np.argsort(-scores[top])]
        return [self.aidx_to_mal_id[int(i)] for i in top]

    def evaluate_user_based(self, ratings_df: pd.DataFrame) -> dict:
        threshold = self.config.relevant_threshold
        top_k = self.config.top_k

        relevant_df = ratings_df[ratings_df["rating"] >= threshold]

        users_evaluated = 0
        hit_users = 0
        recall_sum = 0.0
        precision_sum = 0.0
        ndcg_sum = 0.0

        for user_id, group in relevant_df.groupby("user_id"):
            uid = int(user_id)  # type: ignore[arg-type]
            if uid not in self.user_id_to_uidx:
                continue
            relevant_set = {int(a) for a in group["anime_id"].tolist()}
            rec_ids = self._user_based_recommend(uid)
            if not rec_ids:
                continue
            n_hits = len(set(rec_ids) & relevant_set)

            users_evaluated += 1
            hit_users += 1 if n_hits > 0 else 0
            recall_sum += n_hits / len(relevant_set)
            precision_sum += n_hits / top_k
            ndcg_sum += self._ndcg_at_k(rec_ids, relevant_set, top_k)

        if users_evaluated == 0:
            return {"users_evaluated": 0}

        return {
            "users_evaluated": users_evaluated,
            "k": top_k,
            "relevant_threshold": threshold,
            "hit_rate@k": hit_users / users_evaluated,
            "recall@k": recall_sum / users_evaluated,
            "precision@k": precision_sum / users_evaluated,
            "ndcg@k": ndcg_sum / users_evaluated,
        }

    def run(self, split_name: str, ratings_df: pd.DataFrame) -> EvaluationArtifact:
        out_dir = os.path.join(self.config.evaluation_dir, split_name)
        os.makedirs(out_dir, exist_ok=True)

        hybrid_metrics = self.evaluate_hybrid(ratings_df)
        content_metrics = self.evaluate_content()
        user_metrics = self.evaluate_user_based(ratings_df)

        hybrid_path = os.path.join(out_dir, EVAL_HYBRID_METRICS_FILE)
        content_path = os.path.join(out_dir, EVAL_CONTENT_METRICS_FILE)
        user_path = os.path.join(out_dir, EVAL_USER_METRICS_FILE)
        summary_path = os.path.join(out_dir, EVAL_SUMMARY_FILE)

        with open(hybrid_path, "w", encoding="utf-8") as f:
            json.dump(hybrid_metrics, f, indent=2)
        with open(content_path, "w", encoding="utf-8") as f:
            json.dump(content_metrics, f, indent=2)
        with open(user_path, "w", encoding="utf-8") as f:
            json.dump(user_metrics, f, indent=2)

        summary = {
            "split": split_name,
            "hybrid": hybrid_metrics,
            "content_based": content_metrics,
            "user_based": user_metrics,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return EvaluationArtifact(
            split_dir=out_dir,
            hybrid_metrics_path=hybrid_path,
            content_metrics_path=content_path,
            user_metrics_path=user_path,
            summary_path=summary_path,
        )
