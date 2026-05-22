import os
import pandas as pd
from recommender.constants import *
from recommender.components.content_based import ContentBasedRecommender
from recommender.components.user_based import UserBasedRecommender
from recommender.components.hybrid import HybridRecommender
from recommender.entity.config_entity import (ContentBasedRecommenderConfig, UserBasedRecommenderConfig, HybridRecommenderConfig)


class InferencePipeline:
    def __init__(
        self,
        cb_config: ContentBasedRecommenderConfig | None = None,
        ub_config: UserBasedRecommenderConfig | None = None,
        hy_config: HybridRecommenderConfig | None = None,
    ):
        self.cb_config = cb_config or ContentBasedRecommenderConfig()
        self.ub_config = ub_config or UserBasedRecommenderConfig()
        self.hy_config = hy_config or HybridRecommenderConfig()
        self._preprocessed_dir = self.cb_config.preprocessed_dir
        self.content: ContentBasedRecommender | None = None
        self.user_based: UserBasedRecommender | None = None
        self.hybrid: HybridRecommender | None = None

    def load(self) -> "InferencePipeline":
        """Load all pre-trained artifacts from disk. Must be called before recommend_*."""
        print("[InferencePipeline] Loading ContentBasedRecommender artifacts …")
        anime_df = pd.read_csv(os.path.join(self._preprocessed_dir, ANIME_FILE_NAME))
        cb = ContentBasedRecommender(config=self.cb_config)
        cb.load(anime_df)
        self.content = cb
        print(f"  ✓ Content index loaded  ({cb.index.ntotal:,} anime vectors)")

        print("[InferencePipeline] Loading UserBasedRecommender artifacts …")
        ub = UserBasedRecommender(config=self.ub_config)
        ub._load_if_needed()
        self.user_based = ub
        print(f"  ✓ User index loaded     ({ub.index.ntotal:,} user vectors, avg_idx={ub.avg_idx})")

        print("[InferencePipeline] Initialising HybridRecommender …")
        hy = HybridRecommender(config=self.hy_config)
        # HybridRecommender loads its own sub-recommender artifacts from disk
        # via _load_if_needed() on first call to recommend() — no manual wiring needed.
        self.hybrid = hy
        print("   HybridRecommender ready (artifacts loaded on first call)")

        return self

    def _ensure_loaded(self):
        if self.hybrid is None:
            self.load()


    def recommend_for_user(self, user_id: int) -> pd.DataFrame:
        self._ensure_loaded()
        print(f"[InferencePipeline] Hybrid recommendations for user_id={user_id}")
        recs = self.hybrid.recommend(user_id)
        return recs

    def recommend_similar(self, anime_id: int, k: int = 10) -> pd.DataFrame:
        self._ensure_loaded()
        print(f"[InferencePipeline] Content-based recommendations for anime_id={anime_id}, k={k}")
        recs = self.content.recommend(anime_id, k=k)
        return recs

    def similar_users(self, user_id: int, k: int = 10) -> pd.DataFrame:
        self._ensure_loaded()
        return self.user_based.similar_users(user_id, k=k)

    def run(self):
        self.load()
        anime_df = self.content.anime_df
        sample_anime = int(anime_df["anime_id"].iloc[0])
        print(f"\n[Demo] Content-based recs for anime_id={sample_anime}:")
        cb_recs = self.recommend_similar(sample_anime, k=5)
        print(cb_recs.to_string(index=False))
        warm_users = [
            uid for uid, idx in self.user_based.user_id_to_idx.items()
            if idx != self.user_based.avg_idx
        ]
        if warm_users:
            demo_user = warm_users[0]
            print(f"\n[Demo] Hybrid recs for user_id={demo_user}:")
            hy_recs = self.recommend_for_user(demo_user)
            print(hy_recs.to_string(index=False))
        else:
            print("\n[Demo] No warm users found — skipping hybrid demo.")


if __name__ == "__main__":
    InferencePipeline().run()
