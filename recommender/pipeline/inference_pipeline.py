import os
import pandas as pd
from recommender.constants import *
from recommender.components.content_based import ContentBasedRecommender
from recommender.components.hybrid import HybridRecommender
from recommender.entity.config_entity import ContentBasedRecommenderConfig, HybridRecommenderConfig


class InferencePipeline:
    def __init__(
        self,
        cb_config: ContentBasedRecommenderConfig | None = None,
        hy_config: HybridRecommenderConfig | None = None,
    ):
        self.cb_config = cb_config or ContentBasedRecommenderConfig()
        self.hy_config = hy_config or HybridRecommenderConfig()
        self._preprocessed_dir = self.cb_config.preprocessed_dir
        self.content: ContentBasedRecommender | None = None
        self.hybrid: HybridRecommender | None = None

    def load(self) -> "InferencePipeline":
        """Load all pre-trained artifacts from disk. Must be called before recommend_*."""
        print("[InferencePipeline] Loading ContentBasedRecommender artifacts …")
        cb = ContentBasedRecommender(config=self.cb_config)
        cb._load_if_needed()
        self.content = cb
        print(f"  Content index loaded  ({cb.index.ntotal:,} anime vectors)")

        print("[InferencePipeline] Initialising HybridRecommender …")
        hy = HybridRecommender(config=self.hy_config)
        hy._content = cb  # share already-loaded CB to avoid double load
        self.hybrid = hy
        print("  HybridRecommender ready")

        return self

    def _ensure_loaded(self):
        if self.hybrid is None:
            self.load()

    def recommend_for_user(self, user_id: int) -> pd.DataFrame:
        self._ensure_loaded()
        print(f"[InferencePipeline] Hybrid recommendations for user_id={user_id}")
        return self.hybrid.recommend(user_id)

    def recommend_similar(self, anime_id: int, k: int = 10) -> pd.DataFrame:
        self._ensure_loaded()
        print(f"[InferencePipeline] Content-based recommendations for anime_id={anime_id}, k={k}")
        return self.content.recommend(anime_id, k=k)

    def run(self):
        self.load()
        anime_df = self.content.anime_df
        sample_anime = int(anime_df["anime_id"].iloc[0])
        print(f"\n[Demo] Content-based recs for anime_id={sample_anime}:")
        cb_recs = self.recommend_similar(sample_anime, k=5)
        print(cb_recs.to_string(index=False))

        # Pick a warm user (one who has rated at least one anime in CB)
        uar = self.hybrid._user_ratings
        warm_users = [uid for uid, ratings in uar.items() if ratings]
        if warm_users:
            demo_user = warm_users[0]
            print(f"\n[Demo] Hybrid recs for user_id={demo_user}:")
            hy_recs = self.recommend_for_user(demo_user)
            print(hy_recs.to_string(index=False))
        else:
            print("\n[Demo] No warm users found — skipping hybrid demo.")


if __name__ == "__main__":
    InferencePipeline().run()
