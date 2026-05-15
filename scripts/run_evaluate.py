import os
import pandas as pd

from src.components.evaluation import Evaluation
from src.constants import (
    SPLIT_FILE_NAME,
    RATINGS_SPLIT_FILE_NAME,
    CONTENT_INDEX_FILE,
    CONTENT_MAL_TO_FAISS_FILE,
    CONTENT_FAISS_TO_MAL_FILE,
    USER_INDEX_FILE,
    USER_ID_TO_FAISS_FILE,
    USER_ID_TO_VECTOR_FILE,
    HYBRID_RATINGS_MATRIX_FILE,
)
from src.entity.config_entity import (
    EvaluationConfig,
    ContentBasedRecommenderConfig,
    UserBasedRecommenderConfig,
    HybridRecommenderConfig,
    DataPreprocessingConfig,
)
from src.entity.artifact_entity import (
    ContentBasedRecommenderArtifact,
    UserBasedRecommenderArtifact,
    HybridRecommenderArtifact,
)
from src.tracker import MLflowTracker


if __name__ == "__main__":
    pre_cfg = DataPreprocessingConfig()
    content_cfg = ContentBasedRecommenderConfig()
    user_cfg = UserBasedRecommenderConfig()
    hybrid_cfg = HybridRecommenderConfig()
    eval_cfg = EvaluationConfig()

    content_artifact = ContentBasedRecommenderArtifact(
        content_based_dir=content_cfg.content_based_dir,
        index_path=os.path.join(content_cfg.content_based_dir, CONTENT_INDEX_FILE),
        mal_to_faiss_path=os.path.join(content_cfg.content_based_dir, CONTENT_MAL_TO_FAISS_FILE),
        faiss_to_mal_path=os.path.join(content_cfg.content_based_dir, CONTENT_FAISS_TO_MAL_FILE),
    )
    user_artifact = UserBasedRecommenderArtifact(
        user_based_dir=user_cfg.user_based_dir,
        index_path=os.path.join(user_cfg.user_based_dir, USER_INDEX_FILE),
        user_id_to_faiss_path=os.path.join(user_cfg.user_based_dir, USER_ID_TO_FAISS_FILE),
        user_id_to_vector_path=os.path.join(user_cfg.user_based_dir, USER_ID_TO_VECTOR_FILE),
    )
    hybrid_artifact = HybridRecommenderArtifact(
        hybrid_dir=hybrid_cfg.hybrid_dir,
        ratings_matrix_path=os.path.join(hybrid_cfg.hybrid_dir, HYBRID_RATINGS_MATRIX_FILE),
    )

    train_anime_df = pd.read_csv(os.path.join(pre_cfg.train_data_dir, SPLIT_FILE_NAME))
    val_ratings_df = pd.read_csv(os.path.join(pre_cfg.val_data_dir, RATINGS_SPLIT_FILE_NAME))
    test_ratings_df = pd.read_csv(os.path.join(pre_cfg.test_data_dir, RATINGS_SPLIT_FILE_NAME))

    evaluator = Evaluation(eval_cfg, content_artifact, user_artifact, hybrid_artifact, train_anime_df)

    with MLflowTracker() as tracker:
        tracker.log_config("content_based", content_cfg)
        tracker.log_config("user_based", user_cfg)
        tracker.log_config("hybrid", hybrid_cfg)
        tracker.log_config("evaluation", eval_cfg)

        val_artifact = evaluator.run("val", val_ratings_df)
        test_artifact = evaluator.run("test", test_ratings_df)

        tracker.log_summary("val", val_artifact.summary_path)
        tracker.log_summary("test", test_artifact.summary_path)

        # Per-recommender JSONs (small)
        for split, art in (("val", val_artifact), ("test", test_artifact)):
            tracker.log_json(art.hybrid_metrics_path, f"evaluation/{split}")
            tracker.log_json(art.content_metrics_path, f"evaluation/{split}")
            tracker.log_json(art.user_metrics_path, f"evaluation/{split}")

    print(val_artifact)
    print(test_artifact)
