import os
import pandas as pd

from src.constants import (
    SPLIT_FILE_NAME,
    RATINGS_SPLIT_FILE_NAME,
    ANIME_FILE_NAME,
    CONTENT_INDEX_FILE,
    CONTENT_MAL_TO_FAISS_FILE,
    CONTENT_FAISS_TO_MAL_FILE,
    USER_INDEX_FILE,
    USER_ID_TO_FAISS_FILE,
    USER_ID_TO_VECTOR_FILE,
    HYBRID_RATINGS_MATRIX_FILE,
)
from src.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    ContentBasedRecommenderConfig,
    UserBasedRecommenderConfig,
    HybridRecommenderConfig,
)
from src.entity.artifact_entity import (
    ContentBasedRecommenderArtifact,
    UserBasedRecommenderArtifact,
    HybridRecommenderArtifact,
)
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.recommender import (
    ContentBasedRecommender,
    UserBasedRecommender,
    HybridRecommender,
)


# ============================================================
# Configs
# ============================================================
ingestion_cfg = DataIngestionConfig()
preprocessing_cfg = DataPreprocessingConfig()
content_cfg = ContentBasedRecommenderConfig()
user_cfg = UserBasedRecommenderConfig()
hybrid_cfg = HybridRecommenderConfig()


# ============================================================
# Step 1: Data ingestion
# ============================================================

# --- RUN ---
DataIngestion(ingestion_cfg).run()


# ============================================================
# Step 2: Data preprocessing
# ============================================================

# --- RUN ---
DataPreprocessing(preprocessing_cfg).run()


# ============================================================
# Step 3: Content-based recommender 
# ============================================================

# --- RUN ---
train_df = pd.read_csv(os.path.join(preprocessing_cfg.train_data_dir, SPLIT_FILE_NAME))
content_artifact = ContentBasedRecommender(content_cfg).run(train_df)
print(content_artifact)

# --- LOAD ---
# content_artifact = ContentBasedRecommenderArtifact(
#     content_based_dir=content_cfg.content_based_dir,
#     index_path=os.path.join(content_cfg.content_based_dir, CONTENT_INDEX_FILE),
#     mal_to_faiss_path=os.path.join(content_cfg.content_based_dir, CONTENT_MAL_TO_FAISS_FILE),
#     faiss_to_mal_path=os.path.join(content_cfg.content_based_dir, CONTENT_FAISS_TO_MAL_FILE),
# )


# ============================================================
# Step 4: User-based recommender 
# ============================================================

# --- RUN ---
ratings_df = pd.read_csv(os.path.join(preprocessing_cfg.train_data_dir, RATINGS_SPLIT_FILE_NAME))
user_artifact = UserBasedRecommender(user_cfg).run(ratings_df, content_artifact)
print(user_artifact)

# --- LOAD ---
# user_artifact = UserBasedRecommenderArtifact(
#     user_based_dir=user_cfg.user_based_dir,
#     index_path=os.path.join(user_cfg.user_based_dir, USER_INDEX_FILE),
#     user_id_to_faiss_path=os.path.join(user_cfg.user_based_dir, USER_ID_TO_FAISS_FILE),
#     user_id_to_vector_path=os.path.join(user_cfg.user_based_dir, USER_ID_TO_VECTOR_FILE),
# )


# ============================================================
# Step 5: Hybrid recommender 
# ============================================================
ratings_df = pd.read_csv(os.path.join(preprocessing_cfg.train_data_dir, RATINGS_SPLIT_FILE_NAME))
anime_df = pd.read_csv(
    os.path.join(preprocessing_cfg.raw_data_dir, ANIME_FILE_NAME),
    usecols=["MAL_ID", "Name"],
)
anime_titles: dict[int, str] = {
    int(mal_id): str(name) for mal_id, name in zip(anime_df["MAL_ID"], anime_df["Name"])
}

# --- RUN ---
hybrid_artifact = HybridRecommender(hybrid_cfg).run(
    ratings_df, content_artifact, user_artifact, anime_titles
)
print(hybrid_artifact)

# --- LOAD ---
# hybrid_artifact = HybridRecommenderArtifact(
#     hybrid_dir=hybrid_cfg.hybrid_dir,
#     ratings_matrix_path=os.path.join(hybrid_cfg.hybrid_dir, HYBRID_RATINGS_MATRIX_FILE),
# )


# ============================================================
# Step 6: Try a recommendation
# ============================================================
hybrid = HybridRecommender(hybrid_cfg).load(
    content_artifact, user_artifact, hybrid_artifact, anime_titles
)
print(hybrid.recommend(user_id=0, top_k=10))


# ============================================================
# Step 7: Evaluation on val then test
# ============================================================
from src.entity.config_entity import EvaluationConfig
from src.components.evaluation import Evaluation

eval_cfg = EvaluationConfig()
train_anime_df = pd.read_csv(os.path.join(preprocessing_cfg.train_data_dir, SPLIT_FILE_NAME))

evaluator = Evaluation(
    eval_cfg, content_artifact, user_artifact, hybrid_artifact, train_anime_df
)

val_ratings_df = pd.read_csv(os.path.join(preprocessing_cfg.val_data_dir, RATINGS_SPLIT_FILE_NAME))
val_artifact = evaluator.run("val", val_ratings_df)
print(val_artifact)

test_ratings_df = pd.read_csv(os.path.join(preprocessing_cfg.test_data_dir, RATINGS_SPLIT_FILE_NAME))
test_artifact = evaluator.run("test", test_ratings_df)
print(test_artifact)
