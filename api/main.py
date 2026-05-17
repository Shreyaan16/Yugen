import os

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from api.controllers.authControllers import Base, engine
from api.models import authModels  # noqa: F401  (register tables on Base)
from api.routes.animeRoutes import router as anime_router
from api.routes.authRoutes import router as auth_router
from src.components.recommender import ContentBasedRecommender, HybridRecommender
from src.constants import (
    ANIME_FILE_NAME,
    CONTENT_FAISS_TO_MAL_FILE,
    CONTENT_INDEX_FILE,
    CONTENT_MAL_TO_FAISS_FILE,
    HYBRID_RATINGS_MATRIX_FILE,
    SPLIT_FILE_NAME,
    USER_ID_TO_FAISS_FILE,
    USER_ID_TO_VECTOR_FILE,
    USER_INDEX_FILE,
)
from src.entity.artifact_entity import (
    ContentBasedRecommenderArtifact,
    HybridRecommenderArtifact,
    UserBasedRecommenderArtifact,
)
from src.entity.config_entity import (
    ContentBasedRecommenderConfig,
    DataPreprocessingConfig,
    HybridRecommenderConfig,
    UserBasedRecommenderConfig,
)

Base.metadata.create_all(bind=engine)

# Attach DB-level FK from user_ratings.anime_id -> anime(id). The anime table is
# managed outside SQLAlchemy, so we can't declare this on the model.
with engine.begin() as _conn:
    _conn.execute(
        text(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'user_ratings_anime_id_fkey'
                ) THEN
                    ALTER TABLE user_ratings
                    ADD CONSTRAINT user_ratings_anime_id_fkey
                    FOREIGN KEY (anime_id) REFERENCES anime(id) ON DELETE CASCADE;
                END IF;
            END$$;
            """
        )
    )

_pre_cfg = DataPreprocessingConfig()
_content_cfg = ContentBasedRecommenderConfig()
_user_cfg = UserBasedRecommenderConfig()
_hybrid_cfg = HybridRecommenderConfig()

_train_df = pd.read_csv(os.path.join(_pre_cfg.train_data_dir, SPLIT_FILE_NAME))
content_recommender = ContentBasedRecommender(_content_cfg).load(_train_df)

_content_artifact = ContentBasedRecommenderArtifact(
    content_based_dir=_content_cfg.content_based_dir,
    index_path=os.path.join(_content_cfg.content_based_dir, CONTENT_INDEX_FILE),
    mal_to_faiss_path=os.path.join(_content_cfg.content_based_dir, CONTENT_MAL_TO_FAISS_FILE),
    faiss_to_mal_path=os.path.join(_content_cfg.content_based_dir, CONTENT_FAISS_TO_MAL_FILE),
)
_user_artifact = UserBasedRecommenderArtifact(
    user_based_dir=_user_cfg.user_based_dir,
    index_path=os.path.join(_user_cfg.user_based_dir, USER_INDEX_FILE),
    user_id_to_faiss_path=os.path.join(_user_cfg.user_based_dir, USER_ID_TO_FAISS_FILE),
    user_id_to_vector_path=os.path.join(_user_cfg.user_based_dir, USER_ID_TO_VECTOR_FILE),
)
_hybrid_artifact = HybridRecommenderArtifact(
    hybrid_dir=_hybrid_cfg.hybrid_dir,
    ratings_matrix_path=os.path.join(_hybrid_cfg.hybrid_dir, HYBRID_RATINGS_MATRIX_FILE),
)
_anime_df = pd.read_csv(
    os.path.join(_pre_cfg.raw_data_dir, ANIME_FILE_NAME),
    usecols=["MAL_ID", "Name"],
)
_anime_titles = {int(mid): str(name) for mid, name in zip(_anime_df["MAL_ID"], _anime_df["Name"])}
hybrid_recommender = HybridRecommender(_hybrid_cfg).load(
    _content_artifact, _user_artifact, _hybrid_artifact, _anime_titles
)

app = FastAPI(title="YuGen API")
app.state.content_recommender = content_recommender
app.state.hybrid_recommender = hybrid_recommender

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(anime_router, tags=["anime"])


@app.get("/health")
def health():
    return {"status": "ok"}
