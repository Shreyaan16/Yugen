import os
import pandas as pd

from src.components.recommender import ContentBasedRecommender
from src.constants import SPLIT_FILE_NAME
from src.entity.config_entity import ContentBasedRecommenderConfig, DataPreprocessingConfig

if __name__ == "__main__":
    pre_cfg = DataPreprocessingConfig()
    train_df = pd.read_csv(os.path.join(pre_cfg.train_data_dir, SPLIT_FILE_NAME))
    artifact = ContentBasedRecommender(ContentBasedRecommenderConfig()).run(train_df)
    print(artifact)
