import os
from typing import Optional
import yaml
import pandas as pd
from src.constants import COLS_TO_DROP_CONFIG

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not in the given path")
        with open(file_path,"r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            return config
    
    except Exception as e:
        print(e)
        raise e

def read_df(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not in the given path")
        df = pd.read_csv(file_path)
        return df
    
    except Exception as e:
        print(e)
        raise e
    
def save_df(df: pd.DataFrame, file_path: str):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        df.to_csv(file_path, index=False)
    except Exception as e:
        print(e)
        raise e
    
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if df.duplicated().sum() > 0:
            print(f"Found {df.duplicated().sum()} duplicate rows. Dropping duplicates.")
        return df.drop_duplicates()
    except Exception as e:
        print(e)
        raise e

def drop_unnecessary_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    try:
        return df.drop(columns=[col for col in cols if col in df.columns])
    except Exception as e:
        print(e)
        raise e

def get_cols_to_drop(stage: str, dataset: str, config_path: Optional[str] = None) -> list:
    try:
        yaml_path = config_path or COLS_TO_DROP_CONFIG
        config = read_yaml(yaml_path) or {}
        stage_config = config.get(stage, {}) if isinstance(config, dict) else {}
        cols = stage_config.get(dataset, [])
        return cols or []
    except Exception as e:
        print(e)
        raise e