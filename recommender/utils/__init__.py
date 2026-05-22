import os
import yaml
import pandas as pd


def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            return config
    except Exception as e:
        print(e)
        raise e


def read_df(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
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