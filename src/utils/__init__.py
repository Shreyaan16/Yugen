import os
import yaml
import pandas as pd

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