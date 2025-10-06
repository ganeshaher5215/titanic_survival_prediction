import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def load_titanic(train_filename="train.csv"):
    train_path = os.path.join(DATA_DIR, train_filename)
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Missing {train_path}. Please place Kaggle's train.csv in the data/ folder."
        )
    df = pd.read_csv(train_path)
    return df

def load_optional_test(test_filename="test.csv"):
    test_path = os.path.join(DATA_DIR, test_filename)
    if not os.path.exists(test_path):
        return None
    return pd.read_csv(test_path)
