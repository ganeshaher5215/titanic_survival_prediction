import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_FEATURES = [
    "Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Cabin","Ticket","Name"
]

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Standardize column names just in case
    df.columns = [c.strip() for c in df.columns]
    # Drop obviously empty rows
    df = df.dropna(how="all")
    return df

def train_test_split_titanic(df, test_size=0.2, random_state=42):
    target = "Survived"
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_valid, y_train, y_valid
