import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import re

# Feature engineering helpers
def extract_title(name: str) -> str:
    if not isinstance(name, str):
        return "Unknown"
    m = re.search(r",\s*([^\.]+)\.", name)
    return m.group(1).strip() if m else "Unknown"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Titles
    if "Name" in df.columns:
        df["Title"] = df["Name"].apply(extract_title)
    else:
        df["Title"] = "Unknown"

    # Family size & IsAlone
    sibsp = df["SibSp"] if "SibSp" in df.columns else 0
    parch = df["Parch"] if "Parch" in df.columns else 0
    df["FamilySize"] = sibsp + parch + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Cabin known flag
    df["HasCabin"] = df["Cabin"].notna().astype(int) if "Cabin" in df.columns else 0

    # Drop high-cardinality/noisy raw fields
    for col in ["Ticket", "Cabin", "Name"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    # Drop PassengerId if present
    if "PassengerId" in df.columns:
        df = df.drop(columns=["PassengerId"])
        
    return df

def build_preprocessor(df: pd.DataFrame):
    df = engineer_features(df)
    num_feats = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # Survived might not be present (inference/test), filter it out
    num_feats = [c for c in num_feats if c != "Survived"]
    cat_feats = [c for c in df.columns if c not in num_feats + ["Survived"]]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_feats),
            ("cat", categorical_transformer, cat_feats)
        ]
    )
    return preprocessor
