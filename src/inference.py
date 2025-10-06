import joblib
import pandas as pd
from features import engineer_features

def predict_from_dataframe(model_path: str, df: pd.DataFrame):
    pipe = joblib.load(model_path)
    df_eng = engineer_features(df)
    preds = pipe.predict(df_eng)
    proba = pipe.predict_proba(df_eng)[:,1] if hasattr(pipe, "predict_proba") else None
    out = df.copy()
    out["prediction"] = preds
    if proba is not None:
        out["probability"] = proba
    return out
