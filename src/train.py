import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline

from utils import load_titanic
from data_prep import basic_clean, train_test_split_titanic
from features import engineer_features, build_preprocessor

def get_model(name: str):
    name = name.lower()
    if name == "lr":
        return LogisticRegression(max_iter=200)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=42)
    if name == "gb":
        return GradientBoostingClassifier(random_state=42)
    if name == "svm":
        return SVC(kernel="rbf", probability=True, random_state=42)
    if name == "xgb":
        return XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.8, random_state=42, eval_metric="logloss"
        )
    raise ValueError(f"Unknown model: {name}")

def main(args):
    df = load_titanic()
    df = basic_clean(df)
    if "Survived" not in df.columns:
        raise ValueError("Expected 'Survived' target in train.csv")

    # Feature engineering (for column discovery)
    df_eng = engineer_features(df)

    X_train, X_valid, y_train, y_valid = train_test_split_titanic(df_eng)

    preprocessor = build_preprocessor(df)
    model = get_model(args.model)

    pipe = Pipeline(steps=[("pre", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)

    # Metrics
    y_pred = pipe.predict(X_valid)
    y_proba = pipe.predict_proba(X_valid)[:, 1] if hasattr(pipe, "predict_proba") else None

    acc = accuracy_score(y_valid, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_valid, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y_valid, y_proba) if y_proba is not None else float("nan")

    print(f"Model: {args.model}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")

    if args.save:
        joblib.dump(pipe, args.save)
        print(f"Saved model to {args.save}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rf", help="lr|rf|gb|xgb|svm")
    parser.add_argument("--save", type=str, default="models/model.pkl")
    args = parser.parse_args()
    main(args)
