import argparse
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report
)
from utils import load_titanic
from data_prep import basic_clean, train_test_split_titanic
from features import engineer_features

def main(args):
    model = joblib.load(args.model)
    df = load_titanic()
    df = basic_clean(df)
    df_eng = engineer_features(df)
    X_train, X_valid, y_train, y_valid = train_test_split_titanic(df_eng)

    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_valid, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_valid, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y_valid, y_proba) if y_proba is not None else float("nan")
    cm = confusion_matrix(y_valid, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_valid, y_pred, zero_division=0))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    args = ap.parse_args()
    main(args)
