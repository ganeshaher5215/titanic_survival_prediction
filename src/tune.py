import argparse
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from utils import load_titanic
from data_prep import basic_clean, train_test_split_titanic
from features import build_preprocessor, engineer_features

def main(args):
    df = load_titanic()
    df = basic_clean(df)
    if "Survived" not in df.columns:
        raise ValueError("Expected 'Survived' target in train.csv")

    df_eng = engineer_features(df)
    X_train, X_valid, y_train, y_valid = train_test_split_titanic(df_eng)

    pre = build_preprocessor(df)

    if args.model == "rf":
        clf = RandomForestClassifier(random_state=42)
        grid = {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [None, 6, 10],
            "clf__min_samples_split": [2, 5, 10],
        }
    elif args.model == "xgb":
        clf = XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False)
        grid = {
            "clf__n_estimators": [300, 500],
            "clf__max_depth": [3, 4, 5],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
        }
    else:
        raise ValueError("Supported models for tuning: rf|xgb")

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    print("Best CV score (F1):", gs.best_score_)

    if args.save:
        joblib.dump(gs.best_estimator_, args.save)
        print("Saved tuned model to", args.save)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="rf", help="rf|xgb")
    ap.add_argument("--save", type=str, default="models/model_tuned.pkl")
    args = ap.parse_args()
    main(args)
