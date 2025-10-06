# Titanic Survival Prediction — Final Mini Project (Complete Package)

This repository contains a complete, week-by-week implementation plan,
codebase, and deployment-ready app for the **Titanic Survival Prediction** project.

> **What you need to run it**: Download Kaggle's Titanic dataset and place the CSVs under `data/`.
> Required files:
> - `data/train.csv`
> - `data/test.csv` (optional, for Kaggle-style holdout/testing)

## Quickstart

1. **Create & activate a virtual environment** (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Put Kaggle CSVs into `data/`:
   - `data/train.csv` (required)
   - `data/test.csv` (optional)
4. **Train baseline & save model**:
   ```bash
   python src/train.py --model rf --save models/model_rf.pkl
   ```
   Or run hyperparameter tuning (takes longer):
   ```bash
   python src/tune.py --model rf --save models/model_rf_tuned.pkl
   ```
5. **Evaluate**:
   ```bash
   python src/evaluate.py --model models/model_rf_tuned.pkl
   ```
6. **Run the app**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Project Structure
```
titanic_survival_prediction/
├─ app/
│  └─ streamlit_app.py
├─ data/
│  └─ README_data.md
├─ models/
├─ reports/
│  ├─ WEEK1_Project_Initiation.md
│  ├─ WEEK2_Data_EDA_Preprocess.md
│  ├─ WEEK3_Features_Model_Selection.md
│  ├─ WEEK4_Training_Phase1.md
│  ├─ WEEK5_Optimization.md
│  ├─ WEEK6_Final_Model_and_Docs.md
│  ├─ WEEK7_Deployment_Submission.md
│  └─ WEEK8_Final_Review.md
├─ src/
│  ├─ data_prep.py
│  ├─ features.py
│  ├─ train.py
│  ├─ tune.py
│  ├─ evaluate.py
│  ├─ inference.py
│  └─ utils.py
├─ requirements.txt
└─ README.md  ← (this file)
```

## Notes
- This package is self-contained **except the dataset**. Scripts will guide you if files are missing.
- You can switch models via CLI flags: `--model lr|rf|gb|xgb|svm`.
- Evaluation metrics: **Accuracy**, **Precision**, **Recall**, **F1-score**, **ROC-AUC**.
- The Streamlit app accepts manual inputs and also supports CSV batch inference.

Happy learning & shipping!
