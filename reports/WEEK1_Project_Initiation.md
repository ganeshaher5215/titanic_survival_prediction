# Week 1 (08–14 Aug 2025) — Project Initiation

**Problem Statement**  
Build an ML model to predict passenger survival on the Titanic using demographic and ticket information.

**Objectives**
1. Clean and preprocess the Titanic dataset.
2. Engineer meaningful features (Title, FamilySize, IsAlone, HasCabin).
3. Train, evaluate, and compare models with Accuracy and F1.
4. Deploy a simple Streamlit app for live predictions.
5. Document week-wise progress and results.

**Dataset**
- Kaggle: Titanic — Machine Learning from Disaster (`train.csv`, `test.csv`).

**Methodology Outline**
- EDA → Preprocess → Feature Engg → Baseline Models → Hyperparameter Tuning → Final Model → Deployment.

**Risks & Mitigation**
- Missing Ages: median or segmented imputation by Sex/Pclass.
- Class imbalance: use F1-score and threshold tuning.
- Overfitting: cross-validation and regularization.
