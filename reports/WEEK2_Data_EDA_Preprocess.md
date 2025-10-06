# Week 2 (15–21 Aug 2025) — Data Collection, EDA & Preprocessing

**Data Acquisition**
- Placed `train.csv` in `data/` (Kaggle).

**EDA Summary (what to examine)**
- Survival rate overall and by Sex, Pclass, Embarked.
- Distributions: Age, Fare; outliers.
- Correlations: target with key variables.

**Preprocessing**
- Handle Missing:
  - `Age`: median imputation.
  - `Embarked`: mode.
  - `Cabin`: sparse → derive `HasCabin`.
- Categorical Encoding: One-hot for `Sex`, `Embarked`, `Title`.
- Scaling: Standardize numeric (`Age`, `Fare`).
