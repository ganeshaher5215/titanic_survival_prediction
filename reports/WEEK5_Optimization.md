# Week 5 (05–11 Sep 2025) — Model Optimization

**Method**
- `GridSearchCV` over RF/XGB with 5-fold Stratified CV (scoring = F1).

**Typical Tuned Gains** *(illustrative)*
- RF tuned: Acc ~ 0.85, F1 ~ 0.82
- XGB tuned: Acc ~ 0.86, F1 ~ 0.83

**Outcome**
- Selected best estimator (usually XGB on this dataset). Saved as `models/model_rf_tuned.pkl` or `model_xgb_tuned.pkl`.
- Documented feature importance.
