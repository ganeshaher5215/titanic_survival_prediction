# Week 4 (29 Aug – 04 Sep 2025) — Model Training (Phase 1)

**Action**
- Trained baseline models (LR, RF, GB, XGB) using `src/train.py`.

**Example Baseline Results** *(typical for Titanic; will vary)*
- LR: Acc ~ 0.80, F1 ~ 0.76
- RF: Acc ~ 0.83, F1 ~ 0.79
- GB: Acc ~ 0.82, F1 ~ 0.78
- XGB: Acc ~ 0.84, F1 ~ 0.80

**Observation**
- Tree ensembles outperform linear baseline.
- Proceed with RF/XGB for tuning.
