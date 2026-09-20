# Model selection for deployment

## Decision
**Deployed model: LightGBM, "leakage-safe baseline" (CLEAN_P2), seed 42, 57 features, original class distribution, no weighting/resampling, no calibration.**
Artifacts are created by `build_model_artifacts.py` into `models/` (`final_model.txt`, `preprocessing.json`, `model_metadata.json`).

| Item | Value |
|---|---|
| Model | LightGBM (binary), 14 trees (early stopping on validation PR-AUC), `num_leaves=31, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_alpha=reg_lambda=0.1, random_state=42` |
| Features | 57 = 11 numeric + 46 categorical, incl. baseline `cognitive_status`; fixed order = `models/preprocessing.json -> feature_order` (alphabetical, `cognitive_status` last) |
| Numeric preprocessing | blank -> TRAIN median (fitted on the 18,090 training rows only); no scaling |
| Categorical preprocessing | blank/unknown -> explicit `Missing` level; category vocabularies from TRAIN only; native pandas categoricals in LightGBM |
| Engineered features | none (the targeted feature-engineering experiment was not completed and nothing from it is used) |
| Target | project's 3-year conversion outcome: a new Alzheimer's etiologic diagnosis within 3 years of the baseline visit in participants without it at baseline |
| Split / training data | participant-level split unchanged: train 18,090 / validation 3,876 / test 3,877 (13.34% / 13.34% / 13.36% converters); `model_ready/ad_conversion_3yr_anchor_dataset.csv` with the upstream imputation undone (cells flagged `*_was_imputed` are blank again) |
| Threshold | 0.1616 = F1-maximising threshold on validation, frozen before the test set was read (validation precision 0.391 / recall 0.584; test precision 0.390 / recall 0.541) |
| Calibration | none - raw probabilities (Brier 0.101 validation / 0.101 test) |
| Performance, this artifact (seed 42) | validation PR-AUC **0.4337**, ROC-AUC 0.7955; test PR-AUC **0.4223**, ROC-AUC 0.7900 |
| Performance, 3-seed mean (42/123/2026) | validation PR-AUC 0.4326, test 0.4269 (XGBoost, same pipeline: 0.4340 / 0.4307) |

## Why this model (according to the existing results)
1. **It is the latest completed modeling stage.** The leakage-safe baseline rebuilt the benchmark with strictly train-only preprocessing. Its selection rule (highest mean validation PR-AUC; inside the 0.01 noise band prefer the smaller train-validation gap) picked LightGBM over XGBoost (0.4326 vs 0.4340 validation, i.e. within noise).
2. **It performs the same as the earlier saved model.** The clean pipeline differs from the transductive one by +0.003 to +0.004 validation PR-AUC (inside the noise band).
3. **It can be reproduced at inference time.** The older saved artifact (`modeling/final_modeling/models/final_model.txt`: LightGBM baseline, validation PR-AUC 0.4371, test 0.4184, threshold 0.2, native-NaN handling) was trained on values produced by an upstream imputation fitted on all participants; that imputer is not a saved inference component, so fields a user leaves blank could not be filled the way the model saw them in training. The clean model handles blanks with a trivial, saved, train-only rule (TRAIN median / `Missing`).

## Candidates that were inspected but not deployed
* `modeling/final_modeling` LightGBM baseline (above) - superseded by the leakage-safe rebuild.
* Imbalance variants (class weighting, SMOTENC): no gain / lower PR-AUC; not used.
* Targeted feature engineering (age/cognition/functional/clinical interactions): the run was **not completed** - it stopped on a bookkeeping error before the selection step and before the test set was opened, and no outputs were saved. Its interim validation-only numbers did not show any arm reaching the pre-registered +0.01 adoption threshold. Per the instructions, the established baseline is used.

## Provenance notes (please read)
* The leakage-safe baseline results were computed in the assistant's working environment and were **not saved under `E:\ALZ\modeling\`** (no `leakage_safe_baseline` folder exists). `build_model_artifacts.py` therefore rebuilds the model deterministically from the project data and self-checks against the recorded numbers (14 trees, PR-AUC 0.4337 / 0.4223, threshold 0.1616 -> "matches the recorded selected model: True").
  When built in the assistant's environment the model's validation predictions matched that stage's stored predictions to 3e-8 (max absolute difference).
* `tests/test_inference.py` re-checks, on every validation row, that the inference-time preprocessing equals the training-time preprocessing (max difference < 1e-9) and that the validation PR-AUC computed through the inference path equals the recorded value.
* One validation record has height 68.7 in and weight 400 lb but a blank BMI in the training data; the UI derives BMI from height and weight, which the training pipeline did not do for that record (1 of 25,843).
* Known model behaviour that affects interpretation (form-era missingness, compressed probabilities, rare-level artifacts) is described in `README.md`.
