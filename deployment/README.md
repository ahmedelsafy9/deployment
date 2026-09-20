# AD 3-year conversion model - local inference demo

A small Streamlit app around the project's **final selected model** (LightGBM, leakage-safe baseline, seed 42, 57 features).
You enter 13 baseline items; the app returns the model's estimated probability of the project's **3-year conversion outcome** and the
features that contributed most to that estimate (TreeSHAP). Research demonstration only - **not a medical diagnosis**.
Everything runs locally: no external APIs, no cloud, nothing entered is stored or logged (Streamlit usage statistics are switched off in `.streamlit/config.toml`).

## Run

```
cd E:\ALZ\deployment
pip install -r requirements.txt
python build_model_artifacts.py      # once: writes models/final_model.txt + preprocessing.json + model_metadata.json (skip if models/ is already populated)
streamlit run app.py
```
`build_model_artifacts.py` reads `E:\ALZ\model_ready\ad_conversion_3yr_anchor_dataset.csv` (override with the `ALZ_DIR` or `ANCHOR_CSV` environment variable),
reproduces the selected model deterministically (same data, split, seed, hyper-parameters - no new experiment) and checks itself against the recorded numbers
(14 trees, validation PR-AUC 0.4337, test PR-AUC 0.4223, threshold 0.1616); it prints `matches the recorded selected model: True`.
If Streamlit does not start on very new Python versions (e.g. 3.14), use a Python 3.12/3.13 virtual environment.

## Tests
```
python tests/test_inference.py     # input validation, missing values, one-input-at-a-time changes, exact equality with the training-time preprocessing on every validation row
python tests/test_app.py           # headless UI test: widgets render, Predict works, SHAP chart appears, changing a widget changes the prediction
```

## Files
| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `inference.py` | `predict(input_data)` - loads the saved model + train-only preprocessing, validates input, returns probability / percentage / operating-threshold label / model name+version / SHAP explanation |
| `ui_spec.py` | user-friendly labels, help texts and the exact list of exposed fields |
| `build_model_artifacts.py` | creates `models/*` from the project data (deterministic) |
| `models/final_model.txt` | LightGBM booster (14 trees) |
| `models/preprocessing.json` | feature order, TRAIN medians, TRAIN category vocabularies, reference ranges |
| `models/model_metadata.json` | metrics, threshold, calibration deciles, target definition, versions |
| `models/shap_importance_final_model.csv` | mean \|SHAP\| ranking used to choose the UI inputs |
| `MODEL_SELECTION.md` | which model was selected and why |

## Python API
```python
from inference import predict
r = predict({"age": 72, "cognitive_status": "MCI", "education_years": 16, "height_in": 66, "weight_lb": 150, "bp_systolic": 130,
             "independence_level": "Able to live independently", "hypertension": "Absent", "parkinsonism_pd": "Absent"})
r["probability"], r["percentage"], r["risk_category"], r["model_name"], r["explanation"]
```
Keys are model feature names; `None` / `""` / `"Unknown / Not provided"` = not provided. Categories must be values seen in training (an error lists the allowed ones).

## Fields exposed in the UI (13 entered + 2 derived)
Chosen from the final model's SHAP ranking (`models/shap_importance_final_model.csv`), keeping only items a person can realistically enter.

| UI field | Model feature | SHAP rank | Unknown allowed? |
|---|---|---:|---|
| Age * | `age` | 2 | no (required) |
| Cognitive status * | `cognitive_status` | 1 | no (required; never blank in training) |
| Functional independence * | `independence_level` | 4 | no (its 'Missing' level had only 21 training rows) |
| Years of education | `education_years` | 12 | yes (median) |
| Height, Weight (cm/kg or in/lb) | `height_in`, `weight_lb` -> **derived** `bmi` (rank 6) and `obesity_bmi30plus` | 11 (weight) | yes (BMI unknown if either is missing) |
| Systolic blood pressure | `bp_systolic` | 10 | yes (median) |
| Hypertension * | `hypertension` | 7 | no (61-88-row 'Missing' levels are unreliable) |
| Hearing functionally normal? | `hearing_normal_1yes` | 8 | yes |
| Parkinsonism * | `parkinsonism_pd` | 9 | no (its 'Missing' level had 61 rows and behaves like 'Yes') |
| Bowel incontinence | `bowel_incontinence` | 3 | yes - see note |
| Urinary incontinence | `urinary_incontinence` | 14 | yes - see note |
| Recent cancer | `cancer_recent` | 5 | yes - see note |

Not exposed: `rheumatoid_arthritis` (rank 13; obscure "not applicable" levels, same newer-form issue) and 41 lower-ranked features.

**How the other features are handled (nothing is invented):** all 42+ features that are not entered use the trained pipeline's missing-value handling - numeric features get the TRAIN median,
categorical features get the explicit `Missing` category, exactly as during training. The result page lists which features were entered, derived, or not entered.
BMI = 703 x weight(lb) / height(in)^2 and the obesity flag (BMI >= 30) are derived deterministically, as they are in the training data (0 inconsistencies among observed records).

## Things to know before interpreting a number
* **Newer-form items.** Bowel incontinence, urinary incontinence, recent cancer (and seven other items that are not shown) were blank almost only for older (pre-2015) baseline records, which had about twice the conversion rate (17% vs 8%).
  Leaving them unknown makes the model treat the record like an older-form record and raises the estimate (for a 72-year-old with MCI: 15.3% with the three items marked "No" vs about 27% with all three unknown). The app warns and shows the comparison.
* **Compressed, uncalibrated probabilities.** The model was selected for ranking (PR-AUC), early-stopped at 14 trees, and is not recalibrated. Predictions lie between about 8% and 44%; on the test set the top predicted decile averages 30% but converts at 51%,
  and participants with normal cognition average 10.5% predicted vs 7.3% observed. Use the number as a relative score, not a calibrated risk. In newer (2015+) records the model averages 10.7% vs 7.7% observed.
* **Operating threshold.** "Higher / Lower predicted probability" uses the project's validation-selected F1-maximising threshold (16.2%); on the test set it flags 19% of participants (precision 0.39, recall 0.54). It is not a clinical cut-off.
* **Cognitive status is not monotone in the model.** Dementia at baseline (in a cohort without an Alzheimer's diagnosis at baseline) converted less often than MCI in the training data (14% vs 38%), and the model reflects that.
* Input ranges: impossible values are rejected (age 18-110, education 0-30 years, systolic BP 50-300, BMI 8-100); values outside the range covering 98% of the training data trigger a warning, not an error.
