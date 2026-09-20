#!/usr/bin/env python3
"""
build_model_artifacts.py - creates the deployable artifacts of the FINAL selected model (run once, ~10 s).

Final model = "CLEAN_P2 / LightGBM / seed 42" of the leakage-safe baseline stage (see MODEL_SELECTION.md):
  * frozen 57-feature set (11 numeric + 46 categorical, incl. baseline cognitive_status)
  * pre-imputation baseline values (cells flagged <feature>_was_imputed == 1 are blanked again)
  * TRAIN-only preprocessing: numeric blank -> TRAIN median, categorical blank -> explicit 'Missing', vocabularies from TRAIN
  * LightGBM, original class distribution, unchanged hyper-parameters, early stopping on validation PR-AUC, seed 42
  * operating threshold = F1-maximising threshold chosen on VALIDATION
No experiment is run: configuration, data and seed are those of the selected model, so the fit is reproduced deterministically.
The script checks itself against the numbers recorded when the model was selected (EXPECTED below).
Paths: env ALZ_DIR (default E:\\ALZ) or ANCHOR_CSV.        Outputs: models/{final_model.txt, preprocessing.json, model_metadata.json, shap_importance_final_model.csv}
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb, sklearn
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); MODELS = os.path.join(HERE, "models")
ALZ_DIR = os.environ.get("ALZ_DIR", r"E:\ALZ")
ANCHOR_CSV = os.environ.get("ANCHOR_CSV", os.path.join(ALZ_DIR, "model_ready", "ad_conversion_3yr_anchor_dataset.csv"))
TARGET, SPLIT, COG, SEED = "target_ad_conversion", "split", "cognitive_status", 42
NUMERIC = ["age", "education_years", "height_in", "weight_lb", "bmi", "bp_systolic", "bp_diastolic", "bp_systolic_avg", "bp_diastolic_avg", "smoking_years", "smoking_quit_age"]
METADATA = {"participant_id", "visit_id", "visit_date", "sequence_step", "days_since_baseline", "is_baseline_visit", "is_outcome_determining_visit", "baseline_visit_id", "baseline_visit_date",
            "followup_years_window", "outcome_determining_visit_id", "outcome_determining_visit_date", "target_ad_conversion", "split"}
FORBIDDEN = {"alzheimers_etiologic_dx", "mci_flag", "dementia_criteria_met", "normal_cognition", "other_dementia_flag", "lewy_body_etiologic_dx", "ftld_etiologic_dx"}
LGB_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=6, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=0.1, verbosity=-1, objective="binary", metric="average_precision", random_state=SEED)
EXPECTED = dict(trees=14, val_pr_auc=0.4337, test_pr_auc=0.4223, threshold=0.1616)          # recorded at selection time (leakage-safe baseline, seed 42)

def load_data(anchor_csv=ANCHOR_CSV):
    """Returns (feature_order, numeric, categorical, {'train','validation','test'} raw frames with the upstream imputation undone)."""
    df = pd.read_csv(anchor_csv, dtype=str, keep_default_na=False)
    base = sorted(c for c in df.columns if c not in METADATA and c not in FORBIDDEN and not c.endswith("_was_imputed") and c != COG)
    feats = base + [COG]; num = [f for f in feats if f in NUMERIC]; cat = [f for f in feats if f not in NUMERIC]
    assert (len(feats), len(num), len(cat)) == (57, 11, 46)
    raw = df.copy()
    for f in base: raw.loc[df[f + "_was_imputed"] == "1", f] = ""
    return feats, num, cat, {k: raw[raw[SPLIT] == k].reset_index(drop=True) for k in ("train", "validation", "test")}

def fit_preprocessing(train, num, cat):
    medians = {c: float(pd.to_numeric(train[c].replace("", np.nan), errors="coerce").median()) for c in num}
    vocab = {c: sorted(set(train[c].replace("", "Missing").unique()) | {"Missing"}) for c in cat}
    return medians, vocab

def training_transform(d, feats, medians, vocab):
    X = pd.DataFrame(index=d.index)
    for c in feats:
        if c in medians: X[c] = pd.to_numeric(d[c].replace("", np.nan), errors="coerce").fillna(medians[c]).astype(float)
        else:
            v = d[c].replace("", "Missing"); X[c] = pd.Categorical(v.where(v.isin(vocab[c]), "Missing"), categories=vocab[c])
    return X

def main():
    os.makedirs(MODELS, exist_ok=True)
    feats, num, cat, part = load_data(); tr, va, te = part["train"], part["validation"], part["test"]
    ytr, yva, yte = (p[TARGET].astype(int).values for p in (tr, va, te)); assert (len(tr), len(va), len(te)) == (18090, 3876, 3877)
    medians, vocab = fit_preprocessing(tr, num, cat)
    Xtr, Xva, Xte = (training_transform(p, feats, medians, vocab) for p in (tr, va, te))
    m = lgb.LGBMClassifier(**LGB_PARAMS); m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)])
    best = int(m.best_iteration_); booster = m.booster_; booster.save_model(os.path.join(MODELS, "final_model.txt"), num_iteration=best)
    pv, pt = booster.predict(Xva, num_iteration=best), booster.predict(Xte, num_iteration=best)
    pr, rc, th = precision_recall_curve(yva, pv); f1 = 2 * pr * rc / np.clip(pr + rc, 1e-12, None); thr = float(th[int(np.nanargmax(f1[:-1]))])
    def prf(y, p, t):
        pred = p >= t; tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum()); a = tp / max(tp + fp, 1); b = tp / max(tp + fn, 1)
        return dict(precision=a, recall=b, f1=2 * a * b / max(a + b, 1e-12), flagged_share=float(pred.mean()))
    val_pr, test_pr = average_precision_score(yva, pv), average_precision_score(yte, pt)
    ok = best == EXPECTED["trees"] and abs(val_pr - EXPECTED["val_pr_auc"]) < 5e-4 and abs(test_pr - EXPECTED["test_pr_auc"]) < 5e-4 and abs(thr - EXPECTED["threshold"]) < 1e-3
    print(f"trees={best}  threshold={thr:.4f}  val PR-AUC={val_pr:.4f}  test PR-AUC={test_pr:.4f}  ->  matches the recorded selected model: {ok}")
    if not ok: print("WARNING: the rebuilt model does not match the recorded selected model (check library versions / data file)")
    # SHAP (LightGBM TreeSHAP, log-odds) on validation rows -> ranking used to choose the UI inputs
    sh = booster.predict(Xva, pred_contrib=True, num_iteration=best)[:, :-1]
    shap_df = pd.DataFrame({"feature": feats, "mean_abs_shap_log_odds_validation": np.abs(sh).mean(0)}).sort_values("mean_abs_shap_log_odds_validation", ascending=False)
    shap_df["rank"] = range(1, len(shap_df) + 1); gain = pd.Series(booster.feature_importance("gain", iteration=best), index=booster.feature_name()); shap_df["gain_share"] = (gain / gain.sum()).reindex(shap_df.feature).values
    shap_df.to_csv(os.path.join(MODELS, "shap_importance_final_model.csv"), index=False, float_format="%.5f")
    # reference statistics (documentation / validation warnings)
    year = pd.to_datetime(tr["baseline_visit_date"]).dt.year if "baseline_visit_date" in tr else None
    ref = {c: dict(min=float(v.min()), p01=float(v.quantile(.01)), median=float(v.median()), p99=float(v.quantile(.99)), max=float(v.max()), pct_blank_train=float(100 * v.isna().mean()))
           for c in num for v in [pd.to_numeric(tr[c].replace("", np.nan), errors="coerce")]}
    blank = {c: float(100 * (tr[c] == "").mean()) for c in feats}
    era_linked = []
    if year is not None:
        for c in feats:
            b = (tr[c] == "").astype(int)
            if 0.05 <= b.mean() <= 0.95 and np.corrcoef(b, (year <= 2014).astype(int))[0, 1] >= 0.90: era_linked.append(c)
    json.dump(dict(feature_order=feats, numeric=num, categorical=cat, numeric_medians=medians, categorical_vocab=vocab, missing_token="Missing", numeric_reference=ref, pct_blank_train=blank,
                   era_linked_features=sorted(era_linked), level_counts_train={c: {k: int(n) for k, n in tr[c].replace("", "Missing").value_counts().items()} for c in cat},
                   description="Train-only preprocessing: numeric blank -> TRAIN median; categorical blank/unknown -> 'Missing' level; categories fixed to the TRAIN vocabulary; column order = feature_order."),
              open(os.path.join(MODELS, "preprocessing.json"), "w"), indent=1)
    def deciles(y, p):
        d = pd.DataFrame(dict(y=y, p=p)); d["decile"] = pd.qcut(d.p, 10, labels=False, duplicates="drop") + 1
        return [dict(decile=int(k), n=int(len(g)), mean_predicted=float(g.p.mean()), observed_rate=float(g.y.mean())) for k, g in d.groupby("decile")]
    e_va = (pd.to_datetime(va["baseline_visit_date"]).dt.year <= 2014).astype(int).values; e_te = (pd.to_datetime(te["baseline_visit_date"]).dt.year <= 2014).astype(int).values
    era_cal = {nm: {lab: dict(n=int((e == k).sum()), observed_rate=float(y[e == k].mean()), mean_predicted=float(p[e == k].mean())) for k, lab in ((1, "pre-2015 baseline"), (0, "2015+ baseline"))}
               for nm, y, p, e in (("validation", yva, pv, e_va), ("test", yte, pt, e_te))}
    meta = dict(model_name="LightGBM - leakage-safe baseline (CLEAN_P2, seed 42)", model_version="lgbm-cleanP2-seed42-57f", model_file="models/final_model.txt", built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                model_type="LightGBM gradient-boosted trees (binary)", n_features=len(feats), n_trees=best, n_train=len(tr), n_validation=len(va), n_test=len(te), train_prevalence=float(ytr.mean()),
                training_dataset="model_ready/ad_conversion_3yr_anchor_dataset.csv (baseline rows; pre-imputation values reconstructed via *_was_imputed flags; participant-level train/validation/test split unchanged)",
                target_definition="Project's 3-year conversion outcome: 1 = a new Alzheimer's etiologic diagnosis is recorded within 3 years after the baseline visit in a participant without it at baseline; 0 = a determinate non-converting follow-up observation within the window.",
                hyperparameters=LGB_PARAMS, class_weighting=False, resampling="none", calibration="none - raw LightGBM probabilities",
                operating_threshold=thr, threshold_criterion="F1-maximising threshold on the VALIDATION set (frozen before the test set was read)",
                validation=dict(pr_auc=val_pr, roc_auc=roc_auc_score(yva, pv), brier=brier_score_loss(yva, pv), **prf(yva, pv, thr)), test=dict(pr_auc=test_pr, roc_auc=roc_auc_score(yte, pt), brier=brier_score_loss(yte, pt), **prf(yte, pt, thr)),
                three_seed_mean_reference=dict(validation_pr_auc=0.4326, test_pr_auc=0.4269, seeds=[42, 123, 2026]), calibration_deciles=dict(validation=deciles(yva, pv), test=deciles(yte, pt)), era_calibration=era_cal,
                matches_recorded_selected_model=bool(ok), versions=dict(python=sys.version.split()[0], lightgbm=lgb.__version__, pandas=pd.__version__, numpy=np.__version__, sklearn=sklearn.__version__),
                disclaimer="Research/model demonstration. Not a diagnostic device.")
    json.dump(meta, open(os.path.join(MODELS, "model_metadata.json"), "w"), indent=1)
    print("artifacts written to", MODELS)
if __name__ == "__main__": main()
