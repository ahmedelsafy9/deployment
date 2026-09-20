"""Inference tests.  Run:  python tests/test_inference.py     (pytest also works)
The last two tests need the anchor dataset (E:\\ALZ\\model_ready\\...; override with ANCHOR_CSV) and are skipped when it is not found."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from inference import ConversionModel, InputValidationError
M = ConversionModel()
NON_CONVERTER = dict(age=66, cognitive_status="Normal cognition", education_years=18, height_in=68, weight_lb=155, bp_systolic=124, independence_level="Able to live independently",
                     hypertension="Absent", hearing_normal_1yes="Yes (normal)", parkinsonism_pd="Absent", bowel_incontinence="No", urinary_incontinence="No", cancer_recent="No")
CONVERTER = dict(age=79, cognitive_status="MCI", education_years=12, height_in=64, weight_lb=140, bp_systolic=138, independence_level="Requires assistance-complex activities",
                 hypertension="Recent/Active", hearing_normal_1yes="No (impaired)", parkinsonism_pd="Absent", bowel_incontinence="No", urinary_incontinence="Yes", cancer_recent="No")

def test_non_converter_runs_and_explains():
    r = M.predict(NON_CONVERTER)
    assert 0.0 <= r["probability"] <= 1.0 and math.isclose(r["percentage"], round(100 * r["probability"], 1))
    assert len(r["explanation"]) == 5 and all(math.isfinite(e["contribution_log_odds"]) for e in r["explanation"])
    assert r["model_name"] and r["model_version"] and r["risk_category"] in ("Lower predicted probability", "Higher predicted probability")

def test_converter_profile_scores_higher_and_output_changes():
    a, b = M.predict(NON_CONVERTER)["probability"], M.predict(CONVERTER)["probability"]
    assert 0 <= b <= 1 and b > a, (a, b)

def test_changing_one_input_changes_only_that_model_column():
    base_frame = M.to_frame(M.clean_input(NON_CONVERTER)[0]); base_p = M.predict(NON_CONVERTER, explain=False)["probability"]
    changes = dict(cognitive_status="MCI", age=80, independence_level="Completely dependent", hypertension="Recent/Active", parkinsonism_pd="Recent/Active", bowel_incontinence="Yes",
                   cancer_recent="Yes", education_years=8, bp_systolic=170, hearing_normal_1yes="No (impaired)", urinary_incontinence="Yes")
    moved = 0
    for k, v in changes.items():
        d = dict(NON_CONVERTER); d[k] = v; f = M.to_frame(M.clean_input(d)[0])
        diff = [c for c in M.features if not f[c].equals(base_frame[c])]
        assert diff == [k], (k, diff)                                   # the model receives exactly the changed value, nothing else
        moved += abs(M.predict(d, explain=False)["probability"] - base_p) > 1e-9
    assert moved >= 5, moved                                            # the prediction reacts to the inputs the (14-tree) model actually uses
    d = dict(NON_CONVERTER, weight_lb=230); diff = {c for c in M.features if not M.to_frame(M.clean_input(d)[0])[c].equals(base_frame[c])}
    assert diff == {"weight_lb", "bmi", "obesity_bmi30plus"}, diff      # weight also changes the derived BMI / obesity flag

def test_missing_values_use_training_handling():
    r = M.predict(dict(age=70, cognitive_status="MCI"))                # everything else not provided
    assert 0 <= r["probability"] <= 1 and len(r["defaulted_features"]) == 55
    f = M.to_frame(M.clean_input(dict(age=70, cognitive_status="MCI"))[0])
    assert f["bmi"].iloc[0] == M.medians["bmi"] and str(f["hypertension"].iloc[0]) == "Missing" and str(f["sleep_apnea"].iloc[0]) == "Missing"
    assert M.predict(dict(age=70, cognitive_status="MCI", education_years=None, bp_systolic="", height_in="Unknown / Not provided"))["probability"] == r["probability"]

def test_input_validation():
    bad = [dict(age=5, cognitive_status="MCI"), dict(age=150, cognitive_status="MCI"), dict(age=70, cognitive_status="MCI", bmi=-3), dict(age=70, cognitive_status="MCI", education_years=-1),
           dict(age=70, cognitive_status="MCI", bp_systolic=-120), dict(age=70, cognitive_status="MCI", bp_systolic=900), dict(age=70, cognitive_status="Foo"), dict(age=70),
           dict(age="abc", cognitive_status="MCI"), dict(age=70, cognitive_status="MCI", nonsense=1), dict(age=70, cognitive_status="MCI", height_in=40, weight_lb=700)]
    for b in bad:
        try: M.predict(b); raise AssertionError("accepted invalid input: %r" % (b,))
        except InputValidationError: pass
    assert M.predict(dict(age=70, cognitive_status="Impaired-not-MCI", bp_systolic=110))["probability"] > 0        # legitimate values are not over-restricted

def _anchor():
    import build_model_artifacts as B
    if not os.path.exists(B.ANCHOR_CSV): return None
    return B, B.load_data()

def test_inference_preprocessing_equals_training_preprocessing():
    a = _anchor()
    if a is None: print("  (skipped: anchor dataset not found)"); return
    B, (feats, num, cat, part) = a; va = part["validation"]
    Xtrain_style = B.training_transform(va, feats, M.medians, M.vocab); ref = M.booster.predict(Xtrain_style)
    keep = [i for i in range(len(va)) if all(va.loc[i, c] == "" or va.loc[i, c] in M.vocab[c] for c in cat)]
    got = np.array([M.predict({f: va.loc[i, f] for f in feats}, explain=False, derive=False)["probability"] for i in keep[:600]])
    assert np.abs(got - ref[keep[:600]]).max() < 1e-9, np.abs(got - ref[keep[:600]]).max()

def test_deployed_model_is_the_selected_model():
    a = _anchor()
    if a is None: print("  (skipped: anchor dataset not found)"); return
    from sklearn.metrics import average_precision_score
    B, (feats, num, cat, part) = a; va = part["validation"]; y = va[B.TARGET].astype(int).values
    p = np.array([M.predict({f: va.loc[i, f] for f in feats}, explain=False, derive=False)["probability"] for i in range(len(va))])
    ap = average_precision_score(y, p); assert abs(ap - M.meta["validation"]["pr_auc"]) < 1e-9 and abs(ap - 0.4337) < 5e-4, ap      # end-to-end through the inference path
    assert p.std() > 0.02 and 0 <= p.min() and p.max() <= 1

if __name__ == "__main__":
    fails = 0
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            try: f(); print("PASS", n)
            except Exception as e: fails += 1; print("FAIL", n, "->", repr(e))
    sys.exit(1 if fails else 0)
