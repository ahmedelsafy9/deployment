"""
inference.py - loads the saved final model + train-only preprocessing artifacts and exposes  predict(input_data).

input_data: dict {model_feature_name: value}. Numeric -> number; categorical -> one of the TRAIN categories (see models/preprocessing.json);
None / "" / "Unknown / Not provided" = not provided, which is handled EXACTLY as in training (numeric -> TRAIN median, categorical -> 'Missing').
Only `age` and `cognitive_status` are required (they were never blank in the training data). Features not supplied at all are also treated as not provided.
Derived (disable with derive=False, used to replay stored records exactly): bmi = 703*weight_lb/height_in^2 (1 decimal) when both are given; obesity_bmi30plus = 'Yes' if bmi >= 30 else 'No' when bmi is known.
Nothing is logged or stored.
"""
import json, os, math
import numpy as np, pandas as pd, lightgbm as lgb
from ui_spec import FEATURE_LABELS, value_label, UNKNOWN

HERE = os.path.dirname(os.path.abspath(__file__)); MODELS_DIR = os.path.join(HERE, "models")
REQUIRED = ("age", "cognitive_status")
BOUNDS = {"age": (18, 110), "education_years": (0, 30), "height_in": (36, 96), "weight_lb": (40, 800), "bmi": (8, 100), "bp_systolic": (50, 300), "bp_diastolic": (20, 200),
          "bp_systolic_avg": (50, 300), "bp_diastolic_avg": (20, 200), "smoking_years": (0, 100), "smoking_quit_age": (0, 110)}       # hard limits (impossible values are rejected)
_MISSING = {"", "unknown", "unknown / not provided", "missing", "nan"}

class ModelNotBuiltError(RuntimeError): pass
class InputValidationError(ValueError):
    def __init__(self, errors): super().__init__("; ".join(errors)); self.errors = list(errors)

def _is_missing(v):
    if v is None: return True
    if isinstance(v, float) and math.isnan(v): return True
    return isinstance(v, str) and v.strip().lower() in _MISSING

class ConversionModel:
    def __init__(self, models_dir=MODELS_DIR):
        mp = os.path.join(models_dir, "final_model.txt")
        if not os.path.exists(mp) or not os.path.exists(os.path.join(models_dir, "preprocessing.json")):
            raise ModelNotBuiltError("Model artifacts not found in %s. Run once:  python build_model_artifacts.py" % models_dir)
        self.pre = json.load(open(os.path.join(models_dir, "preprocessing.json"))); self.meta = json.load(open(os.path.join(models_dir, "model_metadata.json")))
        self.booster = lgb.Booster(model_file=mp)
        self.features = self.pre["feature_order"]; self.numeric = set(self.pre["numeric"]); self.medians = self.pre["numeric_medians"]; self.vocab = self.pre["categorical_vocab"]
        assert self.booster.feature_name() == self.features, "model feature order != preprocessing feature order"
        self.threshold = float(self.meta["operating_threshold"])

    # ---- validation + derivation ---------------------------------------------------------------------------------
    def clean_input(self, data, derive=True):
        errors, warns = [], []
        extra = [k for k in data if k not in self.features]
        if extra: errors.append("Unknown input field(s): %s" % ", ".join(extra))
        clean = {}
        for f in self.features:
            v = data.get(f)
            if _is_missing(v): clean[f] = None; continue
            if f in self.numeric:
                try: x = float(v)
                except (TypeError, ValueError): errors.append("%s must be a number (got %r)" % (FEATURE_LABELS.get(f, f), v)); continue
                if not math.isfinite(x): errors.append("%s must be a finite number" % FEATURE_LABELS.get(f, f)); continue
                lo, hi = BOUNDS.get(f, (-math.inf, math.inf))
                if not lo <= x <= hi: errors.append("%s = %g is outside the plausible range %g-%g" % (FEATURE_LABELS.get(f, f), x, lo, hi)); continue
                ref = self.pre["numeric_reference"][f]
                if not ref["p01"] <= x <= ref["p99"]: warns.append("%s = %g is outside the range that covers 98%% of the training data (%g-%g); the estimate is less reliable." % (FEATURE_LABELS.get(f, f), x, ref["p01"], ref["p99"]))
                clean[f] = x
            else:
                s = str(v).strip(); allowed = [c for c in self.vocab[f] if c != "Missing"]
                if s not in allowed: errors.append("%s: '%s' is not a category seen in training. Allowed: %s" % (FEATURE_LABELS.get(f, f), s, ", ".join(allowed))); continue
                clean[f] = s
        for r in REQUIRED:
            if clean.get(r) is None and not any(r in e for e in errors): errors.append("%s is required" % FEATURE_LABELS.get(r, r))
        entered = [f for f, v in clean.items() if v is not None]; derived = []
        h, w = clean.get("height_in"), clean.get("weight_lb")
        if derive and clean.get("bmi") is None and h and w:
            bmi = round(703.0 * w / h ** 2, 1)
            if not BOUNDS["bmi"][0] <= bmi <= BOUNDS["bmi"][1]: errors.append("Height and weight give an implausible BMI (%.1f)" % bmi)
            else: clean["bmi"] = bmi; derived.append("bmi")
        elif clean.get("bmi") is not None and h and w and abs(703.0 * w / h ** 2 - clean["bmi"]) > 1.5: warns.append("Entered BMI differs from the BMI implied by height and weight.")
        if derive and clean.get("bmi") is not None:
            ob = "Yes" if clean["bmi"] >= 30 else "No"
            if clean.get("obesity_bmi30plus") is None: clean["obesity_bmi30plus"] = ob; derived.append("obesity_bmi30plus")
            elif clean["obesity_bmi30plus"] != ob: warns.append("Obesity flag is inconsistent with BMI.")
        if errors: raise InputValidationError(errors)
        return clean, entered, derived, warns

    # ---- exact training-time row construction ------------------------------------------------------------------
    def to_frame(self, clean):
        row = {}
        for f in self.features:
            if f in self.numeric: row[f] = float(clean[f]) if clean.get(f) is not None else float(self.medians[f])        # TRAIN median
            else: row[f] = pd.Categorical([clean[f] if clean.get(f) is not None else "Missing"], categories=self.vocab[f])   # explicit 'Missing'
        return pd.DataFrame({f: (row[f] if f not in self.numeric else [row[f]]) for f in self.features})

    def predict(self, input_data, explain=True, top_k=5, derive=True):
        clean, entered, derived, warns = self.clean_input(input_data, derive=derive); X = self.to_frame(clean)
        p = float(self.booster.predict(X)[0]); assert 0.0 <= p <= 1.0
        above = p >= self.threshold
        out = dict(probability=p, percentage=round(100 * p, 1), operating_threshold=self.threshold, above_operating_threshold=bool(above),
                   risk_category=("Higher predicted probability" if above else "Lower predicted probability"),
                   risk_category_note="Relative to the model's validation-selected operating threshold (%.1f%%, F1-maximising on the validation set); a model operating point, not a clinical cut-off." % (100 * self.threshold),
                   model_name=self.meta["model_name"], model_version=self.meta["model_version"], warnings=warns, entered_features=entered, derived_features=derived,
                   defaulted_features=[f for f in self.features if f not in entered and f not in derived])
        if explain:
            c = self.booster.predict(X, pred_contrib=True)[0]; contrib, base = c[:-1], float(c[-1]); items = []
            for f, v in zip(self.features, contrib):
                src = "entered" if f in entered else ("derived" if f in derived else "default")
                if src == "default": shown = ("median %g used" % self.medians[f]) if f in self.numeric else "not entered (handled as Missing)"
                else: shown = ("%g" % clean[f]) if f in self.numeric else value_label(f, clean[f])
                items.append(dict(feature=f, label=FEATURE_LABELS.get(f, f), value=shown, source=src, contribution_log_odds=float(v), direction="increases" if v > 0 else "decreases"))
            items.sort(key=lambda d: -abs(d["contribution_log_odds"]))
            out.update(explanation=items[:top_k], base_probability=float(1 / (1 + math.exp(-base))),
                       explanation_totals={s: float(sum(d["contribution_log_odds"] for d in items if d["source"] == s)) for s in ("entered", "derived", "default")})
        return out

_MODEL = None
def get_model():
    global _MODEL
    if _MODEL is None: _MODEL = ConversionModel()
    return _MODEL
def predict(input_data, **kw): return get_model().predict(input_data, **kw)
