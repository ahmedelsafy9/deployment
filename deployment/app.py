"""Local Streamlit demo of the final model.  Run:  streamlit run app.py     (nothing entered is stored, logged or sent anywhere)"""
import numpy as np, pandas as pd, altair as alt, streamlit as st
from inference import ConversionModel, InputValidationError, ModelNotBuiltError
from ui_spec import FIELDS, UNKNOWN, NEWER_FORM_KEYS, FEATURE_LABELS

st.set_page_config(page_title="AD 3-year conversion model demo", page_icon="🧠", layout="centered")

@st.cache_resource(show_spinner=False)
def load_model(): return ConversionModel()

st.title("Alzheimer's 3-Year Conversion Prediction")
st.caption("Research / model demonstration - not a medical diagnosis. Nothing you enter is stored, logged or sent anywhere.")
try: model = load_model()
except ModelNotBuiltError as e:
    st.error(str(e)); st.stop()
meta = model.meta

st.write("Enter baseline information below. Only the fields marked * are required; anything else can stay *Unknown / Not provided* and is handled exactly as missing values were handled in training.")
units = st.radio("Units for height and weight", ["Metric (cm, kg)", "US (in, lb)"], horizontal=True); metric = units.startswith("Metric")

def widget(f):
    k, label = f["key"], f["label"] + (" *" if f.get("required") else "")
    if f["kind"] == "select":
        labels = [l for _, l in f["options"]]; opts = labels if f.get("required") else [UNKNOWN] + labels
        idx = opts.index(dict(f["options"])[f["default"]]) if f.get("default") else 0
        choice = st.selectbox(label, opts, index=idx, help=f["help"], key=k)
        return None if choice == UNKNOWN else {l: v for v, l in f["options"]}[choice]
    if f["kind"] == "number":
        return st.number_input(label, min_value=f["min"], max_value=f["max"], value=f.get("default"), step=f["step"], placeholder="Unknown / not provided", help=f["help"], key=k)
    if f["kind"] == "height":
        v = st.number_input("Height (cm)" if metric else "Height (inches)", min_value=90.0 if metric else 36.0, max_value=245.0 if metric else 96.0, value=None, step=1.0, placeholder="Unknown / not provided", help=f["help"], key="height_m" if metric else "height_us")
        return None if v is None else (v / 2.54 if metric else v)
    v = st.number_input("Weight (kg)" if metric else "Weight (pounds)", min_value=25.0 if metric else 55.0, max_value=350.0 if metric else 770.0, value=None, step=0.5, placeholder="Unknown / not provided", help=f["help"], key="weight_m" if metric else "weight_us")
    return None if v is None else (v * 2.2046226218 if metric else v)

values = {}
with st.form("patient_form"):
    for group in dict.fromkeys(f["group"] for f in FIELDS):
        st.subheader(group); cols = st.columns(2)
        for i, f in enumerate(x for x in FIELDS if x["group"] == group):
            with cols[i % 2]:
                v = widget(f); values[{"height": "height_in", "weight": "weight_lb"}.get(f["key"], f["key"])] = v
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    try: res = model.predict(values, explain=True, top_k=5)
    except InputValidationError as e:
        for msg in e.errors: st.error(msg)
        st.stop()
    pct = res["percentage"]
    st.markdown("### Model Prediction")
    st.markdown(f"<div style='padding:1.4rem;border-radius:14px;background:rgba(110,120,200,0.14);text-align:center'>"
                f"<div style='font-size:1rem;opacity:.75'>Predicted probability</div><div style='font-size:3.4rem;font-weight:700;line-height:1.15'>{pct:.1f}%</div></div>", unsafe_allow_html=True)
    st.write(f"This model estimates a **{pct:.1f}%** probability of conversion within 3 years, as defined in this project, based on the information entered.")
    st.info(f"Relative to the model's validation-selected operating threshold ({100*res['operating_threshold']:.1f}%): **{res['risk_category']}**. {res['risk_category_note']}")
    st.caption("Probabilities are raw model outputs (no recalibration). On held-out data the model tends to over-estimate at low predicted values and under-estimate at high ones - see *Model Information*.")
    for w in res["warnings"]: st.warning(w)
    unknown_newer = [k for k in NEWER_FORM_KEYS if k in res["defaulted_features"]]
    if unknown_newer:
        alt_in = dict(values); alt_in.update({k: "No" for k in unknown_newer}); alt_pct = model.predict(alt_in, explain=False)["percentage"]
        st.warning("**Newer-form items not entered:** " + ", ".join(FEATURE_LABELS[k] for k in unknown_newer) + ". In the training data these items were blank almost only for older (pre-2015) records, "
                   f"which had about twice the conversion rate, so leaving them unknown raises the estimate. For comparison, recording them as 'No' would give **{alt_pct:.1f}%**.")
    if res["derived_features"]:
        bmi = model.to_frame(model.clean_input(values)[0])["bmi"].iloc[0]; st.caption(f"Derived from height and weight: BMI {bmi:.1f}" + (" (obesity flag set automatically)" if "obesity_bmi30plus" in res["derived_features"] else ""))

    st.markdown("### Why did the model produce this estimate?")
    ex = pd.DataFrame(res["explanation"]); ex["name"] = ex.apply(lambda r: f"{r.label}: {r.value}", axis=1)
    ex["effect"] = np.where(ex.contribution_log_odds > 0, "Increases model prediction", "Decreases model prediction")
    chart = alt.Chart(ex).mark_bar().encode(
        x=alt.X("contribution_log_odds:Q", title="Contribution to the model output (SHAP, log-odds)"),
        y=alt.Y("name:N", sort=list(ex.assign(a=ex.contribution_log_odds.abs()).sort_values("a", ascending=False).name), title=None, axis=alt.Axis(labelLimit=380)),
        color=alt.Color("effect:N", scale=alt.Scale(domain=["Increases model prediction", "Decreases model prediction"], range=["#d95f02", "#1b9e77"]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["label", "value", alt.Tooltip("contribution_log_odds:Q", format=".3f"), "source"]).properties(height=44 * len(ex) + 30, width="container")
    st.altair_chart(chart)
    t = res["explanation_totals"]
    st.caption("These are the model features that contributed most to this prediction (TreeSHAP values of the final model, log-odds scale, relative to the model's average output of "
               f"{100*res['base_probability']:.1f}%). They describe how the model reached the estimate, not medical causes. Items marked 'not entered' were handled as missing. "
               f"Total contribution - entered/derived: {t['entered']+t['derived']:+.2f}; not entered: {t['default']:+.2f}.")
    with st.expander("How your entries were used"):
        st.write(f"**Entered directly:** {', '.join(FEATURE_LABELS[f] for f in res['entered_features']) or '-'}")
        st.write(f"**Derived from your entries (as in the training data):** {', '.join(FEATURE_LABELS[f] for f in res['derived_features']) or '-'}")
        st.write(f"**Not entered ({len(res['defaulted_features'])} of {meta['n_features']} model features):** numeric features use the training median, categorical features use the 'Missing' category - exactly as during training. No clinical values are invented.")

with st.expander("Model Information"):
    v, t = meta["validation"], meta["test"]
    st.markdown(f"- **Model type:** {meta['model_type']} ({meta['n_trees']} trees), version `{meta['model_version']}`\n- **Training dataset:** {meta['training_dataset']}\n"
                f"- **Training samples:** {meta['n_train']:,} (validation {meta['n_validation']:,}, test {meta['n_test']:,}); training conversion rate {100*meta['train_prevalence']:.1f}%\n- **Features:** {meta['n_features']} (11 numeric, 46 categorical)\n"
                f"- **Validation PR-AUC:** {v['pr_auc']:.3f} (ROC-AUC {v['roc_auc']:.3f})   |   **Test PR-AUC:** {t['pr_auc']:.3f} (ROC-AUC {t['roc_auc']:.3f}); 3-seed mean {meta['three_seed_mean_reference']['validation_pr_auc']:.3f} / {meta['three_seed_mean_reference']['test_pr_auc']:.3f}\n"
                f"- **Operating threshold:** {100*meta['operating_threshold']:.1f}% ({meta['threshold_criterion']}); on the test set it flags {100*t['flagged_share']:.0f}% of participants with precision {t['precision']:.2f} and recall {t['recall']:.2f}\n"
                f"- **Target definition:** {meta['target_definition']}\n- **Calibration:** {meta['calibration']}")
    d = pd.DataFrame(meta["calibration_deciles"]["test"]); d["mean_predicted"] = (100 * d.mean_predicted).round(1); d["observed_rate"] = (100 * d.observed_rate).round(1)
    st.write("Predicted vs observed conversion (%) by predicted-probability decile, held-out test set:"); st.dataframe(d.rename(columns={"decile": "decile (low to high)", "mean_predicted": "mean predicted %", "observed_rate": "observed %"}), hide_index=True)
    ec = meta["era_calibration"]["test"]; st.write("Test set by baseline era - " + "; ".join(f"{k}: observed {100*x['observed_rate']:.1f}%, mean predicted {100*x['mean_predicted']:.1f}% (n={x['n']})" for k, x in ec.items()))
    st.caption("Baseline-visit records span 2005 onward; several items were only collected on newer forms, so form era is entangled with the model's inputs. The model has not been validated for clinical use.")

st.markdown("---"); st.markdown("**Important note:** this is a research/model demonstration, not a medical diagnosis. The output is a statistical estimate of the project's defined 3-year conversion outcome, not a certainty and not a statement about any individual's future.")
