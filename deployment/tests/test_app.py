"""UI tests with Streamlit's headless AppTest.  Run:  python tests/test_app.py   (pytest also works)"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
from streamlit.testing.v1 import AppTest

def _run(): return AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120).run()
def _pct(at):
    for m in at.markdown:
        g = re.search(r"Predicted probability</div><div[^>]*>([0-9.]+)%", m.value)
        if g: return float(g.group(1))
    return None
def _by(ws, label): return next(w for w in ws if w.label.startswith(label))

def test_renders_all_widgets():
    at = _run(); assert not at.exception
    assert len(at.number_input) == 5 and len(at.selectbox) == 8 and [b.label for b in at.button] == ["Predict"]

def test_predict_shows_probability_and_shap_chart():
    at = _run(); at.button[0].click().run(); assert not at.exception, [e.value for e in at.exception]
    p = _pct(at); assert p is not None and 0 <= p <= 100
    assert len(at.get("vega_lite_chart")) + len(at.get("arrow_vega_lite_chart")) == 1, "SHAP bar chart missing"   # element name differs across Streamlit versions
    assert any("Why did the model produce this estimate?" in m.value for m in at.markdown) and any("not a medical diagnosis" in m.value for m in at.markdown)

def test_changing_one_widget_changes_the_prediction():
    at = _run(); at.button[0].click().run(); p0 = _pct(at)
    _by(at.selectbox, "Cognitive status").set_value("Mild cognitive impairment (MCI)"); at.button[0].click().run(); assert not at.exception; p1 = _pct(at)
    assert p1 is not None and p1 != p0 and p1 > p0, (p0, p1)
    _by(at.number_input, "Age").set_value(88.0); at.button[0].click().run(); p2 = _pct(at); assert p2 is not None and p2 != p1, (p1, p2)

def test_missing_values_and_derived_bmi():
    at = _run(); at.button[0].click().run(); assert not at.exception and _pct(at) is not None                # everything optional left unknown
    _by(at.number_input, "Height").set_value(170.0); _by(at.number_input, "Weight").set_value(70.0); at.button[0].click().run(); assert not at.exception
    assert any("BMI 24.2" in c.value for c in at.caption), [c.value for c in at.caption]                       # 703*154.3/66.9^2

def test_unit_switch():
    at = _run(); at.radio[0].set_value("US (in, lb)").run(); assert not at.exception and any(w.label.startswith("Height (inches)") for w in at.number_input)

if __name__ == "__main__":
    fails = 0
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            try: f(); print("PASS", n)
            except Exception as e: fails += 1; print("FAIL", n, "->", repr(e))
    sys.exit(1 if fails else 0)
