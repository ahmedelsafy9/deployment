"""User-facing labels, help texts and the exact list of fields exposed by the UI (chosen from the final model's SHAP ranking)."""
UNKNOWN = "Unknown / Not provided"

FEATURE_LABELS = {
 "age": "Age", "alcohol_abuse_history": "Alcohol abuse history", "alcohol_binge_frequency": "Binge drinking frequency", "alcohol_drinks_per_occasion": "Drinks per occasion",
 "alcohol_frequency_12mo": "Alcohol frequency (past 12 months)", "alcohol_frequency_3mo": "Alcohol frequency (past 3 months)", "anxiety": "Anxiety", "arthritis_any": "Arthritis (any)",
 "bmi": "Body mass index (BMI)", "bowel_incontinence": "Bowel incontinence", "bp_diastolic": "Diastolic blood pressure", "bp_diastolic_avg": "Diastolic BP (average)",
 "bp_systolic": "Systolic blood pressure", "bp_systolic_avg": "Systolic BP (average)", "cancer_recent": "Recent cancer", "chronic_kidney_disease": "Chronic kidney disease",
 "cognitive_status": "Cognitive status", "copd_asthma_combined": "COPD / asthma", "depression": "Depression", "education_level": "Education level", "education_years": "Years of education",
 "ethnicity_hispanic": "Hispanic/Latino ethnicity", "hearing_normal_1yes": "Hearing functionally normal", "height_in": "Height", "hyperlipidemia": "High cholesterol / lipids",
 "hypertension": "Hypertension", "independence_level": "Functional independence", "living_situation": "Living situation", "major_depressive_disorder": "Major depressive disorder",
 "marital_status": "Marital status", "migraine_chronic_headache": "Migraine / chronic headache", "obesity_bmi30plus": "Obesity (BMI 30+)", "osteoarthritis": "Osteoarthritis",
 "parkinsonism_pd": "Parkinsonism", "peripheral_vascular_disease": "Peripheral vascular disease", "physical_activity_hrs_wk": "Physical activity (hours/week)", "race": "Race",
 "residence_type": "Type of residence", "rheumatoid_arthritis": "Rheumatoid arthritis", "seizures_epilepsy": "Seizures / epilepsy", "sex": "Sex", "sleep_apnea": "Sleep apnea",
 "sleep_apnea_clinician": "Sleep apnea (clinician-assessed)", "smoking_current_30days": "Smoked in last 30 days", "smoking_ever_100cigs": "Smoked 100+ cigarettes in lifetime",
 "smoking_packs_per_day_category": "Packs per day (smokers)", "smoking_quit_age": "Age quit smoking", "smoking_status_derived": "Smoking status", "smoking_years": "Years of smoking",
 "stroke": "Stroke", "thyroid_disease": "Thyroid disease", "traumatic_brain_injury": "Traumatic brain injury", "type_2_diabetes": "Type 2 diabetes",
 "urinary_incontinence": "Urinary incontinence", "vision_normal_1yes": "Vision functionally normal", "vitamin_b12_deficiency": "Vitamin B12 deficiency", "weight_lb": "Weight"}

_NEWER_FORM = (" In the training data this item was recorded almost only on newer (2015+) visit forms, so leaving it unknown is handled as 'Missing', "
               "which in that data mostly matched older records (higher conversion rate).")
def _yn(): return [("No", "No"), ("Yes", "Yes")]

# Exposed fields. 'key' = model feature name (height/weight are converted to height_in / weight_lb; BMI and the obesity flag are derived from them).
FIELDS = [
 dict(group="Core information", key="age", label="Age (years)", kind="number", min=18.0, max=110.0, step=1.0, default=72.0, required=True, help="Age at the baseline visit."),
 dict(group="Core information", key="cognitive_status", label="Cognitive status", kind="select", required=True, default="Normal cognition",
      options=[("Normal cognition", "Normal cognition"), ("Impaired-not-MCI", "Impaired, not MCI"), ("MCI", "Mild cognitive impairment (MCI)"), ("Dementia", "Dementia")],
      help="Cognitive status at the baseline visit, as categorised in the NACC data ('Impaired, not MCI' = impaired but not meeting MCI criteria)."),
 dict(group="Core information", key="independence_level", label="Functional independence", kind="select", required=True,
      options=[("Able to live independently", "Able to live independently"), ("Requires assistance-complex activities", "Needs help with complex activities (finances, appointments...)"),
               ("Requires assistance-basic activities", "Needs help with basic activities (dressing, eating...)"), ("Completely dependent", "Completely dependent")],
      help="Level of independence in daily life."),
 dict(group="Core information", key="education_years", label="Years of education", kind="number", min=0.0, max=30.0, step=1.0, help="Total years of formal education."),
 dict(group="Body measures", key="height", label="Height", kind="height", help="Used together with weight to compute BMI (as in the training data)."),
 dict(group="Body measures", key="weight", label="Weight", kind="weight", help="Used together with height to compute BMI. If either is missing, BMI is treated as unknown."),
 dict(group="Body measures", key="bp_systolic", label="Systolic blood pressure (mmHg)", kind="number", min=50.0, max=300.0, step=1.0, help="The top number of a blood-pressure reading."),
 dict(group="Health history", key="hypertension", label="Hypertension (high blood pressure)", kind="select", required=True,
      options=[("Absent", "No"), ("Recent/Active", "Yes - recent or active"), ("Remote/Inactive", "Yes - remote / inactive (past)")], help="History of hypertension."),
 dict(group="Health history", key="hearing_normal_1yes", label="Hearing functionally normal?", kind="select", options=[("Yes (normal)", "Yes - normal"), ("No (impaired)", "No - impaired")],
      help="Whether hearing was recorded as functionally normal."),
 dict(group="Health history", key="parkinsonism_pd", label="Parkinsonism / Parkinson's disease", kind="select", required=True, options=[("Absent", "No"), ("Recent/Active", "Yes")], help="Parkinsonism recorded at baseline."),
 dict(group="Health history", key="bowel_incontinence", label="Bowel incontinence", kind="select", options=_yn(), help="Bowel incontinence recorded at baseline." + _NEWER_FORM),
 dict(group="Health history", key="urinary_incontinence", label="Urinary incontinence", kind="select", options=_yn(), help="Urinary incontinence recorded at baseline." + _NEWER_FORM),
 dict(group="Health history", key="cancer_recent", label="Recent cancer", kind="select", options=_yn(), help="Cancer recorded in the health history at baseline." + _NEWER_FORM)]

def value_label(feature, value):
    """Human-readable label of a stored category value (falls back to the raw value)."""
    for f in FIELDS:
        if f.get("key") == feature and f["kind"] == "select":
            return dict(f["options"]).get(value, value)
    return value

# Newer-form items: in the training data 'Missing' for these mostly marks OLDER (pre-2015) records, which had about twice the conversion rate.
NEWER_FORM_KEYS = ("bowel_incontinence", "urinary_incontinence", "cancer_recent")
# Categorical fields whose 'Missing' level had < 100 training rows (<0.6%) are required in the UI: the model's response to such a rare level is not reliable.
