from pathlib import Path
from io import BytesIO
import base64

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="CanineCare AI | Screening",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model_outputs" / "tick_fever_gastroenteritis_model_bundle.joblib"
MODEL_TEXT_PATH = BASE / "model_bundle.b64"

LABELS = {
    "Age_Months": "Age in months",
    "Breed": "Breed",
    "Gender": "Sex",
    "Physical_Condition": "Physical condition",
    "General_Appearance": "General appearance",
    "Mucous_Membrane_State": "Mucous membrane",
    "Vomiting": "Vomiting",
    "Diarrhea": "Diarrhoea",
    "Bloody_Stool": "Bloody stool",
    "Eye_Discharge": "Eye discharge",
    "Weakness": "Weakness",
    "Weight_Loss": "Weight loss",
    "Anorexia": "Loss of appetite",
    "Skin_Lesions": "Skin lesions",
    "Parasite_Presence": "Visible parasite presence",
    "Tick_Infection": "Tick infestation observed",
    "Flea_Infection": "Flea infestation observed",
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    :root {
        --navy: #102A43;
        --navy-2: #163B58;
        --teal: #13A89E;
        --teal-soft: #E7F8F6;
        --coral: #F26B5E;
        --ink: #243B53;
        --muted: #66788A;
        --line: #DDE7EE;
        --canvas: #F4F8FB;
    }

    html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
    h1, h2, h3, h4 { font-family: "Manrope", sans-serif !important; color: var(--navy); }

    .stApp { background: var(--canvas); }
    .block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 3rem; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102A43 0%, #123A4B 100%);
        border-right: 0;
    }
    [data-testid="stSidebar"] * { color: #F7FAFC; }
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(255,255,255,.05);
        border: 1px solid rgba(255,255,255,.09);
        border-radius: 12px;
        padding: .65rem .8rem;
        margin-bottom: .35rem;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(19,168,158,.18);
        border-color: rgba(53,211,194,.4);
    }

    .brand {
        display:flex; gap:.75rem; align-items:center; margin:.2rem 0 1.6rem;
    }
    .brand-mark {
        width:44px; height:44px; border-radius:14px; display:grid; place-items:center;
        background: linear-gradient(135deg, #25C2B6, #0F8D86);
        box-shadow: 0 8px 20px rgba(0,0,0,.22); font-size:1.35rem;
    }
    .brand-name { font:800 1.05rem "Manrope"; color:white; line-height:1.1; }
    .brand-sub { color:#B8D6DE; font-size:.76rem; margin-top:.2rem; }

    .hero {
        position:relative; overflow:hidden; border-radius:26px; padding:2.7rem 3rem;
        background: linear-gradient(120deg, #102A43 0%, #164E63 65%, #0F766E 100%);
        box-shadow: 0 18px 45px rgba(16,42,67,.15); color:white; margin-bottom:1.35rem;
    }
    .hero:after {
        content:""; position:absolute; width:330px; height:330px; border-radius:50%;
        right:-90px; top:-130px; background:rgba(255,255,255,.08);
    }
    .eyebrow {
        display:inline-block; color:#9FF2E8; letter-spacing:.12em; text-transform:uppercase;
        font-weight:700; font-size:.75rem; margin-bottom:.8rem;
    }
    .hero h1 { color:white !important; font-size:2.55rem; line-height:1.13; margin:.1rem 0 .7rem; }
    .hero p { color:#D7EEF1; max-width:720px; font-size:1.04rem; line-height:1.65; margin:0; }

    .notice {
        display:flex; gap:.8rem; align-items:flex-start; padding:1rem 1.15rem;
        border:1px solid #F3D3A3; background:#FFF9ED; border-radius:14px;
        color:#704B18; margin: .25rem 0 1.35rem;
    }
    .notice strong { color:#5E3D11; }

    .metric-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin:1rem 0 1.5rem; }
    .metric-card {
        background:white; border:1px solid var(--line); border-radius:18px; padding:1.25rem;
        box-shadow:0 7px 22px rgba(16,42,67,.05);
    }
    .metric-label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
    .metric-value { color:var(--navy); font:800 1.75rem "Manrope"; margin:.35rem 0 .15rem; }
    .metric-note { color:#738596; font-size:.84rem; }

    .section-card {
        background:white; border:1px solid var(--line); border-radius:20px;
        padding:1.35rem 1.5rem .7rem; box-shadow:0 8px 26px rgba(16,42,67,.05);
        margin-bottom:1rem;
    }
    .step-pill {
        display:inline-block; color:#087D77; background:var(--teal-soft); border-radius:999px;
        padding:.35rem .65rem; font-size:.76rem; font-weight:700; margin-bottom:.5rem;
    }
    .helper { color:var(--muted); font-size:.91rem; margin:-.2rem 0 .8rem; }

    div[data-baseweb="select"] > div, .stNumberInput input {
        background:#F8FBFD !important; border-color:#D7E3EA !important; border-radius:11px !important;
    }
    .stButton > button, .stFormSubmitButton > button {
        min-height:3.15rem; border-radius:12px; font-weight:700; border:0;
        background:linear-gradient(90deg, #13A89E, #0F8D86); color:white;
        box-shadow:0 8px 20px rgba(19,168,158,.22);
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background:linear-gradient(90deg, #0F948B, #0B7772); color:white; border:0;
    }

    .result-wrap {
        background:white; border:1px solid var(--line); border-radius:22px; padding:1.5rem;
        box-shadow:0 14px 38px rgba(16,42,67,.09); margin-top:1.3rem;
    }
    .result-tag {
        display:inline-block; padding:.35rem .65rem; border-radius:999px; background:#E7F8F6;
        color:#087D77; font-size:.75rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em;
    }
    .result-name { color:var(--navy); font:800 2.1rem "Manrope"; margin:.65rem 0 .15rem; }
    .result-copy { color:var(--muted); margin-bottom:1.2rem; }
    .prob-row { margin:.9rem 0; }
    .prob-label { display:flex; justify-content:space-between; color:var(--ink); font-weight:600; margin-bottom:.35rem; }
    .prob-track { height:12px; border-radius:99px; background:#E8EFF3; overflow:hidden; }
    .prob-fill-teal { height:100%; background:linear-gradient(90deg,#13A89E,#36CDBF); border-radius:99px; }
    .prob-fill-coral { height:100%; background:linear-gradient(90deg,#F26B5E,#FF9A7B); border-radius:99px; }
    .next-step {
        background:#EEF6FF; border-left:4px solid #2B6CB0; border-radius:10px;
        padding:1rem 1.1rem; color:#234E72; margin-top:1.25rem;
    }
    .small-print { color:#8293A3; font-size:.78rem; line-height:1.55; margin-top:.9rem; }

    .about-card {
        background:white; border:1px solid var(--line); border-radius:18px; padding:1.3rem 1.4rem;
        height:100%; box-shadow:0 7px 22px rgba(16,42,67,.04);
    }
    .about-card h4 { margin-top:0; }
    .about-card p { color:var(--muted); line-height:1.65; }

    @media(max-width:800px) {
        .hero { padding:2rem 1.4rem; }
        .hero h1 { font-size:2rem; }
        .metric-grid { grid-template-columns:1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    if MODEL_TEXT_PATH.exists():
        model_bytes = base64.b64decode(MODEL_TEXT_PATH.read_text(encoding="ascii"))
        return joblib.load(BytesIO(model_bytes))
    raise FileNotFoundError("No model bundle was found.")


try:
    bundle = load_model()
except Exception:
    st.error("The trained screening model could not be loaded. Confirm that the model_outputs folder is deployed with the app.")
    st.stop()

preprocessor = bundle["preprocessor"]
model = bundle["model"]
encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
category_options = {
    feature: [str(value) for value in categories]
    for feature, categories in zip(bundle["categorical_features"], encoder.categories_)
}


def default_index(options):
    preferred = "Not recorded"
    return options.index(preferred) if preferred in options else 0


def select_feature(feature, key):
    options = category_options[feature]
    return st.selectbox(
        LABELS.get(feature, feature.replace("_", " ")),
        options=options,
        index=default_index(options),
        key=key,
    )


def hero(title, copy, eyebrow):
    st.markdown(
        f"""
        <section class="hero">
            <span class="eyebrow">{eyebrow}</span>
            <h1>{title}</h1>
            <p>{copy}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def safety_notice():
    st.markdown(
        """
        <div class="notice">
            <span>⚕️</span>
            <div><strong>Academic screening prototype.</strong> This tool cannot confirm a diagnosis,
            identify a causative organism, recommend medication or replace a veterinarian.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def open_screening():
    st.session_state["nav"] = "New screening"


def render_home():
    hero(
        "Smarter preliminary screening for canine illness",
        "A focused decision-support prototype that compares patterns associated with the clinic-recorded classes of tick fever and gastroenteritis.",
        "CanineCare AI • Research prototype",
    )
    safety_notice()
    st.markdown(
        """
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Study records</div>
                <div class="metric-value">197</div>
                <div class="metric-note">Target records analysed</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Selected model</div>
                <div class="metric-value">RBF-SVM</div>
                <div class="metric-note">Chosen by five-fold macro F1</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Hold-out macro F1</div>
                <div class="metric-value">0.812</div>
                <div class="metric-note">Internal test performance</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, middle, right = st.columns(3)
    cards = [
        (
            left,
            "01",
            "Record the profile",
            "Enter only information present in the veterinary record. Unavailable observations should remain “Not recorded.”",
        ),
        (
            middle,
            "02",
            "Review observed signs",
            "Provide the general, gastrointestinal and exposure-related observations used during model development.",
        ),
        (
            right,
            "03",
            "Generate a screen",
            "The model returns one indicated class and both estimated probabilities, followed by a veterinary referral reminder.",
        ),
    ]
    for column, step, title, copy in cards:
        with column:
            st.markdown(
                f'<div class="about-card"><span class="step-pill">STEP {step}</span><h4>{title}</h4><p>{copy}</p></div>',
                unsafe_allow_html=True,
            )
    st.write("")
    st.button(
        "Start a new screening →",
        use_container_width=True,
        on_click=open_screening,
    )


def render_result(predicted_class, probabilities):
    probability_map = {
        name: float(value)
        for name, value in zip(bundle["class_names"], probabilities)
    }
    tick = probability_map["Tick fever"] * 100
    gastro = probability_map["Gastroenteritis"] * 100
    st.markdown(
        f"""
        <div class="result-wrap">
            <span class="result-tag">Preliminary model result</span>
            <div class="result-name">{predicted_class}</div>
            <div class="result-copy">The model found the entered record more similar to the
            <strong>{predicted_class}</strong> class in the study data.</div>

            <div class="prob-row">
                <div class="prob-label"><span>Tick fever</span><span>{tick:.1f}%</span></div>
                <div class="prob-track"><div class="prob-fill-teal" style="width:{tick:.1f}%"></div></div>
            </div>
            <div class="prob-row">
                <div class="prob-label"><span>Gastroenteritis</span><span>{gastro:.1f}%</span></div>
                <div class="prob-track"><div class="prob-fill-coral" style="width:{gastro:.1f}%"></div></div>
            </div>

            <div class="next-step"><strong>Recommended next step:</strong> Arrange an examination
            by a qualified veterinarian and complete any appropriate laboratory confirmation.
            Do not begin treatment from this result alone.</div>
            <div class="small-print">These values are model estimates from retrospective records,
            not clinical certainty. “Tick fever” is a broad recorded label and does not identify
            a particular tick-borne organism.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_screening():
    hero(
        "New canine screening",
        "Complete the three short sections using recorded observations. The model uses 17 demographic and clinical variables.",
        "Tick fever • Gastroenteritis",
    )
    safety_notice()

    with st.form("screening_form", clear_on_submit=False):
        values = {}

        st.markdown('<div class="section-card"><span class="step-pill">1 OF 3</span><h3>Dog profile</h3><p class="helper">Basic information and general examination findings.</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            values["Age_Months"] = st.number_input(
                LABELS["Age_Months"], min_value=0.0, max_value=300.0, value=12.0, step=1.0
            )
            values["Physical_Condition"] = select_feature("Physical_Condition", "physical")
        with c2:
            values["Breed"] = select_feature("Breed", "breed")
            values["General_Appearance"] = select_feature("General_Appearance", "appearance")
        with c3:
            values["Gender"] = select_feature("Gender", "gender")
            values["Mucous_Membrane_State"] = select_feature("Mucous_Membrane_State", "mucous")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card"><span class="step-pill">2 OF 3</span><h3>Gastrointestinal and systemic signs</h3><p class="helper">Choose only what was documented in the case record.</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            values["Vomiting"] = select_feature("Vomiting", "vomiting")
            values["Weakness"] = select_feature("Weakness", "weakness")
        with c2:
            values["Diarrhea"] = select_feature("Diarrhea", "diarrhea")
            values["Weight_Loss"] = select_feature("Weight_Loss", "weight_loss")
        with c3:
            values["Bloody_Stool"] = select_feature("Bloody_Stool", "bloody_stool")
            values["Anorexia"] = select_feature("Anorexia", "anorexia")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card"><span class="step-pill">3 OF 3</span><h3>Exposure and additional observations</h3><p class="helper">External signs and parasite-related observations.</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            values["Eye_Discharge"] = select_feature("Eye_Discharge", "eye")
            values["Parasite_Presence"] = select_feature("Parasite_Presence", "parasite")
        with c2:
            values["Skin_Lesions"] = select_feature("Skin_Lesions", "skin")
            values["Tick_Infection"] = select_feature("Tick_Infection", "tick")
        with c3:
            values["Flea_Infection"] = select_feature("Flea_Infection", "flea")
        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "Generate preliminary result",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        row = pd.DataFrame([values], columns=bundle["feature_columns"])
        transformed = preprocessor.transform(row)
        predicted_index = int(model.predict(transformed)[0])
        probabilities = model.predict_proba(transformed)[0]
        predicted_class = bundle["class_names"][predicted_index]
        st.session_state["last_result"] = (predicted_class, probabilities)

    if "last_result" in st.session_state:
        predicted_class, probabilities = st.session_state["last_result"]
        render_result(predicted_class, probabilities)


def render_about():
    hero(
        "About the research prototype",
        "How the data, selected model and safeguards fit together.",
        "Transparent by design",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="about-card">
                <h4>What the system does</h4>
                <p>It compares a single entered dog record with patterns learned from 197
                retrospective records labelled tick fever or gastroenteritis.</p>
                <p>The selected RBF-SVM was chosen using five-fold stratified cross-validation
                and macro F1, with random oversampling applied only to training data.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="about-card">
                <h4>What the system does not do</h4>
                <p>It does not diagnose every canine disease, identify a pathogen, determine
                disease severity, prescribe medication or establish whether an illness is zoonotic.</p>
                <p>The result should be treated as a research output requiring veterinary
                examination and appropriate confirmatory testing.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.write("")
    st.markdown("### Internal evaluation summary")
    metrics = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Balanced accuracy", "Macro F1", "ROC-AUC", "PR-AUC"],
            "Score": [0.850, 0.864, 0.812, 0.946, 0.828],
        }
    )
    st.dataframe(metrics, hide_index=True, use_container_width=True)
    st.caption(
        "The test set contained 40 records, including only nine gastroenteritis cases. "
        "These figures are internal results and do not establish clinical readiness."
    )


with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">🐾</div>
            <div><div class="brand-name">CanineCare AI</div>
            <div class="brand-sub">Veterinary screening research</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    navigation = st.radio(
        "Navigation",
        ["Overview", "New screening", "About the model"],
        key="nav",
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("PROJECT")
    st.markdown("**Adeoti Dennis**  \nCPE/20/3372")
    st.caption("Federal University of Technology, Akure")
    st.markdown("---")
    st.caption("RBF-SVM • 17 predictors • 2 classes")

if navigation == "Overview":
    render_home()
elif navigation == "New screening":
    render_screening()
else:
    render_about()
