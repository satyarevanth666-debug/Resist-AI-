from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
CLEANED_DATA_PATH = MODELS_DIR / "cleaned_data.csv"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from recommend import rank_antibiotics_for_bacteria
from train import train_and_save_model
from visualize import (
    class_distribution_chart,
    confusion_matrix_chart,
    feature_importance_chart,
    resistance_heatmap,
)


st.set_page_config(
    page_title="ResistAI - Antibiotic Resistance Intelligence",
    page_icon="🧬",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --card-bg: rgba(17, 24, 39, 0.62);
            --card-border: rgba(148, 163, 184, 0.22);
            --text-soft: #cbd5e1;
            --brand: #22d3ee;
            --bg-base: #0e1117;
        }
        .stApp {
            background:
                radial-gradient(1200px 500px at 12% -8%, rgba(56, 189, 248, 0.16), transparent 62%),
                radial-gradient(900px 420px at 100% 0%, rgba(99, 102, 241, 0.14), transparent 60%),
                linear-gradient(140deg, #0e1117 0%, #0b1220 45%, #121a2a 100%);
            color: #e5e7eb;
        }
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1450px; }
        .main-title { font-size: 3.1rem; font-weight: 800; margin-bottom: 0.15rem; letter-spacing: 0.35px; }
        .subtitle { font-size: 1.08rem; color: var(--text-soft); margin-bottom: 1rem; }
        .hero {
            background: linear-gradient(108deg, rgba(56,189,248,0.20), rgba(79,70,229,0.14), rgba(2,6,23,0.40));
            border: 1px solid rgba(125,211,252,0.28);
            border-radius: 22px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.30);
        }
        .pill-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
        .pill {
            background: rgba(15,23,42,0.55);
            border: 1px solid rgba(148,163,184,0.25);
            color: #dbeafe;
            border-radius: 999px;
            padding: 3px 10px;
            font-size: 0.8rem;
        }
        .glass-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 18px;
            padding: 1rem 1rem 0.8rem 1rem;
            margin-bottom: 0.85rem;
            backdrop-filter: blur(8px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.24);
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        }
        .glass-card:hover {
            transform: translateY(-2px);
            border-color: rgba(125,211,252,0.35);
            box-shadow: 0 14px 30px rgba(2, 6, 23, 0.45);
        }
        .kpi-card {
            background: rgba(15, 23, 42, 0.66);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 14px;
            padding: 0.65rem 0.9rem;
            margin-bottom: 0.9rem;
            transition: transform 0.22s ease, border-color 0.22s ease;
        }
        .kpi-card:hover {
            transform: translateY(-1px);
            border-color: rgba(165, 180, 252, 0.44);
        }
        .kpi-label { color: #94a3b8; font-size: 0.8rem; }
        .kpi-value { color: #f8fafc; font-size: 1.1rem; font-weight: 700; }
        .section-title { margin: 0.8rem 0 0.45rem 0; font-weight: 800; font-size: 1.18rem; letter-spacing: 0.2px; }
        [data-testid="stMetric"] {
            background: rgba(2, 6, 23, 0.56);
            border: 1px solid rgba(148,163,184,0.2);
            border-radius: 12px;
            padding: 0.55rem 0.7rem;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 12px;
            overflow: hidden;
        }
        .small-note {
            color: #93c5fd;
            font-size: 0.82rem;
            padding-top: 4px;
        }
        div[data-testid="stButton"] > button {
            border-radius: 12px !important;
            border: 1px solid rgba(125,211,252,0.4) !important;
            background: linear-gradient(92deg, rgba(14,165,233,0.82), rgba(59,130,246,0.82)) !important;
            color: white !important;
            font-weight: 650 !important;
            transition: transform .2s ease, box-shadow .2s ease !important;
            box-shadow: 0 8px 16px rgba(2, 132, 199, 0.24);
        }
        div[data-testid="stButton"] > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 22px rgba(2, 132, 199, 0.36);
        }
        .result-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 6px 14px;
            font-weight: 700;
            margin-bottom: 8px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .result-red { background: rgba(239,68,68,0.20); color: #fecaca; }
        .result-green { background: rgba(34,197,94,0.20); color: #bbf7d0; }
        .result-yellow { background: rgba(234,179,8,0.20); color: #fde68a; }
        .rank-item {
            background: rgba(2,6,23,0.5);
            border: 1px solid rgba(148,163,184,0.20);
            border-radius: 12px;
            padding: 10px;
            margin-bottom: 8px;
            transition: border-color .2s ease;
        }
        .rank-item:hover { border-color: rgba(125,211,252,0.36); }
        .rank-head { display: flex; justify-content: space-between; gap: 10px; align-items: center; }
        .rank-name { font-weight: 650; color: #f1f5f9; }
        .rank-score { color: #93c5fd; font-size: 0.84rem; }
        .rank-bar { width: 100%; height: 8px; background: rgba(30,41,59,0.85); border-radius: 999px; margin-top: 8px; overflow: hidden; }
        .rank-fill { height: 100%; border-radius: 999px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_or_train_artifact():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)
        return artifact
    with st.spinner("No trained model found. Training pipeline is running..."):
        artifact, _ = train_and_save_model()
    return artifact


@st.cache_data
def load_cleaned_data():
    if CLEANED_DATA_PATH.exists():
        return pd.read_csv(CLEANED_DATA_PATH)
    return pd.DataFrame()


artifact = load_or_train_artifact()
cleaned_df = load_cleaned_data()

best_model = artifact["best_model"]
target_inverse = artifact["inverse_target_mapping"]
bacteria_encoder = artifact["encoders"]["bacteria_encoder"]
antibiotic_encoder = artifact["encoders"]["antibiotic_encoder"]

def style_fig(fig, height: int = 360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=54, b=20),
        height=height,
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def render_ranked_cards(rank_df: pd.DataFrame, positive: bool = True):
    palette = "#22c55e" if positive else "#ef4444"
    for idx, row in rank_df.reset_index(drop=True).iterrows():
        risk_pct = float(row["ResistanceProbability"] * 100)
        safe_score = 100 - risk_pct
        display_score = safe_score if positive else risk_pct
        st.markdown(
            f"""
            <div class="rank-item">
                <div class="rank-head">
                    <span class="rank-name">#{idx + 1} {row["Antibiotic"]}</span>
                    <span class="rank-score">{display_score:.1f}% {'Suitability' if positive else 'Risk'}</span>
                </div>
                <div class="rank-bar">
                    <div class="rank-fill" style="width:{max(min(display_score, 100), 0):.1f}%; background:{palette};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def infer_prediction(selected_bacteria: str, selected_antibiotic: str):
    b_enc = int(bacteria_encoder.transform([selected_bacteria])[0])
    a_enc = int(antibiotic_encoder.transform([selected_antibiotic])[0])
    probs = best_model.predict_proba([[b_enc, a_enc]])[0]
    pred_code = int(best_model.predict([[b_enc, a_enc]])[0])
    pred_label = target_inverse[pred_code]
    confidence = float(max(probs))
    sorted_probs = sorted([float(p) for p in probs], reverse=True)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
    return pred_label, probs, confidence, margin


def build_prediction_report(
    selected_bacteria: str,
    selected_antibiotic: str,
    pred_label: str,
    confidence: float,
    probs,
    recommended_df: pd.DataFrame,
    avoid_df: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Bacteria": selected_bacteria,
                "SelectedAntibiotic": selected_antibiotic,
                "PredictedClass": pred_label,
                "Confidence": round(confidence, 4),
                "SusceptibleProbability": round(float(probs[0]), 4),
                "ResistantProbability": round(float(probs[1]), 4),
                "IntermediateProbability": round(float(probs[2]) if len(probs) > 2 else 0.0, 4),
                "TopRecommended": ", ".join(recommended_df["Antibiotic"].head(3).tolist()),
                "TopAvoid": ", ".join(avoid_df["Antibiotic"].head(3).tolist()),
            }
        ]
    )


st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown('<div class="main-title">🧬 ResistAI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Intelligent Antibiotic Resistance Prediction & Recommendation System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="pill-row">
        <span class="pill">Real-time Prediction</span>
        <span class="pill">Clinical Decision Support</span>
        <span class="pill">Explainable AI</span>
        <span class="pill">Interactive Analytics</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Best Model</div><div class="kpi-value">{artifact["best_model_name"].replace("_", " ").title()}</div></div>',
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Bacteria Classes</div><div class="kpi-value">{len(bacteria_encoder.classes_)}</div></div>',
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Antibiotic Classes</div><div class="kpi-value">{len(antibiotic_encoder.classes_)}</div></div>',
        unsafe_allow_html=True,
    )
with k4:
    samples = len(cleaned_df) if not cleaned_df.empty else "N/A"
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Clean Samples</div><div class="kpi-value">{samples}</div></div>',
        unsafe_allow_html=True,
    )

left, right = st.columns([1, 1.5], gap="large")

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

with left:
    _, center, _ = st.columns([0.08, 0.84, 0.08])
    with center:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧪 Input Panel</div>', unsafe_allow_html=True)
        bacteria = st.selectbox(
            "🦠 Select Bacteria",
            options=sorted(list(bacteria_encoder.classes_)),
            help="Choose the bacterial organism for resistance prediction.",
        )
        antibiotic = st.selectbox(
            "💊 Select Antibiotic",
            options=sorted(list(antibiotic_encoder.classes_)),
            help="Choose an antibiotic to evaluate predicted susceptibility.",
        )
        do_predict = st.button("🚀 Predict Resistance", use_container_width=True, help="Run model inference for current inputs.")
        st.markdown('<div class="small-note">Tip: compare multiple antibiotics for the same bacteria to identify safer options.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧠 Prediction Output</div>', unsafe_allow_html=True)
    if do_predict:
        with st.spinner("Generating prediction and confidence scores..."):
            pred_label, probs, confidence, margin = infer_prediction(bacteria, antibiotic)

            status_map = {
                "Resistant": ("🔴 Resistant", "result-red", "High resistance risk detected."),
                "Susceptible": ("🟢 Susceptible", "result-green", "Likely effective treatment option."),
                "Intermediate": ("🟡 Intermediate", "result-yellow", "Moderate uncertainty in efficacy."),
            }
            display_label, badge_class, expl = status_map[pred_label]
            st.markdown(f'<div class="result-badge {badge_class}">{display_label}</div>', unsafe_allow_html=True)
            st.success(f"Model confidence: {confidence * 100:.2f}%", icon="✅")
            st.caption(f"This prediction is influenced by bacteria profile and antibiotic history patterns. {expl}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Susceptible", f"{probs[0] * 100:.2f}%")
            c2.metric("Resistant", f"{probs[1] * 100:.2f}%")
            c3.metric(
                "Intermediate",
                f"{(probs[2] * 100 if len(probs) > 2 else 0.0):.2f}%",
            )
            if confidence < 0.65 or margin < 0.15:
                st.warning(
                    "Low certainty detected for this case. Consider reviewing multiple antibiotic options and confirm with lab evidence.",
                    icon="⚠️",
                )
            st.session_state["last_prediction"] = {
                "display_label": display_label,
                "confidence": confidence,
            }
    elif st.session_state["last_prediction"] is not None:
        last = st.session_state["last_prediction"]
        st.info(
            f"Last prediction: {last['display_label']} | Confidence: {last['confidence'] * 100:.2f}%",
            icon="🧾",
        )
    else:
        st.info("Select inputs and click **Predict Resistance** to see outcomes.", icon="ℹ️")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-title">💊 Recommendation Panel</div>', unsafe_allow_html=True)
r1, r2 = st.columns(2)
recommended_df = pd.DataFrame()
avoid_df = pd.DataFrame()
try:
    recommended_df, avoid_df = rank_antibiotics_for_bacteria(bacteria, artifact, top_n=5)
    with r1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("✅ Recommended (Low Resistance)")
        st.caption("Ranked by highest predicted suitability.")
        render_ranked_cards(recommended_df, positive=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("❌ Avoid (High Resistance)")
        st.caption("Ranked by highest predicted resistance risk.")
        render_ranked_cards(avoid_df, positive=False)
        st.markdown("</div>", unsafe_allow_html=True)
except Exception as ex:
    st.error(f"Recommendation unavailable: {ex}")

st.markdown('<div class="section-title">🧪 What-If Antibiotic Comparison</div>', unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
compare_list = st.multiselect(
    "Select antibiotics to compare for this bacteria",
    options=sorted(list(antibiotic_encoder.classes_)),
    default=sorted(list(antibiotic_encoder.classes_))[:3],
    help="Compare predictions side-by-side for multiple options.",
)
if compare_list:
    comparison_rows = []
    for med in compare_list[:8]:
        pred_label, med_probs, conf, _ = infer_prediction(bacteria, med)
        comparison_rows.append(
            {
                "Antibiotic": med,
                "PredictedClass": pred_label,
                "Confidence": round(conf * 100, 2),
                "Susceptible%": round(float(med_probs[0]) * 100, 2),
                "Resistant%": round(float(med_probs[1]) * 100, 2),
                "Intermediate%": round(float(med_probs[2]) * 100 if len(med_probs) > 2 else 0.0, 2),
            }
        )
    comp_df = pd.DataFrame(comparison_rows).sort_values("Resistant%", ascending=True)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.caption("Lower Resistant% and higher Susceptible% are usually better treatment choices.")
else:
    st.info("Select at least one antibiotic to compare.", icon="ℹ️")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-title">📥 Export Reports</div>', unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
if st.session_state["last_prediction"] is not None and not recommended_df.empty and not avoid_df.empty:
    report_label, report_probs, report_conf, _ = infer_prediction(bacteria, antibiotic)
    report_df = build_prediction_report(
        bacteria,
        antibiotic,
        report_label,
        report_conf,
        report_probs,
        recommended_df,
        avoid_df,
    )
    cexp1, cexp2 = st.columns(2)
    with cexp1:
        st.download_button(
            "⬇️ Download Prediction Summary (CSV)",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="resistai_prediction_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with cexp2:
        full_report = {
            "prediction_summary": report_df.to_dict(orient="records"),
            "recommended": recommended_df.to_dict(orient="records"),
            "avoid": avoid_df.to_dict(orient="records"),
        }
        st.download_button(
            "⬇️ Download Full Report (JSON)",
            data=json.dumps(full_report, indent=2).encode("utf-8"),
            file_name="resistai_full_report.json",
            mime="application/json",
            use_container_width=True,
        )
else:
    st.info("Run a prediction first to enable report downloads.", icon="📄")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-title">📈 Analytics Dashboard</div>', unsafe_allow_html=True)
if cleaned_df.empty:
    st.error(
        "Cleaned data not found. Run training once with `python src/train.py` to generate analytics artifacts."
    )
else:
    a1, a2 = st.columns(2)
    with a1:
        st.plotly_chart(style_fig(class_distribution_chart(cleaned_df)), use_container_width=True)
    with a2:
        st.plotly_chart(style_fig(resistance_heatmap(cleaned_df)), use_container_width=True)

    b1, b2 = st.columns(2)
    with b1:
        cm = artifact["confusion_matrices"][artifact["best_model_name"]]
        st.plotly_chart(style_fig(confusion_matrix_chart(cm)), use_container_width=True)
    with b2:
        fi = artifact.get("feature_importance")
        if fi:
            st.plotly_chart(style_fig(feature_importance_chart(fi)), use_container_width=True)
        else:
            st.info("Feature importance is unavailable for current model.")

st.markdown('<div class="section-title">🔬 Explainability Panel</div>', unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
try:
    import shap

    sample = [[int(bacteria_encoder.transform([bacteria])[0]), int(antibiotic_encoder.transform([antibiotic])[0])]]
    feature_names = ["BacteriaEncoded", "AntibioticEncoded"]

    if hasattr(best_model, "feature_importances_") or "xgb" in artifact["best_model_name"]:
        explainer = shap.TreeExplainer(best_model)
    else:
        explainer = shap.Explainer(best_model.predict_proba, sample)

    shap_values = explainer(sample)
    values = shap_values.values
    if values.ndim == 3:
        values = values[0, :, 1]
    elif values.ndim == 2:
        values = values[0]

    shap_df = pd.DataFrame({"Feature": feature_names, "SHAP Value": values})
    st.plotly_chart(
        style_fig(
            px.bar(shap_df, x="Feature", y="SHAP Value", color="SHAP Value", title="SHAP Local Explanation"),
            height=340,
        ),
        use_container_width=True,
    )
    st.caption(
        "This prediction is influenced by the encoded bacteria identity and selected antibiotic response patterns learned during training."
    )
except Exception as ex:
    st.info(f"SHAP explanation unavailable: {ex}")
st.markdown("</div>", unsafe_allow_html=True)
