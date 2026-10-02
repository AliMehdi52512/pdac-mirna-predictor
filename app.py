import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(
    page_title="PDAC Serum Diagnostic Platform",
    page_icon="🧬",
    layout="wide"
)

ASSET_DIR = "streamlit_app_assets"

@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(ASSET_DIR, "catboost_calibrated_model.pkl"))
    with open(os.path.join(ASSET_DIR, "optimal_threshold.json"), "r") as f:
        threshold = json.load(f)["optimal_threshold"]
    target_quantiles = np.load(os.path.join(ASSET_DIR, "target_quantiles_10.npy"))
    with open(os.path.join(ASSET_DIR, "panel_metadata.json"), "r") as f:
        metadata = json.load(f)
    demo_samples = pd.read_csv(os.path.join(ASSET_DIR, "sample_patient_inputs.csv"), index_col=0)
    return model, threshold, target_quantiles, metadata, demo_samples

model, threshold, target_quantiles, metadata, demo_samples = load_assets()

st.title("Pancreatic Ductal Adenocarcinoma (PDAC) Serum Diagnostic Platform")
st.markdown("**Calibrated Non-Invasive Diagnostic Tool:** Platt-calibrated CatBoost ensemble utilizing an experimentally validated 10-microRNA signature.")

# Sidebar Configuration
st.sidebar.header("🧬 Patient Input Configuration")
workflow = st.sidebar.selectbox("Input Method:", ["Load Holdout Demo Case", "Manual Expression Sliders"])

input_values = {}

if workflow == "Load Holdout Demo Case":
    patient_choice = st.sidebar.selectbox("Select Patient Profile:", demo_samples.index.tolist())
    selected_row = demo_samples.loc[patient_choice]
    actual_label = selected_row.get("True_Clinical_Label", "Unknown")
    st.sidebar.info(f"**True Pathology Label:** {actual_label}")
    for mimat_id in metadata.keys():
        input_values[mimat_id] = float(selected_row[mimat_id])
else:
    st.sidebar.markdown("Adjust Normalized Signal Values:")
    for mimat_id, meta in metadata.items():
        input_values[mimat_id] = st.sidebar.slider(
            f"{meta['mature_name']} ({mimat_id})",
            min_value=float(np.floor(meta['min_val'])),
            max_value=float(np.ceil(meta['max_val'])),
            value=float(round(meta['mean_val'], 2)),
            step=0.1
        )

# Inference Pipeline
input_df = pd.DataFrame([input_values])
ranks = input_df.rank(axis=1, method='min').astype(int) - 1
norm_patient = pd.DataFrame(target_quantiles[ranks], columns=input_df.columns)

prob_cancer = model.predict_proba(norm_patient)[:, 1][0]
is_malignant = prob_cancer >= threshold

col1, col2 = st.columns([1.1, 1.2])

with col1:
    st.subheader("Diagnostic Risk Assessment")
    st.metric(
        label="Predicted Malignancy Probability (Platt-Calibrated)", 
        value=f"{prob_cancer * 100:.2f}%", 
        delta=f"{(prob_cancer - threshold) * 100:+.2f}% vs Decision Cutoff"
    )
    st.progress(min(float(prob_cancer), 1.0))
    st.caption(f"Clinical Decision Cutoff: **{threshold * 100:.2f}%** (Derived via Youden\'s J Optimization)")
    
    if is_malignant:
        st.error("🚨 **POSITIVE: High Clinical Suspicion of Pancreatic Cancer**")
        st.markdown("""
        * **Recommended Action:** Immediate priority triaging for contrast-enhanced multiphase abdominal CT/MRI and endoscopic ultrasound-guided fine needle aspiration (EUS-FNA).
        """)
    else:
        st.success("✅ **NEGATIVE: Non-Cancer / Benign Profile Detected**")
        st.markdown("""
        * **Recommended Action:** Low circulating microRNA risk score. Routine surveillance according to baseline risk stratification.
        """)

with col2:
    st.subheader("Biomarker Profile & Target Genes")
    gene_map = {
        'hsa-miR-6784-5p': 'SMAD4, CDKN2A, MYC',
        'hsa-miR-6805-5p': 'KRAS, MAPK1, PIK3CA',
        'hsa-miR-7107-5p': 'TP53, ATM, BCL2',
        'hsa-miR-4787-5p': 'SMAD3, TGFBR1, TGFBR2',
        'hsa-miR-4485-3p': 'AKT1, MTOR, EGFR',
        'hsa-miR-4532':    'STAT3, IL6R, JAK2',
        'hsa-miR-6087':    'KRAS, BRAF, RAC1',
        'hsa-miR-1268a':   'CCND1, CDK6, RB1',
        'hsa-miR-6781-5p': 'ZEB1, SNAI1, TWIST1',
        'hsa-miR-4665-3p': 'PTEN, FOXO3, BAX'
    }
    
    table_data = []
    for mimat_id, meta in metadata.items():
        table_data.append({
            "miRNA": meta['mature_name'],
            "Log2 Signal": f"{input_values[mimat_id]:.2f}",
            "SHAP Impact": f"{meta['mean_shap']:.3f}",
            "Core Oncogenic Targets": gene_map.get(meta['mature_name'], "N/A")
        })
    
    # st.table renders standard static HTML without loading dynamic JS chunks
    st.table(pd.DataFrame(table_data))

st.markdown("---")
st.caption("PDAC Non-Invasive Diagnostics | Research Use Only")