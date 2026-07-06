import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import joblib
import json
import matplotlib.pyplot as plt
import shap

from src.preprocessing import preprocess_input
from src.predict import predict

# ==================== Load Model & Metadata ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PIPELINE = joblib.load(os.path.join(BASE_DIR, 'model', 'model.pkl'))
SCALER = PIPELINE.named_steps['scaler']
MODEL = PIPELINE.named_steps['model']

COEF = MODEL.coef_

with open(os.path.join(BASE_DIR, 'model', 'feature_names.json'), 'r', encoding='utf-8') as f:
    FEATURE_NAMES = json.load(f)

with open(os.path.join(BASE_DIR, 'model', 'class_labels.json'), 'r', encoding='utf-8') as f:
    CLASS_INFO = json.load(f)

train_stats_path = os.path.join(BASE_DIR, 'model', 'train_stats.json')
TRAIN_STATS = pd.read_json(train_stats_path) if os.path.exists(train_stats_path) else None

background_path = os.path.join(BASE_DIR, 'model', 'background.npy')
BACKGROUND = np.load(background_path) if os.path.exists(background_path) else None

# Define class names - FULL version for detail display
CLASS_NAMES_FULL = [
    'Cluster 0: Low Burden — Complexity Preserved — Cerebello-Thalamic Reorganization',
    'Cluster 1: High Clinical Burden — Low Complexity — Widespread Hypoconnectivity',
    'Cluster 2: High Reactivity — High Dispersion — Attention Control Network Reorganization',
    'Cluster 3: Peripheral Low Deviation — Central Integration Preserved'
]

# Define class names - SHORT version for table headers and exports
CLASS_NAMES_SHORT = ['Cluster0', 'Cluster1', 'Cluster2', 'Cluster3']

# ==================== Multi-Dimensional Profile Descriptions ====================
# Structured descriptions for each cluster - used for patient profiling

CLUSTER_PROFILES = {
    0: {
        "tagline": "Early-Stage Compensatory Insomnia",
        "autonomic_summary": "Relatively preserved autonomic complexity with elevated multiscale entropy (MAE_1) and increased MeanNN, suggesting maintained regulatory flexibility and slower-timescale adaptive processes.",
        "symptom_description": "Youngest age distribution with relatively mild clinical symptom burden. PSQI and ISI may be mildly elevated. Anxiety-depression scores (SAS, SDS) are generally lower than other clusters. Daytime fatigue (FSS) is moderate.",
        "network_signature": "No widespread pathological hypoconnectivity vs. healthy controls. Instead, shows 13 consistent ALPHA-band enhancement edges concentrated in bilateral cerebellar internal connections, cerebello-thalamic pathways, and hippocampal-superior parietal circuits. This suggests COMPENSATORY reorganization rather than degeneration.",
        "cross_scale_coupling": "Energy-entropy–ventral striatum–thalamus association: peripheral autonomic complexity (AttentionEntropy, MAE_1) correlates with enhanced ventral striatum–thalamus connectivity, suggesting reward-motivation circuit compensation for mild peripheral autonomic changes.",
        "clinical_implication": "Likely represents an early or compensatory stage of insomnia pathophysiology. Protective mechanisms (enhanced autonomic complexity and cerebellar-subcortical connectivity) are maintaining relatively intact daytime function. May benefit from non-pharmacological interventions targeting sleep hygiene and circadian stabilization."
    },
    1: {
        "tagline": "Decompensated Hyperarousal — Most Severe Endotype",
        "autonomic_summary": "Widespread HRV suppression: decreased SDNN, RMSSD, pNN50, SD1, and AttentionEntropy. Reduced overall autonomic variability and diminished parasympathetic-related short-term fluctuations. Impaired physiological complexity indicating autonomic regulatory reserve depletion.",
        "symptom_description": "Highest scores across all six clinical scales (PSQI, ISI, SAS, SDS, HAS, FSS all highest among clusters). Highest sedative-hypnotic medication dependency. Elevated risks of somatic symptoms: chest tightness, headache, fatigue, dry mouth, forgetfulness, irritability, palpitations.",
        "network_signature": "Most severe central network impairment: 32 consistent REDUCTION edges in alpha band with NO enhancement edges — pure uncompensated hypoconnectivity. Extensive cerebello-thalamic decoupling, cerebello-basal ganglia impairment, cerebello-somatomotor dysfunction, and subcortical-brainstem disconnection. Cross-band (delta, alpha, beta) systematic hypoconnectivity vs. healthy controls.",
        "cross_scale_coupling": "Long-term variability–cerebellum–thalamus association: SDANN negatively correlates with cerebellar lobule X–thalamic VPL connectivity (LOOCV R²=0.361), suggesting impaired autonomic feedback loops through the cerebello-thalamic pathway and reduced long-term heart rate variability.",
        "clinical_implication": "Likely corresponds to the most refractory insomnia population in clinical practice — 'decompensated hyperarousal' where chronic central hyperarousal has exhausted autonomic reserves. Requires the most intensive intervention, potentially combining pharmacotherapy with neuromodulation approaches targeting the cerebello-thalamic-cortical axis."
    },
    2: {
        "tagline": "Hyperreactive Autonomic Endotype",
        "autonomic_summary": "Distinct 'high reactivity, high dispersion' pattern: elevated RMSSD, pNN50, SD1, SD2, SD1/SD2, C2d, and multiple information entropy indices synchronously. This represents abnormally expanded beat-to-beat fluctuation amplitude and decreased autonomic stability — upregulated response gain rather than enhanced regulatory capacity.",
        "symptom_description": "Intermediate clinical severity across scales. Most distinctive phenotypic feature: oily facial skin (likely sympathetic-sebaceous axis hyperactivity). PSQI, ISI, SAS, SDS moderately elevated. HAS (hyperarousal) notably elevated.",
        "network_signature": "'Local reorganization, global preservation': no significant whole-brain network differences vs. healthy controls. One stable enhancement edge: right cerebellar lobule VI–left superior parietal lobule (cerebello-dorsal attention network interface), suggesting attention control system dysregulation.",
        "cross_scale_coupling": "Complexity–striatum–cerebellar association: Approximate Entropy shows strongest predictive relationship with cerebellar lobule X–putamen connectivity (LOOCV R²=0.523, best in cohort), suggesting basal ganglia–cerebellar circuit-mediated arousal maintenance abnormalities. RMSSD correlates with cerebellar Crus 2–orbitofrontal connectivity.",
        "clinical_implication": "Peripheral autonomic hyper-reactivity may represent the projection of persistent central threat-monitoring system activation. The sympathetic-sebaceous axis phenotype is unique. Intervention targeting sympathetic hyperactivity (e.g., relaxation training, heart rate variability biofeedback) may be particularly beneficial."
    },
    3: {
        "tagline": "Age-Related Autonomic Decline Endotype",
        "autonomic_summary": "Lowest HRV phenotype across time-domain, frequency-domain, and nonlinear indices, likely reflecting age-related autonomic regulatory decline superimposed on insomnia pathology. Natural aging of sleep homeostasis and circadian rhythm systems contributes to the peripheral phenotype.",
        "symptom_description": "Oldest patient group with mildest subjective symptom burden. ISI, PSQI, HAS, SDS significantly but modestly elevated relative to healthy controls. FSS (fatigue) relatively low. Lower hyperarousal scores (HAS) compared to Cluster 1 and 2.",
        "network_signature": "'Peripheral low deviation, central integration preserved': no significant whole-brain network differences vs. healthy controls. One consistent enhancement edge: left cerebellar Crus I–Crus II internal connectivity — local functional reorganization within the posterior cerebellar cognitive-emotion processing network, confined to cerebellum without cross-regional extension.",
        "cross_scale_coupling": "Nonlinear dynamics–globus pallidus–cerebellar association: BubbleEntropy correlates with bilateral pallidum–cerebellar connectivity. ISI positively correlates with cerebellar lobule X–thalamic VPL connectivity, suggesting that subjective insomnia severity in elderly patients may relate to somatosensory processing pathway function — abnormal somatosensory integration (e.g., physical discomfort hypersensitivity) may be an important source of subjective distress.",
        "clinical_implication": "Phenotype may be primarily driven by age-related sleep architecture changes and natural autonomic decline rather than strong pathological hyperarousal. Clinical assessment should carefully distinguish pathological autonomic changes from physiological aging. Interventions should prioritize age-appropriate approaches (e.g., cognitive behavioral therapy for insomnia, light therapy) over aggressive pharmacological sedation."
    }
}


# ==================== Page Configuration ====================
st.set_page_config(
    page_title="Insomnia Subtype Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; }
    .profile-box {
        background-color: #f0f4f8;
        border-left: 4px solid #1976d2;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .profile-section-title {
        font-weight: 600;
        color: #1565c0;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .disclaimer-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        border-radius: 4px;
        padding: 10px 14px;
        margin-top: 12px;
        font-size: 12px;
        color: #5d4037;
    }
    .tagline-badge {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Disclaimer Content ====================
DISCLAIMER_TEXT = """
**Disclaimer:** This prediction platform is a research tool developed for academic purposes and is currently in the validation phase. 
The outputs, including cluster labels and probability distributions, are algorithmic inferences based on machine learning models trained 
on specific research cohorts and **should not be used as the sole basis for clinical diagnosis or treatment decisions**. 
The multi-dimensional profile descriptions are derived from group-level statistical patterns and represent probabilistic tendencies rather 
than deterministic individual characteristics. Healthcare professionals should always integrate these results with comprehensive clinical 
assessments, patient history, and established diagnostic criteria. The developers assume no liability for clinical decisions made based 
on this tool's outputs. For research collaboration inquiries, please contact the corresponding authors.
"""


# ==================== Sidebar ====================
with st.sidebar:
    st.title("🧠 Insomnia Subtype Classifier")
    st.markdown("""
    **Based on Overnight HRV and Clinical Features**  
    Using a Logistic Regression model trained with nested cross-validation for automatic four-cluster classification.
    """)
    st.markdown("---")
    st.markdown("### Usage Steps")
    st.markdown("1. Upload an Excel/CSV file containing ECG features")
    st.markdown("2. System automatically preprocesses and aligns features")
    st.markdown("3. View prediction probabilities and cluster labels")
    st.markdown("4. Review multi-dimensional profile for selected samples")
    st.markdown("5. Download data table with predictions")
    st.markdown("---")
    
    example_path = os.path.join(BASE_DIR, 'model', 'example_input.xlsx')
    if os.path.exists(example_path):
        with open(example_path, 'rb') as f:
            st.download_button(
                label="📥 Download Example Input Template",
                data=f,
                file_name="example_input.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    st.markdown("---")
    st.info(f"Model Features: {len(FEATURE_NAMES)}", icon="ℹ️")
    
    # ---- Sidebar Disclaimer ----
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 11px; color: #888; line-height: 1.5;">
    <b>Disclaimer:</b> This tool is for <b>research purposes only</b> and does not constitute medical advice. 
    Clinical decisions should not be made solely based on this platform's outputs. 
    Always consult qualified healthcare professionals.
    </div>
    """, unsafe_allow_html=True)


# ==================== Main Interface ====================
st.title("Insomnia Patient Cluster Prediction Platform")
st.caption("Insomnia Cluster Classification via Resting-State EEG Functional Connectivity")

uploaded_file = st.file_uploader(
    "📤 Upload Data for Prediction (.xlsx or .csv)",
    type=["xlsx", "csv"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # -------------------- Read & Predict --------------------
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"File read failed: {e}")
        st.stop()

    with st.spinner("Performing feature alignment, standardization, and model inference..."):
        try:
            X_processed, row_ids = preprocess_input(df_raw)
            pred, proba, _ = predict(X_processed)
            class_names = CLASS_NAMES_SHORT
        except ValueError as ve:
            st.error(str(ve))
            st.stop()
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    # -------------------- Build Result Table (Probability Only) --------------------
    result_df = pd.DataFrame({
        'Sample_ID': row_ids
    })
    for i, name in enumerate(CLASS_NAMES_SHORT):
        result_df[f'Prob_{name}'] = np.round(proba[:, i], 4)

    display_cols = [f'Prob_{c}' for c in CLASS_NAMES_SHORT]

    # -------------------- Parallel Layout --------------------
    st.markdown("---")
    
    col_table, col_right = st.columns([1, 2.5])
    
    # ===== Left: Prediction Results =====
    with col_table:
        st.markdown("#### 📊 Prediction Results")
        
        format_dict = {f'Prob_{c}': "{:.2%}" for c in CLASS_NAMES_SHORT}
        try:
            styled = result_df[display_cols].style.background_gradient(
                subset=[f'Prob_{c}' for c in CLASS_NAMES_SHORT],
                cmap='YlGnBu',
                vmin=0, vmax=1
            ).format(format_dict)
            st.dataframe(
                styled,
                use_container_width=True,
                height=min(900, 35 * len(result_df) + 50)
            )
        except Exception:
            st.dataframe(result_df[display_cols], use_container_width=True)
    
    # ===== Right: Single Sample Analysis =====
    with col_right:
        st.markdown("#### 🔬 Single Sample Analysis")
        
        # First Row: Selector (Left) + Cluster Label (Right)
        c_select, c_label = st.columns(2)
        with c_select:
            selected_idx = st.selectbox("Select sample for details", result_df['Sample_ID'].tolist())
            sample_pos = row_ids.index(selected_idx)
            pred_label = int(pred[sample_pos])
            x_raw = X_processed[sample_pos].copy()
            x_std = SCALER.transform(X_processed[sample_pos:sample_pos+1])[0]
        
        with c_label:
            cluster_name_full = CLASS_NAMES_FULL[pred_label]
            st.markdown(f"""
            <div style="padding: 6px 0; width: 100%; text-align: center; border-radius: 6px; background-color: #1976d2; color: white; font-weight: bold; font-size: 14px; margin-top: 28px;">
                {cluster_name_full}
            </div>
            """, unsafe_allow_html=True)
        
        # ==================== NEW: Multi-Dimensional Profile Panel ====================
        profile = CLUSTER_PROFILES[pred_label]
        
        st.markdown("---")
        st.markdown("#### 🧬 Multi-Dimensional Patient Profile")
        
        # Tagline badge
        st.markdown(f'<div class="tagline-badge">{profile["tagline"]}</div>', unsafe_allow_html=True)
        
        # --- Autonomic Features ---
        st.markdown(f"""
        <div class="profile-box">
            <div class="profile-section-title">🫀 Autonomic Nervous System Features</div>
            <div style="font-size: 13px; line-height: 1.6; color: #333;">{profile["autonomic_summary"]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # --- Symptom Description ---
        st.markdown(f"""
        <div class="profile-box">
            <div class="profile-section-title">📝 Symptom & Clinical Phenotype</div>
            <div style="font-size: 13px; line-height: 1.6; color: #333;">{profile["symptom_description"]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # --- Network Signature ---
        st.markdown(f"""
        <div class="profile-box">
            <div class="profile-section-title">🧠 EEG Functional Connectivity Signature</div>
            <div style="font-size: 13px; line-height: 1.6; color: #333;">{profile["network_signature"]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # --- Cross-Scale Coupling ---
        st.markdown(f"""
        <div class="profile-box">
            <div class="profile-section-title">🔗 Cross-Scale Brain-Heart Coupling</div>
            <div style="font-size: 13px; line-height: 1.6; color: #333;">{profile["cross_scale_coupling"]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # --- Clinical Implication ---
        st.markdown(f"""
        <div class="profile-box" style="border-left-color: #388e3c;">
            <div class="profile-section-title" style="color: #2e7d32;">💡 Clinical Interpretation & Guidance</div>
            <div style="font-size: 13px; line-height: 1.6; color: #333;">{profile["clinical_implication"]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # ==================== End of Profile Panel ====================
        
        st.markdown("---")
        
        # Second Row: SHAP (Left) + Data Drift (Right)
        c_shap, c_drift = st.columns(2)
        
        with c_shap:
            st.caption("SHAP Feature Contribution")
            
            sv = None
            if BACKGROUND is not None:
                try:
                    shap_explainer = shap.Explainer(MODEL, BACKGROUND)
                    sv = shap_explainer(x_std.reshape(1, -1))
                except Exception:
                    pass
            
            if sv is not None:
                exp = shap.Explanation(
                    values=sv.values[0, :, pred_label],
                    base_values=sv.base_values[0, pred_label],
                    data=x_std,
                    feature_names=FEATURE_NAMES
                )
                fig = plt.figure(figsize=(5, 8))
                shap.plots.waterfall(exp, max_display=len(FEATURE_NAMES), show=False)
                fig = plt.gcf()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("Background data not found")
        
        with c_drift:
            st.caption("Data Drift Detection")
            
            z_scores = None
            if TRAIN_STATS is not None:
                means = TRAIN_STATS['mean'].values
                stds = TRAIN_STATS['std'].values
                z_scores = (x_raw - means) / stds
            
            if z_scores is not None:
                fig, ax = plt.subplots(figsize=(5, 4))
                colors = ['#d62728' if abs(z) > 2 else '#ff7f0e' if abs(z) > 1 else '#2ca02c' 
                          for z in z_scores]
                ax.barh(
                    range(len(FEATURE_NAMES)),
                    z_scores,
                    color=colors,
                    alpha=0.7,
                    edgecolor='none'
                )
                ax.set_yticks(range(len(FEATURE_NAMES)))
                ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
                ax.invert_yaxis()
                ax.set_xlabel("Z-score", fontsize=9)
                ax.axvline(x=0, color='black', linewidth=0.8)
                ax.axvline(x=2, color='red', linestyle='--', alpha=0.5)
                ax.axvline(x=-2, color='red', linestyle='--', alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                abnormal = np.where(np.abs(z_scores) > 2)[0]
                if len(abnormal) > 0:
                    st.warning(f"⚠️ {len(abnormal)} features deviate >2σ")
                else:
                    st.success("✅ Distribution normal")
            else:
                st.info("No training stats")

    # -------------------- Lower Detail Table (Drift Only) --------------------
    st.markdown("---")
    if z_scores is not None:
        with st.expander("📐 View Data Drift Details"):
            drift_df = pd.DataFrame({
                'Feature': FEATURE_NAMES,
                'Train_Mean': np.round(TRAIN_STATS['mean'].values, 3),
                'Train_Std': np.round(TRAIN_STATS['std'].values, 3),
                'Current_Value': np.round(x_raw, 3),
                'Z_Score': np.round(z_scores, 2)
            }).sort_values('Z_Score', key=abs, ascending=False)
            st.dataframe(drift_df, use_container_width=True, hide_index=True)

    # -------------------- Export --------------------
    st.markdown("---")
    st.subheader("💾 Export Results")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Predictions')
        output.seek(0)
        st.download_button(
            label="📥 Download Predictions (.xlsx)",
            data=output,
            file_name="insomnia_cluster_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col_dl2:
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions (.csv)",
            data=csv,
            file_name="insomnia_cluster_predictions.csv",
            mime="text/csv"
        )
    
    # ==================== NEW: Footer Disclaimer ====================
    st.markdown("---")
    st.markdown(f"""
    <div class="disclaimer-box">
        {DISCLAIMER_TEXT}
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👈 Please upload a data file in the sidebar, or download the example template to start predicting.", icon="⬆️")
    st.markdown("""
    ### Input File Format Requirements
    | Requirement | Description |
    |-------------|-------------|
    | Format | `.xlsx` or `.csv` |
    | Feature Columns | Must match training data features exactly (total **11** columns) |
    | Target Column | Optional. If the last column is a label, it will be ignored automatically |
    | Missing Values | Automatically filled with 0 |
    | Outliers | Infinity (Inf) will be replaced with 0 |
    """
    )
    
    # ==================== NEW: Disclaimer on empty state ====================
    st.markdown("---")
    st.markdown(f"""
    <div class="disclaimer-box">
        {DISCLAIMER_TEXT}
    </div>
    """, unsafe_allow_html=True)
