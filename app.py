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
</style>
""", unsafe_allow_html=True)

# ==================== Sidebar ====================
with st.sidebar:
    st.title("🧠 Insomnia Subtype Classifier")
    st.markdown("""
    **Based on Overnight HRV and Clinical Features**  
    Using a Logistic Regression model trained with nested cross-validation for automatic four-subtype classification.
    """)
    st.markdown("---")
    st.markdown("### Usage Steps")
    st.markdown("1. Upload an Excel/CSV file containing ECG features")
    st.markdown("2. System automatically preprocesses and aligns features")
    st.markdown("3. View prediction probabilities and subtype labels")
    st.markdown("4. Download data table with predictions")
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

# ==================== Main Interface ====================
st.title("Insomnia Patient Subtype Prediction Platform")
st.caption("Insomnia Subtype Classification via Resting-State EEG Functional Connectivity")

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
            pred, proba, class_names = predict(X_processed)
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
    for i, name in enumerate(class_names):
        result_df[f'Prob_{name}'] = np.round(proba[:, i], 4)

    display_cols = [f'Prob_{c}' for c in class_names]

    # -------------------- Parallel Layout --------------------
    st.markdown("---")
    
    col_table, col_right = st.columns([1, 2.5])
    
    # ===== Left: Prediction Results =====
    with col_table:
        st.markdown("#### 📊 Prediction Results")
        
        format_dict = {f'Prob_{c}': "{:.2%}" for c in class_names}
        try:
            styled = result_df[display_cols].style.background_gradient(
                subset=[f'Prob_{c}' for c in class_names],
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
        
        # First Row: Selector (Left) + Subtype Label (Right)
        c_select, c_label = st.columns(2)
        with c_select:
            selected_idx = st.selectbox("Select sample for details", result_df['Sample_ID'].tolist())
            # Calculate common variables for right side immediately
            sample_pos = row_ids.index(selected_idx)
            pred_label = int(pred[sample_pos])
            x_raw = X_processed[sample_pos].copy()
            x_std = SCALER.transform(X_processed[sample_pos:sample_pos+1])[0]
        
        with c_label:
            subtype_name = class_names[pred_label]
            st.markdown(f"""
            <div style="padding: 6px 0; width: 100%; text-align: center; border-radius: 6px; background-color: #1976d2; color: white; font-weight: bold; font-size: 14px; margin-top: 28px;">
                {subtype_name}
            </div>
            """, unsafe_allow_html=True)
        
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
            file_name="insomnia_subtype_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col_dl2:
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions (.csv)",
            data=csv,
            file_name="insomnia_subtype_predictions.csv",
            mime="text/csv"
        )

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
    """)
