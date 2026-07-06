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
        "tagline": "早期代偿性失眠",
        "autonomic_summary": "自主神经复杂性相对保留，多尺度熵（MAE_1）升高，MeanNN增加，表明维持了调节灵活性以及较慢时间尺度的适应过程。",
        "symptom_description": "年龄分布最年轻，临床症状负担相对较轻。PSQI和ISI可能轻度升高。焦虑-抑郁评分（SAS、SDS）通常低于其他聚类。日间疲劳（FSS）为中等程度。",
        "network_signature": "与健康对照组相比，无广泛病理性低连接。相反，显示出13条一致的ALPHA频带增强连接，集中在双侧小脑内部连接、小脑-丘脑通路以及海马-顶上小叶回路。这表明是代偿性重组而非退行性改变。",
        "cross_scale_coupling": "能量熵-腹侧纹状体-丘脑关联：外周自主神经复杂性（AttentionEntropy, MAE_1）与腹侧纹状体-丘脑连接增强相关，提示奖赏-动机回路对轻度外周自主神经变化的代偿作用。",
        "clinical_implication": "可能代表失眠病理生理学的早期或代偿阶段。保护性机制（增强的自主神经复杂性和小脑-皮层下连接）维持了相对完整的日间功能。可能受益于针对睡眠卫生和昼夜节律稳定的非药物干预。"
    },
    1: {
        "tagline": "失代偿性过度觉醒 — 最严重内型",
        "autonomic_summary": "广泛的HRV抑制：SDNN、RMSSD、pNN50、SD1和AttentionEntropy降低。整体自主神经变异性降低，副交感相关短期波动减少。生理复杂性受损，提示自主神经调节储备耗竭。",
        "symptom_description": "所有六个临床量表（PSQI、ISI、SAS、SDS、HAS、FSS均为聚类中最高）得分最高。镇静催眠药物依赖性最高。躯体症状风险升高：胸闷、头痛、疲劳、口干、健忘、易怒、心悸。",
        "network_signature": "最严重的中枢网络损伤：α频带32条一致性降低连接，无增强连接——纯失代偿性低连接。广泛的小脑-丘脑解耦、小脑-基底节损伤、小脑-躯体运动功能障碍以及皮层下-脑干断连。与健康对照组相比，跨频带（δ、α、β）系统性低连接。",
        "cross_scale_coupling": "长期变异性-小脑-丘脑关联：SDANN与小脑小叶X-丘脑VPL连接呈负相关（LOOCV R²=0.361），提示通过小脑-丘脑通路的自主神经反馈环路受损，心率长期变异性降低。",
        "clinical_implication": "可能对应临床实践中最难治的失眠人群——'失代偿性过度觉醒'，即慢性中枢过度觉醒已耗竭自主神经储备。需要最密集的干预，可能需将药物治疗与针对小脑-丘脑-皮层轴的神经营养调节方法相结合。"
    },
    2: {
        "tagline": "高反应性自主神经内型",
        "autonomic_summary": "独特的'高反应性、高离散度'模式：RMSSD、pNN50、SD1、SD2、SD1/SD2、C2d及多个信息熵指标同步升高。这代表心跳间期波动幅度异常扩大，自主神经稳定性降低——表现为反应增益上调而非调节能力增强。",
        "symptom_description": "临床严重程度居中。表型特征：面部皮肤油腻（可能为交感-皮脂腺轴过度活跃）。PSQI、ISI、SAS、SDS中度升高。HAS（过度觉醒）显著升高。",
        "network_signature": "'局部重组，整体保留'：与健康对照组相比无显著全脑网络差异。一条稳定增强连接：右侧小脑小叶VI-左侧顶上小叶（小脑-背侧注意网络界面），提示注意控制系统失调。",
        "cross_scale_coupling": "复杂性-纹状体-小脑关联：近似熵与右侧小脑小叶X-壳核连接呈现预测关系，提示基底节-小脑回路介导的觉醒维持异常。RMSSD与小脑Crus 2-眶额叶连接相关。",
        "clinical_implication": "外周自主神经过度反应可能代表持续性中枢威胁监测系统激活的外周投射。交感-皮脂腺轴表型独特。针对交感神经过度活跃的干预（如放松训练、心率变异性生物反馈）可能有效。"
    },
    3: {
        "tagline": "外周低偏差 — 中枢整合保留",
        "autonomic_summary": "时域、频域和非线性指标均为最低HRV表型，可能反映年龄相关的自主神经调节衰退叠加在失眠病理上。睡眠稳态和昼夜节律系统的自然老化促成了外周表型。",
        "symptom_description": "最年长患者组，主观症状负担最轻。ISI、PSQI、HAS、SDS显著但适度升高（相对于健康对照组）。FSS（疲劳）相对较低。与Cluster 1和2相比，过度觉醒评分（HAS）较低。",
        "network_signature": "'外周低偏差，中枢整合保留'：与健康对照组相比无显著全脑网络差异。一条一致性增强连接：左侧小脑Crus I-Crus II内部连接——后部小脑认知-情绪处理网络内的局部功能重组，局限于小脑，无跨区域扩展。",
        "cross_scale_coupling": "非线性动力学-苍白球-小脑关联：BubbleEntropy与双侧苍白球-小脑连接相关。ISI与小脑小叶X-丘脑VPL连接呈正相关，提示老年患者的主观失眠严重程度可能与躯体感觉处理通路功能有关——异常的躯体感觉整合（如身体不适超敏反应）可能是主观痛苦的重要来源。",
        "clinical_implication": "表型可能主要由自主神经衰退驱动，而非强烈的病理性过度觉醒。临床评估应谨慎区分病理性自主神经变化与生理性衰老。干预应优先考虑适龄方法（如失眠认知行为疗法、光疗），而非积极的药物镇静。"
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
    **Disclaimer:** This tool is for **research purposes only** and does not constitute medical advice. 
    Clinical decisions should not be made solely based on this platform's outputs. 
    Always consult qualified healthcare professionals.
    """)


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
        
        st.markdown("---")
        
        # ===== SHAP (Left) + Data Drift (Right) =====
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
        
        # ==================== Subgroup Population Characteristics (Moved to bottom) ====================
        st.markdown("---")
        st.markdown("#### 🧬 Subgroup Population Characteristics")
        
        profile = CLUSTER_PROFILES[pred_label]
        
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
    
    # ==================== Footer Disclaimer (without div) ====================
    st.markdown("---")
    st.markdown(DISCLAIMER_TEXT)

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
    
    # ==================== Disclaimer on empty state (without div) ====================
    st.markdown("---")
    st.markdown(DISCLAIMER_TEXT)
