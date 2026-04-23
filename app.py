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

# ==================== 加载模型与元数据 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pipeline（StandardScaler + LogReg）
PIPELINE = joblib.load(os.path.join(BASE_DIR, 'model', 'model.pkl'))
SCALER = PIPELINE.named_steps['scaler']
MODEL = PIPELINE.named_steps['model']

# 模型系数与截距（多分类 shape: n_classes × n_features）
COEF = MODEL.coef_
INTERCEPT = MODEL.intercept_

with open(os.path.join(BASE_DIR, 'model', 'feature_names.json'), 'r', encoding='utf-8') as f:
    FEATURE_NAMES = json.load(f)

with open(os.path.join(BASE_DIR, 'model', 'class_labels.json'), 'r', encoding='utf-8') as f:
    CLASS_INFO = json.load(f)

# 训练分布统计（用于漂移检测）
train_stats_path = os.path.join(BASE_DIR, 'model', 'train_stats.json')
TRAIN_STATS = pd.read_json(train_stats_path) if os.path.exists(train_stats_path) else None

# SHAP 背景数据（标准化后的训练样本，用于定义基线）
background_path = os.path.join(BASE_DIR, 'model', 'background.npy')
BACKGROUND = np.load(background_path) if os.path.exists(background_path) else None

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="失眠亚型预测系统",
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

# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("🧠 失眠亚型分类器")
    st.markdown("""
    **基于静息态心电功能连接特征**  
    使用嵌套交叉验证训练的 Logistic Regression 模型，实现四亚型自动判别。
    """)
    st.markdown("---")
    st.markdown("### 使用步骤")
    st.markdown("1. 上传包含心电特征的 Excel/CSV 文件")
    st.markdown("2. 系统自动预处理并对齐特征")
    st.markdown("3. 查看预测概率与亚型标签")
    st.markdown("4. 下载带预测结果的数据表")
    st.markdown("---")
    
    example_path = os.path.join(BASE_DIR, 'model', 'example_input.xlsx')
    if os.path.exists(example_path):
        with open(example_path, 'rb') as f:
            st.download_button(
                label="📥 下载示例输入模板",
                data=f,
                file_name="example_input.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    st.markdown("---")
    st.info(f"模型特征数：{len(FEATURE_NAMES)}", icon="ℹ️")

# ==================== 主界面 ====================
st.title("失眠症患者亚型预测平台")
st.caption("Insomnia Subtype Classification via Resting-State EEG Functional Connectivity")

# 免责声明
with st.expander("⚠️ 免责声明（点击展开）", expanded=False):
    st.markdown("""
    <div style="padding:15px;border-radius:8px;background-color:#fff3e0;border-left:5px solid #f57c00;">
    <b>本工具仅供科研参考，不替代临床诊断。</b><br>
    预测结果基于机器学习模型，实际临床决策需结合医生专业判断。
    </div>
    """, unsafe_allow_html=True)

# 文件上传
uploaded_file = st.file_uploader(
    "📤 上传待预测数据（.xlsx 或 .csv）",
    type=["xlsx", "csv"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # -------------------- 读取数据 --------------------
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"文件读取失败：{e}")
        st.stop()

    st.success(f"✅ 成功读取 {len(df_raw)} 行 × {df_raw.shape[1]} 列")

    # -------------------- 预处理与预测 --------------------
    with st.spinner("正在进行特征对齐、标准化与模型推理..."):
        try:
            X_processed, row_ids = preprocess_input(df_raw)
            pred, proba, class_names = predict(X_processed)
        except ValueError as ve:
            st.error(str(ve))
            st.stop()
        except Exception as e:
            st.error(f"预测过程出错：{e}")
            st.stop()

    # -------------------- 构建结果表 --------------------
    result_df = pd.DataFrame({
        'Sample_ID': row_ids,
        'Predicted_Subtype': [class_names[p] for p in pred],
        'Predicted_Label': pred
    })
    for i, name in enumerate(class_names):
        result_df[f'Prob_{name}'] = np.round(proba[:, i], 4)
    result_df['Max_Prob'] = result_df[[f'Prob_{c}' for c in class_names]].max(axis=1)

    # -------------------- 全局结果表格 --------------------
    st.markdown("---")
    st.subheader("📊 预测结果总表")
    
    try:
        st.dataframe(
            result_df.style.background_gradient(
                subset=[f'Prob_{c}' for c in class_names],
                cmap='YlGnBu',
                vmin=0, vmax=1
            ),
            use_container_width=True,
            height=min(400, 35 * len(result_df) + 50)
        )
    except Exception:
        st.dataframe(result_df, use_container_width=True)

    # -------------------- 单样本深度解析 --------------------
    st.markdown("---")
    st.subheader("🔬 单样本深度解析")

    selected_idx = st.selectbox("选择样本查看详情", result_df['Sample_ID'].tolist())
    sample_pos = row_ids.index(selected_idx)
    pred_label = int(result_df[result_df['Sample_ID'] == selected_idx]['Predicted_Label'].values[0])
    
    # 原始特征值（来自 preprocess_input 的输出，尚未标准化）
    x_raw = X_processed[sample_pos].copy()
    # 标准化后的特征值（与 SHAP 背景数据同空间）
    x_std = SCALER.transform(X_processed[sample_pos:sample_pos+1])[0]

    tab1, tab2, tab3 = st.tabs(["📊 概率概览", "🔍 SHAP 解释", "⚖️ 数据漂移"])

    # ===== Tab 1: 概率概览 =====
    with tab1:
        cols = st.columns(4)
        for i, name in enumerate(class_names):
            prob = result_df[result_df['Sample_ID'] == selected_idx][f'Prob_{name}'].values[0]
            with cols[i]:
                st.metric(label=name, value=f"{prob:.1%}")

    # ===== Tab 2: 真正的 SHAP waterfall 解释 =====
    with tab2:
        st.markdown(f"**预测亚型：{class_names[pred_label]}**")
        st.caption("基于 SHAP LinearExplainer 的精确特征贡献分解（以训练分布为基线）")

        if BACKGROUND is not None:
            # 创建 Explainer 并计算 SHAP 值
            shap_explainer = shap.Explainer(MODEL, BACKGROUND)
            sv = shap_explainer(x_std.reshape(1, -1))

            # 构造当前预测类别的 shap.Explanation 对象（waterfall 必需）
            exp = shap.Explanation(
                values=sv.values[0, :, pred_label],
                base_values=sv.base_values[0, pred_label],
                data=x_std,
                feature_names=FEATURE_NAMES
            )

            # 绘制 waterfall
            fig = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(exp, max_display=10, show=False)
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)

            # SHAP 数值明细表
            with st.expander("📋 查看 SHAP 数值明细（展开）"):
                shap_df = pd.DataFrame({
                    'Feature': FEATURE_NAMES,
                    'Raw_Value': np.round(x_raw, 3),
                    'Std_Value': np.round(x_std, 3),
                    'SHAP_Value': np.round(sv.values[0, :, pred_label], 4),
                    'Direction': ['🔴 推动预测' if v > 0 else '🔵 抑制预测' 
                                  for v in sv.values[0, :, pred_label]]
                }).sort_values('SHAP_Value', key=abs, ascending=False)
                st.dataframe(shap_df, use_container_width=True, hide_index=True)
        else:
            st.warning("未找到背景数据文件 (background.npy)，无法生成 SHAP waterfall。")

        # 模型系数参考（与 SHAP 互补）
        with st.expander("📐 查看模型系数（LogReg 参数，展开）"):
            coef_df = pd.DataFrame({
                'Feature': FEATURE_NAMES,
                'Coefficient': np.round(COEF[pred_label], 4),
                'Abs_Coef': np.round(np.abs(COEF[pred_label]), 4)
            }).sort_values('Abs_Coef', ascending=False)
            st.dataframe(coef_df, use_container_width=True, hide_index=True)

    # ===== Tab 3: 数据漂移检测 =====
    with tab3:
        if TRAIN_STATS is not None:
            means = TRAIN_STATS['mean'].values
            stds = TRAIN_STATS['std'].values
            z_scores = (x_raw - means) / stds

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#d62728' if abs(z) > 2 else '#ff7f0e' if abs(z) > 1 else '#2ca02c' 
                      for z in z_scores]
            ax.barh(range(len(FEATURE_NAMES)), z_scores, color=colors, 
                    edgecolor='black', linewidth=0.5)
            ax.set_yticks(range(len(FEATURE_NAMES)))
            ax.set_yticklabels(FEATURE_NAMES)
            ax.invert_yaxis()
            ax.set_xlabel("Z-score (vs Training Distribution)", fontsize=11)
            ax.set_title("Data Drift: Feature Deviation from Training Mean", 
                        fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=-1, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=2, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=-2, color='red', linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # 异常警告
            abnormal = np.where(np.abs(z_scores) > 2)[0]
            if len(abnormal) > 0:
                st.warning(
                    f"⚠️ 以下特征偏离训练分布 >2σ，预测可靠性可能下降："
                    f"{', '.join([FEATURE_NAMES[i] for i in abnormal])}"
                )
            else:
                st.success("✅ 所有特征均在训练分布正常范围内（±2σ）")
            
            # 数值表
            drift_df = pd.DataFrame({
                'Feature': FEATURE_NAMES,
                'Train_Mean': np.round(means, 3),
                'Train_Std': np.round(stds, 3),
                'Current_Value': np.round(x_raw, 3),
                'Z_Score': np.round(z_scores, 2)
            }).sort_values('Z_Score', key=abs, ascending=False)
            st.dataframe(drift_df, use_container_width=True, hide_index=True)
        else:
            st.info("未找到训练分布统计文件 (train_stats.json)，无法执行漂移检测。")

    # -------------------- 导出 --------------------
    st.markdown("---")
    st.subheader("💾 导出结果")

    c1, c2 = st.columns(2)
    with c1:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Predictions')
        output.seek(0)
        st.download_button(
            label="📥 下载预测结果 (.xlsx)",
            data=output,
            file_name="insomnia_subtype_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with c2:
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载预测结果 (.csv)",
            data=csv,
            file_name="insomnia_subtype_predictions.csv",
            mime="text/csv"
        )

else:
    st.info("👈 请在左侧上传数据文件，或下载示例模板以开始预测。", icon="⬆️")
    st.markdown("""
    ### 输入文件格式要求
    | 要求 | 说明 |
    |------|------|
    | 格式 | `.xlsx` 或 `.csv` |
    | 特征列 | 必须与训练数据特征完全一致（共 **11** 列） |
    | 目标列 | 可选。若最后一列为标签，系统会自动忽略 |
    | 缺失值 | 系统自动填充为 0 |
    | 异常值 | 无穷大(Inf) 会被替换为 0 |
    """)
